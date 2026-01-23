"""Utilities for extracting gene modules from adjacency matrices.

This module provides Leiden and spectral clustering helpers along with
convenience utilities to compute module eigengenes and persist results to disk.

Example
-------
>>> import pandas as pd
>>> from bsvae.networks.module_extraction import load_adjacency, leiden_modules, compute_module_eigengenes
>>> adjacency, genes = load_adjacency("adjacency.csv")
>>> modules = leiden_modules(adjacency)
>>> expr = pd.read_csv("expression.csv", index_col=0)  # genes x samples
>>> eigengenes = compute_module_eigengenes(expr, modules)
>>> save_modules(modules, "modules.csv")
>>> save_eigengenes(eigengenes, "eigengenes.csv")
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bsvae.networks.utils import transform_adjacency_for_clustering
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _infer_separator(path: str) -> str:
    # Handle compressed files
    suffixes = Path(path).suffixes
    for suffix in suffixes:
        if suffix.lower() == ".tsv":
            return "\t"
    return ","


def _ensure_array_and_genes(A: np.ndarray | pd.DataFrame, genes: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, List[str]]:
    if isinstance(A, pd.DataFrame):
        genes = list(A.index)
        arr = A.values
    else:
        arr = np.asarray(A)
        if genes is None:
            genes = [str(i) for i in range(arr.shape[0])]
        else:
            genes = list(genes)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    return arr, genes


def load_adjacency(path: str, genes_path: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix with gene labels.

    Supports multiple formats:
    - CSV/TSV files with genes as index and columns (legacy dense format)
    - NPZ files with sparse matrix storage (compressed sparse format)
    - Parquet edge lists (.parquet) - recommended format
    - Gzipped edge lists (.csv.gz/.tsv.gz) with optional gene lookup (deprecated)

    Parameters
    ----------
    path:
        Path to adjacency file. Format is auto-detected from extension.
    genes_path:
        Optional path to gene lookup file for legacy gzipped edge list format.

    Returns
    -------
    adjacency : np.ndarray
        Square matrix of edge weights.
    genes : list[str]
        Gene identifiers derived from the file.
    """
    path_obj = Path(path)
    suffixes = [s.lower() for s in path_obj.suffixes]

    # Handle sparse NPZ format
    if ".npz" in suffixes:
        from bsvae.networks.extract_networks import load_adjacency_sparse
        return load_adjacency_sparse(path)

    # Handle Parquet edge list format (recommended)
    if ".parquet" in suffixes:
        from bsvae.networks.extract_networks import load_edge_list_parquet
        return load_edge_list_parquet(path)

    # Handle compressed edge list format (deprecated)
    if ".gz" in suffixes:
        import warnings
        warnings.warn(
            "Loading from gzipped edge list is deprecated. "
            "Use Parquet format (.parquet) for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        from bsvae.networks.extract_networks import load_edge_list_compressed
        return load_edge_list_compressed(path, genes_path)

    # Legacy dense CSV/TSV format
    sep = _infer_separator(path)
    df = pd.read_csv(path, sep=sep)

    gene_col = df.columns[0]
    if gene_col == "" or gene_col.startswith("Unnamed"):
        df = df.rename(columns={gene_col: "gene"}).set_index("gene")
    else:
        df = df.set_index(gene_col)

    if df.shape[0] != df.shape[1]:
        raise ValueError("Adjacency file must be square with genes as both index and columns")
    if not df.columns.equals(df.index):
        raise ValueError("Adjacency file must have matching gene labels for rows and columns")

    arr = df.values.astype(float, copy=False)
    if not np.allclose(arr, arr.T):
        raise ValueError("Adjacency matrix must be symmetric")

    logger.info("Loaded adjacency from %s with %d genes", path, df.shape[0])
    return arr, list(df.index)


def _module_size_summary(modules: Mapping[str, int] | pd.Series) -> Tuple[int, int]:
    """Return the number of modules and median module size.

    Parameters
    ----------
    modules:
        Mapping from gene to module label or a pandas Series indexed by gene.

    Returns
    -------
    tuple
        ``(n_modules, median_size)`` where ``median_size`` is 0 when no modules
        are present.
    """

    module_series = pd.Series(modules, name="module")
    counts = module_series.value_counts()
    if counts.empty:
        return 0, 0
    return module_series.nunique(), int(np.median(counts.values))


def format_module_feedback(
    method: str,
    modules: Mapping[str, int] | pd.Series,
    *,
    resolution: Optional[float] = None,
    n_clusters: Optional[int] = None,
) -> str:
    """Human-friendly summary of module extraction results.

    Examples
    --------
    >>> modules = pd.Series([0, 0, 1, 1, 1], index=["g1", "g2", "g3", "g4", "g5"])
    >>> format_module_feedback("Leiden", modules, resolution=1.0)
    'Leiden resolution=1.0 produced 2 modules (median size=2 genes)'
    """

    n_modules, median_size = _module_size_summary(modules)
    details = []
    if resolution is not None:
        details.append(f"resolution={resolution}")
    if n_clusters is not None:
        details.append(f"n_clusters={n_clusters}")
    detail_str = f" {'; '.join(details)}" if details else ""
    return f"{method}{detail_str} produced {n_modules} modules (median size={median_size} genes)"


def build_graph_from_adjacency(A: np.ndarray | pd.DataFrame, genes: Optional[Sequence[str]] = None):
    """Construct an igraph Graph from an adjacency matrix.

    Parameters
    ----------
    A:
        Adjacency matrix as ``numpy.ndarray`` or ``pandas.DataFrame``.
    genes:
        Optional gene identifiers. If ``A`` is a DataFrame, its index is used.

    Returns
    -------
    igraph.Graph
        Undirected weighted graph with gene names as vertex attributes.
    """

    try:
        import igraph as ig
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("igraph is required for graph construction") from exc

    arr, genes = _ensure_array_and_genes(A, genes)
    arr = np.array(arr, dtype=float)
    np.fill_diagonal(arr, 0.0)
    graph = ig.Graph.Weighted_Adjacency(arr.tolist(), mode="UNDIRECTED", attr="weight", loops=False)
    graph.vs["name"] = genes
    return graph


def leiden_modules(
    A: np.ndarray | pd.DataFrame,
    resolution: float = 1.0,
    *,
    adjacency_mode: str = "wgcna-signed",
) -> pd.Series:
    """Cluster genes into modules using Leiden community detection.

    Parameters
    ----------
    A:
        Adjacency matrix (numpy array or pandas DataFrame). If a DataFrame, the
        index is used as gene names.
    resolution:
        Resolution parameter for Leiden (higher values produce more clusters).

    Notes
    -----
    Leiden does not support negative edge weights. By default, BSVAE uses a
    WGCNA-style signed network where negative edges are clipped to zero before
    clustering.

    Returns
    -------
    pandas.Series
        Module assignments indexed by gene identifiers.
    """

    try:
        import leidenalg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("leidenalg is required for Leiden clustering") from exc

    arr, genes = _ensure_array_and_genes(A)
    logger.info(
        "Running Leiden clustering on %d genes (resolution=%.3f, adjacency_mode=%s)",
        len(genes),
        resolution,
        adjacency_mode,
    )
    adj = transform_adjacency_for_clustering(arr, mode=adjacency_mode)

    # TODO: support signed community detection without violating Leiden's
    # non-negative edge weight requirement. Do NOT attempt signed Leiden
    # clustering or silently rescale/abs/drop edges.
    if adjacency_mode == "signed":
        raise NotImplementedError(
            "Signed community detection is not yet supported. "
            "Leiden requires non-negative edge weights. "
            "Use adjacency_mode='wgcna-signed'."
        )

    if np.any(adj < 0):
        raise ValueError(
            "Negative weights detected after adjacency transform. "
            "This should not occur for wgcna-signed mode."
        )

    graph = build_graph_from_adjacency(adj, genes)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights=graph.es["weight"],
    )
    modules = pd.Series(partition.membership, index=genes, name="module")
    logger.info(format_module_feedback("Leiden", modules, resolution=resolution))
    return modules


def spectral_modules(
    A: np.ndarray | pd.DataFrame,
    n_clusters: Optional[int] = None,
    n_components: Optional[int] = None,
) -> pd.Series:
    """Cluster genes using spectral clustering on the adjacency Laplacian.

    Parameters
    ----------
    A:
        Adjacency matrix (numpy array or pandas DataFrame).
    n_clusters:
        Number of clusters. Defaults to ``max(2, sqrt(n_genes))`` when ``None``.
    n_components:
        Number of eigenvectors to use. Defaults to ``n_clusters``.

    Returns
    -------
    pandas.Series
        Module assignments indexed by gene identifiers.
    """

    arr, genes = _ensure_array_and_genes(A)
    n_genes = len(genes)
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(n_genes)))
    if n_components is None:
        n_components = n_clusters
    logger.info("Running spectral clustering with %d clusters", n_clusters)

    arr = np.array(arr, dtype=float)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        n_components=n_components,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
    )
    labels = clustering.fit_predict(arr)
    modules = pd.Series(labels, index=genes, name="module")
    logger.info(format_module_feedback("Spectral", modules, n_clusters=n_clusters))
    return modules


def compute_module_eigengenes(datExpr: pd.DataFrame, modules: Mapping[str, int]) -> pd.DataFrame:
    """Compute module eigengenes (first principal component per module).

    Parameters
    ----------
    datExpr:
        Gene expression DataFrame with genes as rows and samples as columns.
    modules:
        Mapping from gene identifier to module assignment.

    Returns
    -------
    pandas.DataFrame
        Samples × modules matrix of eigengene values.
    """

    module_series = pd.Series(modules, name="module")
    shared_genes = module_series.index.intersection(datExpr.index)
    logger.debug(
        "Expression genes: %d | Module genes: %d | Overlap: %d",
        datExpr.shape[0],
        len(module_series),
        len(shared_genes),
    )
    if shared_genes.empty:
        raise ValueError("No overlapping genes between expression matrix and modules")

    overlap_fraction = len(shared_genes) / len(module_series) if len(module_series) else 0
    if overlap_fraction < 0.9:
        logger.warning(
            "Only %.1f%% of module genes overlap with expression matrix", overlap_fraction * 100
        )

    logger.info("Computing eigengenes for %d modules", module_series.nunique())
    eigengenes = {}
    samples = datExpr.columns

    for module_id, genes in module_series.groupby(module_series):
        gene_list = list(genes.index.intersection(datExpr.index))
        if not gene_list:
            logger.warning("Module %s has no genes in expression matrix; skipping", module_id)
            continue
        expr_subset = datExpr.loc[gene_list].T  # samples x genes
        scaler = StandardScaler()
        scaled = scaler.fit_transform(expr_subset)
        pca = PCA(n_components=1)
        comp = pca.fit_transform(scaled)
        eigengenes[str(module_id)] = comp[:, 0]

    eigengene_df = pd.DataFrame(eigengenes, index=samples)
    eigengene_df.index.name = "sample_id"
    logger.info("Computed eigengenes for %d modules", eigengene_df.shape[1])
    return eigengene_df


def save_modules(modules: Mapping[str, int] | pd.Series, output_path: str) -> None:
    """Save gene-to-module assignments to CSV.

    Parameters
    ----------
    modules:
        Mapping from gene to module label or a pandas Series.
    output_path:
        Destination CSV/TSV path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module_series = pd.Series(modules, name="module")
    df = module_series.reset_index()
    df.columns = ["gene", "module"]
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df.to_csv(path, index=False, sep=sep)
    logger.info("Saved %d module assignments to %s", df.shape[0], path)


def save_eigengenes(eigengenes: pd.DataFrame, output_path: str) -> None:
    """Persist eigengenes matrix to disk.

    Parameters
    ----------
    eigengenes:
        Samples × modules DataFrame produced by :func:`compute_module_eigengenes`.
    output_path:
        Destination CSV/TSV path.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    eigengenes.to_csv(path, sep=sep)
    logger.info("Saved eigengenes matrix with %d samples to %s", eigengenes.shape[0], path)


__all__ = [
    "load_adjacency",
    "build_graph_from_adjacency",
    "format_module_feedback",
    "leiden_modules",
    "spectral_modules",
    "compute_module_eigengenes",
    "save_modules",
    "save_eigengenes",
]
