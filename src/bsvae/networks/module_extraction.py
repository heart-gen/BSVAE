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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

from bsvae.networks.utils import transform_adjacency_for_clustering
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import igraph as ig


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


def load_adjacency(
    path: str,
    genes_path: Optional[str] = None,
    *,
    return_type: str = "dense",
) -> Tuple[np.ndarray, List[str]] | Tuple[np.ndarray, np.ndarray, List[str]] | Tuple["ig.Graph", List[str]]:
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

    return_type:
        ``"dense"`` (default) returns a dense adjacency array. ``"edges"``
        returns ``(edges, weights, genes)`` without densifying, and ``"graph"``
        returns an ``igraph.Graph`` built from the edge list when possible.

    Returns
    -------
    adjacency : np.ndarray | (edges, weights, genes) | igraph.Graph
        Loaded adjacency representation.
    genes : list[str]
        Gene identifiers derived from the file (returned separately for ``graph``).
    """
    path_obj = Path(path)
    suffixes = [s.lower() for s in path_obj.suffixes]

    # Handle sparse NPZ format
    if ".npz" in suffixes:
        if return_type in {"edges", "graph"}:
            from bsvae.networks.extract_networks import load_adjacency_sparse_edges
            edges, weights, genes = load_adjacency_sparse_edges(path)
            if return_type == "graph":
                return build_graph_from_edge_list(edges, weights, genes), genes
            return edges, weights, genes
        from bsvae.networks.extract_networks import load_adjacency_sparse
        return load_adjacency_sparse(path)

    # Handle Parquet edge list format (recommended)
    if ".parquet" in suffixes:
        if return_type in {"edges", "graph"}:
            from bsvae.networks.extract_networks import load_edge_list_parquet_edges
            edges, weights, genes = load_edge_list_parquet_edges(path)
            if return_type == "graph":
                return build_graph_from_edge_list(edges, weights, genes), genes
            return edges, weights, genes
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
        if return_type in {"edges", "graph"}:
            from bsvae.networks.extract_networks import load_edge_list_compressed_edges
            edges, weights, genes = load_edge_list_compressed_edges(path, genes_path)
            if return_type == "graph":
                return build_graph_from_edge_list(edges, weights, genes), genes
            return edges, weights, genes
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
    if return_type == "graph":
        graph = build_graph_from_adjacency(arr, list(df.index))
        return graph, list(df.index)
    if return_type == "edges":
        n = arr.shape[0]
        triu_idx = np.triu_indices(n, k=1)
        values = arr[triu_idx]
        nonzero_mask = values != 0
        edges = np.column_stack((triu_idx[0][nonzero_mask], triu_idx[1][nonzero_mask]))
        weights = values[nonzero_mask].astype(float)
        return edges, weights, list(df.index)
    return arr, list(df.index)


def _normalize_edge_list(
    edges: np.ndarray | Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(edges)
    if arr.ndim != 2 or arr.shape[1] not in {2, 3}:
        raise ValueError("Edges must be a 2D array with 2 or 3 columns")
    if arr.shape[1] == 3 and weights is None:
        weights_arr = arr[:, 2].astype(float)
        arr = arr[:, :2]
    else:
        weights_arr = None if weights is None else np.asarray(weights, dtype=float)
    return arr.astype(np.int64), weights_arr


def build_graph_from_edge_list(
    edges: np.ndarray | Sequence[Sequence[float]],
    weights: Optional[Sequence[float]] = None,
    genes: Optional[Sequence[str]] = None,
):
    """Construct an igraph Graph from an edge list."""
    import igraph as ig

    edges_arr, weight_arr = _normalize_edge_list(edges, weights)
    if genes is None:
        n_vertices = int(edges_arr.max() + 1) if edges_arr.size else 0
        genes = [str(i) for i in range(n_vertices)]
    genes = list(genes)
    graph = ig.Graph(n=len(genes), edges=edges_arr.tolist(), directed=False)
    if weight_arr is not None:
        graph.es["weight"] = weight_arr.tolist()
    graph.vs["name"] = genes
    return graph


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

    import igraph as ig

    arr, genes = _ensure_array_and_genes(A, genes)
    arr = np.array(arr, dtype=float)
    np.fill_diagonal(arr, 0.0)
    graph = ig.Graph.Weighted_Adjacency(arr.tolist(), mode="UNDIRECTED", attr="weight", loops=False)
    graph.vs["name"] = genes
    return graph


def prepare_leiden_graph(
    A: np.ndarray | pd.DataFrame,
    genes: Optional[Sequence[str]] = None,
    *,
    adjacency_mode: str = "wgcna-signed",
):
    """Transform adjacency and build the Leiden graph once.

    Returns the igraph Graph and ordered gene list.
    """

    arr, genes = _ensure_array_and_genes(A, genes)
    logger.info(
        "Preparing Leiden graph for %d genes (adjacency_mode=%s)",
        len(genes),
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
    return graph, genes


def _leiden_partitions_for_resolutions(
    graph,
    resolutions: Iterable[float],
    *,
    progress: bool = False,
    progress_desc: str = "Leiden resolution search",
):
    """Run Leiden clustering for multiple resolutions.

    Parameters
    ----------
    graph:
        igraph Graph to cluster.
    resolutions:
        Iterable of resolution values to test.
    progress:
        If True, display a tqdm progress bar.
    progress_desc:
        Description string for the progress bar.

    Returns
    -------
    dict
        Mapping from resolution value to leidenalg partition.
    """
    import leidenalg

    edge_weights = graph.es["weight"] if "weight" in graph.es.attributes() else None
    resolutions_list = list(resolutions)
    partitions = {}

    iterator = resolutions_list
    if progress:
        iterator = tqdm(
            resolutions_list,
            desc=progress_desc,
            unit="res",
            leave=False,
        )

    for res in iterator:
        partitions[res] = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=res,
            weights=edge_weights,
        )
        if progress and hasattr(iterator, "set_postfix"):
            n_modules = len(partitions[res])
            iterator.set_postfix(resolution=f"{res:.2f}", modules=n_modules)

    return partitions


def leiden_modules_from_graph(graph, genes: Sequence[str], resolution: float) -> pd.Series:
    """Run Leiden clustering using a pre-built igraph Graph."""

    partition = _leiden_partitions_for_resolutions(graph, [resolution])[resolution]
    modules = pd.Series(partition.membership, index=list(genes), name="module")
    logger.info(format_module_feedback("Leiden", modules, resolution=resolution))
    return modules


def leiden_modules_for_resolutions(
    graph,
    genes: Sequence[str],
    resolutions: Iterable[float],
    *,
    progress: bool = False,
) -> Dict[float, pd.Series]:
    """Run Leiden clustering for multiple resolutions using a pre-built graph.

    Parameters
    ----------
    graph:
        Pre-built igraph Graph.
    genes:
        Gene names aligned to vertex indices.
    resolutions:
        Iterable of resolution values to test.
    progress:
        If True, display a progress bar.

    Returns
    -------
    dict
        Mapping from resolution to module assignments (pd.Series).
    """
    partitions = _leiden_partitions_for_resolutions(
        graph,
        resolutions,
        progress=progress,
        progress_desc="Leiden resolution sweep",
    )
    modules_by_resolution = {}
    for res, partition in partitions.items():
        modules_by_resolution[res] = pd.Series(partition.membership, index=list(genes), name="module")
    return modules_by_resolution


def optimize_resolution_modularity(
    A: np.ndarray | pd.DataFrame | None = None,
    *,
    adjacency_mode: str = "wgcna-signed",
    resolution_min: float = 0.5,
    resolution_max: float = 15.0,
    n_steps: int = 30,
    graph: Optional["ig.Graph"] = None,
    edges: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    genes: Optional[Sequence[str]] = None,
    transformed_adjacency: Optional[np.ndarray] = None,
    return_modules: bool = False,
    progress: bool = False,
    two_phase: bool = False,
    coarse_steps: int = 10,
    fine_steps: int = 10,
    fine_range_fraction: float = 0.3,
) -> tuple[float, float] | tuple[float, float, pd.Series]:
    """Find optimal Leiden resolution by maximizing modularity.

    This method selects resolution without using ground truth labels,
    making it suitable for unbiased benchmarking.

    Parameters
    ----------
    A:
        Adjacency matrix (numpy array or pandas DataFrame). Optional when
        providing ``graph`` or ``edges``.
    graph:
        Pre-built igraph Graph (optional).
    edges:
        Edge list of shape ``(n_edges, 2)`` or ``(n_edges, 3)`` (optional).
    weights:
        Edge weights aligned to ``edges`` (optional).
    genes:
        Gene names aligned to vertex indices (optional for ``edges``).
    transformed_adjacency:
        Optional adjacency matrix already transformed for Leiden clustering.
    adjacency_mode:
        How to handle negative edges ('wgcna-signed' clips to zero).
    resolution_min:
        Minimum resolution to search.
    resolution_max:
        Maximum resolution to search.
    n_steps:
        Number of resolution values to test. Ignored when ``two_phase=True``.
    return_modules:
        If True, also return the module assignments for the best resolution,
        avoiding redundant re-computation of Leiden clustering.
    progress:
        If True, display a progress bar during resolution search.
    two_phase:
        If True, use a memory-efficient two-phase search: a coarse sweep to
        find a ballpark resolution, then a fine sweep in a narrower range.
        This reduces peak memory by avoiding storing all partitions at once.
    coarse_steps:
        Number of resolution values in the coarse phase (default: 10).
        Only used when ``two_phase=True``.
    fine_steps:
        Number of resolution values in the fine phase (default: 10).
        Only used when ``two_phase=True``.
    fine_range_fraction:
        Fraction of the original range to use for fine search, centered on
        the best coarse resolution (default: 0.3 = 30% of original range).
        Only used when ``two_phase=True``.

    Returns
    -------
    tuple[float, float] or tuple[float, float, pd.Series]
        ``(best_resolution, best_modularity)`` if return_modules is False,
        otherwise ``(best_resolution, best_modularity, modules)``.
    """
    if adjacency_mode == "signed":
        raise NotImplementedError(
            "Signed community detection is not yet supported. "
            "Leiden requires non-negative edge weights. "
            "Use adjacency_mode='wgcna-signed'."
        )

    if graph is None:
        if edges is not None:
            edges_arr, weight_arr = _normalize_edge_list(edges, weights)
            if weight_arr is not None:
                weight_arr = np.maximum(weight_arr, 0.0)
                if np.any(weight_arr < 0):
                    raise ValueError("Negative weights detected after adjacency transform.")
            graph = build_graph_from_edge_list(edges_arr, weight_arr, genes)
            genes = list(graph.vs["name"])
        elif transformed_adjacency is not None:
            if genes is None:
                if A is None:
                    raise ValueError("Provide genes when supplying transformed_adjacency.")
                _, genes = _ensure_array_and_genes(A, genes)
            adj = np.asarray(transformed_adjacency)
            if np.any(adj < 0):
                raise ValueError("Negative weights detected after adjacency transform.")
            graph = build_graph_from_adjacency(adj, genes)
        else:
            if A is None:
                raise ValueError("Provide adjacency data, edges, or a graph.")
            graph, genes = prepare_leiden_graph(A, genes, adjacency_mode=adjacency_mode)
    else:
        if genes is None:
            genes = list(graph.vs["name"]) if "name" in graph.vs.attributes() else [str(i) for i in range(graph.vcount())]
        if "weight" in graph.es.attributes():
            weights_arr = np.array(graph.es["weight"], dtype=float)
            if np.any(weights_arr < 0):
                raise ValueError("Negative weights detected in provided graph.")
    if two_phase:
        # Two-phase search: coarse sweep then fine sweep around best
        # This reduces peak memory by not storing all partitions at once
        total_range = resolution_max - resolution_min

        # Phase 1: Coarse search
        coarse_resolutions = np.linspace(resolution_min, resolution_max, coarse_steps)
        logger.info(
            "Two-phase search: coarse phase in [%.2f, %.2f] with %d steps",
            resolution_min,
            resolution_max,
            coarse_steps,
        )

        coarse_best_res, coarse_best_qual = 1.0, -np.inf
        coarse_best_partition = None
        coarse_partitions = _leiden_partitions_for_resolutions(
            graph,
            coarse_resolutions,
            progress=progress,
            progress_desc="Coarse resolution search",
        )
        for res, partition in coarse_partitions.items():
            qual = partition.quality()
            if qual > coarse_best_qual:
                coarse_best_qual, coarse_best_res = qual, res
                if return_modules:
                    coarse_best_partition = partition

        # Free coarse partitions before fine search
        del coarse_partitions

        logger.info(
            "Coarse phase complete: best resolution=%.3f (modularity=%.4f)",
            coarse_best_res,
            coarse_best_qual,
        )

        # Phase 2: Fine search around coarse best
        fine_half_range = (total_range * fine_range_fraction) / 2.0
        fine_min = max(resolution_min, coarse_best_res - fine_half_range)
        fine_max = min(resolution_max, coarse_best_res + fine_half_range)

        fine_resolutions = np.linspace(fine_min, fine_max, fine_steps)
        logger.info(
            "Two-phase search: fine phase in [%.2f, %.2f] with %d steps",
            fine_min,
            fine_max,
            fine_steps,
        )

        best_res, best_qual = coarse_best_res, coarse_best_qual
        best_partition = coarse_best_partition if return_modules else None
        fine_partitions = _leiden_partitions_for_resolutions(
            graph,
            fine_resolutions,
            progress=progress,
            progress_desc="Fine resolution search",
        )
        for res, partition in fine_partitions.items():
            qual = partition.quality()
            if qual > best_qual:
                best_qual, best_res = qual, res
                if return_modules:
                    best_partition = partition

        logger.info(
            "Two-phase search complete: optimal resolution=%.3f (modularity=%.4f)",
            best_res,
            best_qual,
        )
    else:
        # Single-phase search (original behavior)
        resolutions = np.linspace(resolution_min, resolution_max, n_steps)

        best_res, best_qual = 1.0, -np.inf
        best_partition = None
        logger.info(
            "Searching for optimal resolution in [%.2f, %.2f] with %d steps",
            resolution_min,
            resolution_max,
            n_steps,
        )

        partitions = _leiden_partitions_for_resolutions(
            graph,
            resolutions,
            progress=progress,
            progress_desc="Optimizing Leiden resolution",
        )
        for res, partition in partitions.items():
            qual = partition.quality()
            if qual > best_qual:
                best_qual, best_res = qual, res
                if return_modules:
                    best_partition = partition

        logger.info("Optimal resolution: %.3f (modularity=%.4f)", best_res, best_qual)

    if return_modules:
        if best_partition is None:
            # Re-run for the best resolution if partition wasn't stored
            best_partition = _leiden_partitions_for_resolutions(graph, [best_res])[best_res]
        modules = pd.Series(best_partition.membership, index=list(genes), name="module")
        logger.info(format_module_feedback("Leiden", modules, resolution=best_res))
        return best_res, best_qual, modules

    return best_res, best_qual


def leiden_modules(
    A: np.ndarray | pd.DataFrame | None = None,
    resolution: float = 1.0,
    *,
    adjacency_mode: str = "wgcna-signed",
    graph: Optional["ig.Graph"] = None,
    edges: Optional[np.ndarray] = None,
    weights: Optional[Sequence[float]] = None,
    genes: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Cluster genes into modules using Leiden community detection.

    Parameters
    ----------
    A:
        Adjacency matrix (numpy array or pandas DataFrame). If a DataFrame, the
        index is used as gene names. Optional when providing ``graph`` or
        ``edges``.
    graph:
        Pre-built igraph Graph (optional).
    edges:
        Edge list of shape ``(n_edges, 2)`` or ``(n_edges, 3)`` (optional).
    weights:
        Edge weights aligned to ``edges`` (optional).
    genes:
        Gene names aligned to vertex indices (optional for ``edges``).
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

    if adjacency_mode == "signed":
        raise NotImplementedError(
            "Signed community detection is not yet supported. "
            "Leiden requires non-negative edge weights. "
            "Use adjacency_mode='wgcna-signed'."
        )

    if graph is None:
        if edges is not None:
            edges_arr, weight_arr = _normalize_edge_list(edges, weights)
            if weight_arr is not None:
                weight_arr = np.maximum(weight_arr, 0.0)
                if np.any(weight_arr < 0):
                    raise ValueError(
                        "Negative weights detected after adjacency transform. "
                        "This should not occur for wgcna-signed mode."
                    )
            graph = build_graph_from_edge_list(edges_arr, weight_arr, genes)
            genes = list(graph.vs["name"])
        else:
            if A is None:
                raise ValueError("Provide adjacency data, edges, or a graph.")
            graph, genes = prepare_leiden_graph(A, genes, adjacency_mode=adjacency_mode)
    else:
        if genes is None:
            genes = list(graph.vs["name"]) if "name" in graph.vs.attributes() else [str(i) for i in range(graph.vcount())]
        if "weight" in graph.es.attributes():
            weights_arr = np.array(graph.es["weight"], dtype=float)
            if np.any(weights_arr < 0):
                raise ValueError(
                    "Negative weights detected after adjacency transform. "
                    "This should not occur for wgcna-signed mode."
                )

    logger.info(
        "Running Leiden clustering on %d genes (resolution=%.3f, adjacency_mode=%s)",
        len(genes),
        resolution,
        adjacency_mode,
    )
    return leiden_modules_from_graph(graph, genes, resolution)


def spectral_modules(
    A: np.ndarray | pd.DataFrame,
    n_clusters: Optional[int] = None,
    n_components: Optional[int] = None,
    *,
    adjacency_mode: str = "wgcna-signed",
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
    adjacency_mode:
        How to handle negative edge weights before clustering.
        - "wgcna-signed" (default): clip negative values to zero.
        - "signed": preserve negative edges (may cause numerical issues with
          spectral clustering since it expects a similarity matrix).

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
    logger.info(
        "Running spectral clustering with %d clusters (adjacency_mode=%s)",
        n_clusters,
        adjacency_mode,
    )

    # Transform adjacency to handle negative weights
    arr = transform_adjacency_for_clustering(arr, mode=adjacency_mode)

    if adjacency_mode == "signed" and np.any(arr < 0):
        logger.warning(
            "Spectral clustering with negative edge weights may produce "
            "unreliable results. Consider using adjacency_mode='wgcna-signed'."
        )

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
    "build_graph_from_edge_list",
    "prepare_leiden_graph",
    "format_module_feedback",
    "optimize_resolution_modularity",
    "leiden_modules",
    "leiden_modules_from_graph",
    "leiden_modules_for_resolutions",
    "spectral_modules",
    "compute_module_eigengenes",
    "save_modules",
    "save_eigengenes",
]
