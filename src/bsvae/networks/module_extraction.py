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
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _infer_separator(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return "\t" if suffix == ".tsv" else ","


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


def load_adjacency(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix with gene labels.

    Parameters
    ----------
    path:
        CSV/TSV file with genes as the index and columns.

    Returns
    -------
    adjacency : np.ndarray
        Square matrix of edge weights.
    genes : list[str]
        Gene identifiers derived from the file index.
    """

    sep = _infer_separator(path)
    df = pd.read_csv(path, index_col=0, sep=sep)
    if df.shape[0] != df.shape[1]:
        raise ValueError("Adjacency file must be square with genes as both index and columns")
    logger.info("Loaded adjacency from %s with %d genes", path, df.shape[0])
    return df.values, list(df.index)


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


def leiden_modules(A: np.ndarray | pd.DataFrame, resolution: float = 1.0) -> pd.Series:
    """Cluster genes into modules using Leiden community detection.

    Parameters
    ----------
    A:
        Adjacency matrix (numpy array or pandas DataFrame). If a DataFrame, the
        index is used as gene names.
    resolution:
        Resolution parameter for Leiden (higher values produce more clusters).

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
    logger.info("Running Leiden clustering on %d genes (resolution=%.3f)", len(genes), resolution)
    graph = build_graph_from_adjacency(arr, genes)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        weights=graph.es["weight"],
    )
    modules = pd.Series(partition.membership, index=genes, name="module")
    logger.info("Identified %d modules via Leiden", modules.nunique())
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
    logger.info("Identified %d modules via spectral clustering", modules.nunique())
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
    if shared_genes.empty:
        raise ValueError("No overlapping genes between expression matrix and modules")

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
    "leiden_modules",
    "spectral_modules",
    "compute_module_eigengenes",
    "save_modules",
    "save_eigengenes",
]
