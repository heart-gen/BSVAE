"""Network extraction utilities for BSVAE.

This module provides reusable helpers to derive gene–gene networks from
trained :class:`~bsvae.models.StructuredFactorVAE` instances using several
complimentary strategies:

1. **Decoder-loading similarity (Method A)** — cosine similarity between rows
   of the decoder weight matrix ``W``.
2. **Latent-space covariance propagation (Method B)** — propagate posterior
   uncertainty ``diag(exp(logvar_mean))`` through the decoder.
3. **Conditional independence graph (Method C)** — fit a Graphical Lasso on
   reconstructed expression ``\\hat{X} = Z W^T``.
4. **Laplacian-refined network (Method D)** — constrain decoder similarity to a
   supplied Laplacian prior.

The functions here are device-agnostic and written for integration in both CLI
workflows and unit tests.

GPU Acceleration
----------------
The following operations use GPU when available:

- **Model inference** (encoder forward pass): GPU
- **W similarity** (``compute_W_similarity``): GPU (PyTorch matmul)
- **Latent covariance** (``compute_latent_covariance``): GPU (PyTorch matmul)
- **Laplacian refinement** (``compute_laplacian_refined``): GPU (PyTorch matmul)
- **Graphical Lasso** (``compute_graphical_lasso``): GPU for reconstruction
  (Z @ W^T), CPU for sklearn GraphicalLasso fitting

See :mod:`bsvae.networks.module_extraction` for module extraction GPU notes.
"""
from __future__ import annotations

import gzip
import heapq
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.covariance import GraphicalLasso
from torch.utils.data import DataLoader, Dataset, TensorDataset

from bsvae.utils.modelIO import load_metadata
from bsvae.utils import modelIO as model_io
from bsvae.latent.latent_export import extract_latents

logger = logging.getLogger(__name__)


@dataclass
class NetworkResults:
    """Container for multiple adjacency matrices.

    Attributes
    ----------
    method : str
        Name of the method that produced the adjacency.
    adjacency : np.ndarray
        Symmetric matrix (G, G) encoding gene–gene connectivity.
    aux : dict
        Optional auxiliary outputs such as covariance or precision matrices.
    """

    method: str
    adjacency: np.ndarray
    aux: Optional[dict] = None


def load_model(model_path: str, device: Optional[torch.device] = None) -> torch.nn.Module:
    """Load a trained StructuredFactorVAE from a directory or checkpoint path.

    Parameters
    ----------
    model_path
        Path to the directory containing ``specs.json`` and ``model.pt`` or a
        direct path to the checkpoint file.
    device
        Torch device to place the model on. Defaults to CUDA when available.

    Returns
    -------
    torch.nn.Module
        Loaded model in evaluation mode.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isdir(model_path):
        directory, filename = model_path, model_io.MODEL_FILENAME
    else:
        directory, filename = os.path.dirname(model_path), os.path.basename(model_path)
    metadata = load_metadata(directory)
    model = model_io._get_model(metadata, device, os.path.join(directory, filename))
    model.eval()
    return model


def load_weights(model: torch.nn.Module, masked: bool = True) -> torch.Tensor:
    """Return the decoder weights ``W`` with optional masking applied.

    Parameters
    ----------
    model
        StructuredFactorVAE instance.
    masked
        When ``True``, apply the decoder mask if present.

    Returns
    -------
    torch.Tensor
        Decoder weights of shape ``(G, K)``.
    """

    W = model.decoder.W
    if masked and getattr(model.decoder, "mask", None) is not None:
        W = W * model.decoder.mask
    return W.detach()


def compute_W_similarity(W: torch.Tensor, eps: float = 1e-8) -> np.ndarray:
    """Compute cosine similarity between gene loading vectors (Method A).

    **GPU**: Uses GPU if input tensor ``W`` is on a CUDA device.

    Parameters
    ----------
    W
        Decoder weight matrix ``(G, K)``.
    eps
        Numerical stability constant added to norms.

    Returns
    -------
    np.ndarray
        Symmetric adjacency matrix ``(G, G)`` with cosine similarities.
    """

    W = W.float()
    W_norm = F.normalize(W, dim=1, eps=eps)
    adjacency = torch.matmul(W_norm, W_norm.T)
    return adjacency.cpu().numpy()


def compute_latent_covariance(W: torch.Tensor, logvar_mean: torch.Tensor, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate latent posterior variance through the decoder (Method B).

    Covariance is approximated as ``W diag(exp(logvar_mean)) W^T`` where
    ``logvar_mean`` is the dataset-average latent log-variance.

    Parameters
    ----------
    W
        Decoder weight matrix ``(G, K)``.
    logvar_mean
        Mean log-variance across samples ``(K,)``.
    eps
        Numerical jitter applied to the diagonal when computing correlation.

    Returns
    -------
    cov : np.ndarray
        Gene–gene covariance matrix ``(G, G)``.
    corr : np.ndarray
        Gene–gene Pearson correlation matrix ``(G, G)``.
    """

    if logvar_mean.dim() != 1:
        raise ValueError("logvar_mean must be a 1D tensor of length K")

    latent_var = torch.exp(logvar_mean)
    cov = torch.matmul(W, torch.diag(latent_var))
    cov = torch.matmul(cov, W.T)

    diag = torch.diag(cov).clamp(min=eps)
    std = torch.sqrt(diag)
    corr = cov / torch.outer(std, std)
    return cov.cpu().numpy(), corr.cpu().numpy()


def compute_graphical_lasso(
    latent_samples: np.ndarray,
    W: torch.Tensor,
    alpha: float = 0.01,
    max_iter: int = 100,
    precision_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit Graphical Lasso on reconstructed expression (Method C).

    Parameters
    ----------
    latent_samples
        Array of latent representations ``(n_samples, K)``; typically the
        posterior means ``mu``.
    W
        Decoder weights ``(G, K)``.
    alpha
        Regularization strength for :class:`sklearn.covariance.GraphicalLasso`.
    max_iter
        Maximum number of iterations for the solver.
    precision_tol
        Tolerance for considering precision entries as non-zero. Values with
        absolute magnitude below this threshold are treated as zero to avoid
        spurious edges from floating-point artifacts. Default is 1e-10.

    Returns
    -------
    precision : np.ndarray
        Estimated precision matrix ``(G, G)``.
    covariance : np.ndarray
        Model-implied covariance matrix from the Graphical Lasso.
    adjacency : np.ndarray
        Binary adjacency where non-zero precision entries indicate edges.
    """

    # Compute Xhat = Z @ W^T on GPU if W is on GPU, then transfer to CPU for sklearn
    latent_tensor = torch.from_numpy(latent_samples).to(W.device)
    Xhat = torch.matmul(latent_tensor, W.detach().T).cpu().numpy()
    del latent_tensor  # Free GPU memory before sklearn fitting

    gl = GraphicalLasso(alpha=alpha, max_iter=max_iter)
    gl.fit(Xhat)
    precision = gl.precision_
    covariance = gl.covariance_
    adjacency = (np.abs(precision) > precision_tol).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return precision, covariance, adjacency


def compute_laplacian_refined(W: torch.Tensor, laplacian: torch.Tensor) -> np.ndarray:
    """Mask decoder similarity by a Laplacian prior (Method D).

    Parameters
    ----------
    W
        Decoder weights ``(G, K)``.
    laplacian
        Laplacian matrix or sparse Laplacian compatible with the decoder.

    Returns
    -------
    np.ndarray
        Adjacency matrix refined by the Laplacian structure.
    """

    similarity = torch.matmul(W, W.T)
    if laplacian.is_sparse:
        mask = laplacian.coalesce().to_dense() != 0
    else:
        mask = laplacian != 0
    refined = similarity * mask
    return refined.cpu().numpy()


def save_adjacency_matrix(adjacency: np.ndarray, output_path: str, genes: Optional[Sequence[str]] = None) -> None:
    """Persist an adjacency matrix to disk.

    Parameters
    ----------
    adjacency
        Square matrix to save.
    output_path
        Destination path. ``.csv``/``.tsv`` are written via pandas, otherwise
        ``.npy`` is used.
    genes
        Optional gene identifiers to use as row/column labels.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".csv", ".tsv"}:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        n_genes = adjacency.shape[0]
        if genes is None:
            genes = [str(i) for i in range(n_genes)]
        with open(path, "w") as f:
            # Write header: gene column + all gene names
            f.write(f"gene{sep}{sep.join(str(g) for g in genes)}\n")
            # Write each row: row gene name + values
            for i in range(n_genes):
                row_values = sep.join(str(v) for v in adjacency[i])
                f.write(f"{genes[i]}{sep}{row_values}\n")
    else:
        np.save(path, adjacency)


def save_edge_list(adjacency: np.ndarray, output_path: str, genes: Optional[Sequence[str]] = None, threshold: float = 0.0, include_self: bool = False) -> None:
    """Save an adjacency matrix as an edge list.

    Parameters
    ----------
    adjacency
        Square matrix encoding weights.
    output_path
        CSV/TSV path for the edge list.
    genes
        Optional list of gene names; defaults to integer indices.
    threshold
        Minimum absolute weight to keep an edge.
    include_self
        Whether to keep self-loops.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if genes is None:
        genes = list(range(adjacency.shape[0]))
    genes = list(genes)

    sep = "," if path.suffix.lower() == ".csv" else "\t"
    with open(path, "w") as f:
        f.write(f"source{sep}target{sep}weight\n")
        for i in range(adjacency.shape[0]):
            for j in range(adjacency.shape[1]):
                if not include_self and i == j:
                    continue
                weight = adjacency[i, j]
                if abs(weight) >= threshold:
                    f.write(f"{genes[i]}{sep}{genes[j]}{sep}{weight}\n")


def _infer_separator(path: str) -> str:
    suffixes = [s.lower() for s in Path(path).suffixes]
    return "\t" if ".tsv" in suffixes else ","


def _iter_upper_triangle(
    adjacency: np.ndarray,
    threshold: float = 0.0,
    include_diagonal: bool = False,
) -> Iterator[Tuple[int, int, float]]:
    """Stream upper-triangle edges without materializing the full triangle.

    Yields (i, j, weight) tuples for edges with |weight| >= threshold.
    Iterates row-by-row to avoid O(n^2) memory allocation.

    Parameters
    ----------
    adjacency
        Square adjacency matrix.
    threshold
        Minimum absolute weight to yield an edge (default 0 yields all).
    include_diagonal
        Whether to include diagonal entries (i == j).

    Yields
    ------
    Tuple[int, int, float]
        (row_index, col_index, weight) for each edge above threshold.
    """
    n = adjacency.shape[0]
    for i in range(n):
        start_j = i if include_diagonal else i + 1
        row = adjacency[i, start_j:]
        if threshold > 0:
            mask = np.abs(row) >= threshold
            cols = np.where(mask)[0] + start_j
            for j in cols:
                yield i, j, float(adjacency[i, j])
        else:
            for offset, w in enumerate(row):
                yield i, start_j + offset, float(w)


def compute_adaptive_threshold(adjacency: np.ndarray, target_sparsity: float = 0.01) -> float:
    """Compute threshold to achieve target sparsity in the adjacency matrix.

    Uses a streaming heap-based algorithm to avoid materializing the full
    upper triangle, which is critical for large networks.

    Parameters
    ----------
    adjacency
        Square adjacency matrix.
    target_sparsity
        Target fraction of edges to keep (e.g., 0.01 = top 1%).

    Returns
    -------
    float
        Threshold value such that keeping edges >= threshold yields target sparsity.
    """
    n = adjacency.shape[0]
    n_total_edges = n * (n - 1) // 2  # Upper triangle without diagonal
    n_edges_target = max(1, int(n_total_edges * target_sparsity))

    if n_edges_target >= n_total_edges:
        return 0.0

    # Use a min-heap to track top k largest absolute weights.
    # The heap stores absolute values; at the end the minimum is our threshold.
    top_k: List[float] = []

    for i in range(n):
        row = np.abs(adjacency[i, i + 1:])
        for w in row:
            if len(top_k) < n_edges_target:
                heapq.heappush(top_k, w)
            elif w > top_k[0]:
                heapq.heapreplace(top_k, w)

    return float(top_k[0]) if top_k else 0.0


def save_adjacency_sparse(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    compress: bool = True,
    quantize: Union[bool, str] = True,
) -> None:
    """Save an adjacency matrix in compressed format.

    For symmetric matrices, only the upper triangle values are stored (no indices
    needed for dense matrices), reducing size by ~50%. Combined with quantization
    and gzip compression, a 20k gene network can be stored in ~50-80 MB.

    Storage format:
    - Dense networks (>50% non-zero): Store upper triangle values as flat array
    - Sparse networks: Store COO format (row, col, data)

    When threshold > 0, streaming is used to avoid materializing the full upper
    triangle, which is critical for large networks.

    Parameters
    ----------
    adjacency
        Square matrix to save.
    output_path
        Destination path. Should end with ``.npz``.
    genes
        Optional gene identifiers to save alongside the matrix.
    threshold
        Minimum absolute weight to keep an edge. Edges below threshold are zeroed.
        Use threshold=0 to keep all edges (recommended for clustering).
    compress
        Whether to use compression (default True).
    quantize
        Quantization level for values:
        - True or "float16": Use float16 (default, good balance)
        - "int8": Use int8 for values in [-1, 1] range (smallest, ~50 MB for 20k genes)
        - False or "float32": No quantization (largest, most precise)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = adjacency.shape[0]

    # Prepare gene list
    if genes is None:
        genes = [str(i) for i in range(n)]
    genes_arr = np.array(genes, dtype=object)

    save_func = np.savez_compressed if compress else np.savez

    # Determine dtype name for metadata
    if quantize == "int8":
        dtype_name = "int8"
    elif quantize is False or quantize == "float32":
        dtype_name = "float32"
    else:
        dtype_name = "float16"

    # When threshold > 0, use streaming to avoid materializing full upper triangle.
    # This is critical for large networks where upper triangle has ~n^2/2 entries.
    if threshold > 0:
        # Stream edges row-by-row, collecting only those above threshold
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        for i, j, w in _iter_upper_triangle(adjacency, threshold=threshold, include_diagonal=True):
            rows.append(i)
            cols.append(j)
            data.append(w)

        n_nonzero = len(data)
        n_total = n * (n + 1) // 2  # Upper triangle including diagonal
        density = n_nonzero / n_total

        # Quantize sparse data
        data_arr = np.array(data, dtype=np.float32)
        scale_factor = None
        if quantize == "int8":
            scale_factor = np.abs(data_arr).max() if len(data_arr) > 0 else 0.0
            if scale_factor > 0:
                data_q = (data_arr / scale_factor * 127).astype(np.int8)
            else:
                data_q = np.zeros(len(data_arr), dtype=np.int8)
        elif quantize is False or quantize == "float32":
            data_q = data_arr
        else:
            data_q = data_arr.astype(np.float16)

        # Use int16 for indices if n < 32768, otherwise int32
        idx_dtype = np.int16 if n < 32768 else np.int32

        save_dict = {
            "data": data_q,
            "row": np.array(rows, dtype=idx_dtype),
            "col": np.array(cols, dtype=idx_dtype),
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["sparse_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved sparse adjacency to %s: %d genes, %d edges (%.1f%% density), dtype=%s",
            path,
            n,
            n_nonzero,
            density * 100,
            dtype_name,
        )
        return

    # threshold = 0: need full upper triangle for dense storage or density check.
    # Extract upper triangle values (including diagonal)
    triu_idx = np.triu_indices(n)
    triu_data = adjacency[triu_idx]

    # Check sparsity to decide storage format
    n_nonzero = np.sum(triu_data != 0)
    n_total = len(triu_data)
    density = n_nonzero / n_total

    # Determine quantization
    scale_factor = None
    if quantize == "int8":
        # Quantize to int8 - good for similarity values in [-1, 1]
        # Scale to use full int8 range
        vmin, vmax = triu_data.min(), triu_data.max()
        scale_factor = max(abs(vmin), abs(vmax))
        if scale_factor > 0:
            triu_data_q = (triu_data / scale_factor * 127).astype(np.int8)
        else:
            triu_data_q = np.zeros_like(triu_data, dtype=np.int8)
    elif quantize is False or quantize == "float32":
        triu_data_q = triu_data.astype(np.float32)
    else:  # True or "float16"
        triu_data_q = triu_data.astype(np.float16)

    if density > 0.5:
        # Dense storage: just store upper triangle values as flat array
        save_dict = {
            "triu_values": triu_data_q,
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["dense_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved dense adjacency to %s: %d genes, %.1f%% density, dtype=%s",
            path,
            n,
            density * 100,
            dtype_name,
        )
    else:
        # Sparse storage: store only non-zero entries with indices
        nonzero_mask = triu_data != 0
        row = triu_idx[0][nonzero_mask]
        col = triu_idx[1][nonzero_mask]

        if quantize == "int8" and scale_factor is not None and scale_factor > 0:
            data_q = (triu_data[nonzero_mask] / scale_factor * 127).astype(np.int8)
        elif quantize is False or quantize == "float32":
            data_q = triu_data[nonzero_mask].astype(np.float32)
        else:
            data_q = triu_data[nonzero_mask].astype(np.float16)

        # Use int16 for indices if n < 32768, otherwise int32
        idx_dtype = np.int16 if n < 32768 else np.int32

        save_dict = {
            "data": data_q,
            "row": row.astype(idx_dtype),
            "col": col.astype(idx_dtype),
            "n": np.array([n], dtype=np.int32),
            "genes": genes_arr,
            "storage_format": np.array(["sparse_triu"], dtype=object),
            "dtype": np.array([dtype_name], dtype=object),
        }
        if scale_factor is not None:
            save_dict["scale_factor"] = np.array([scale_factor], dtype=np.float32)

        save_func(path, **save_dict)
        logger.info(
            "Saved sparse adjacency to %s: %d genes, %d edges (%.1f%% density), dtype=%s",
            path,
            n,
            n_nonzero,
            density * 100,
            dtype_name,
        )


def load_adjacency_sparse(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix from NPZ format.

    Handles multiple storage formats:
    - dense_triu: Upper triangle values stored as flat array (for dense matrices)
    - sparse_triu: Upper triangle with indices (for sparse matrices)
    - Legacy formats with 'shape' key

    Automatically handles int8/float16/float32 quantization and rescaling.

    Parameters
    ----------
    path
        Path to the ``.npz`` file.

    Returns
    -------
    adjacency : np.ndarray
        Dense adjacency matrix reconstructed from storage.
    genes : list[str]
        Gene identifiers.
    """
    data = np.load(path, allow_pickle=True)
    genes = list(data["genes"])

    # Detect storage format and dtype
    storage_format = None
    if "storage_format" in data:
        storage_format = str(data["storage_format"][0])

    # Get scale factor for int8 quantization
    scale_factor = None
    if "scale_factor" in data:
        scale_factor = float(data["scale_factor"][0])

    def dequantize(values):
        """Convert quantized values back to float32."""
        if scale_factor is not None and values.dtype == np.int8:
            return (values.astype(np.float32) / 127.0) * scale_factor
        return values.astype(np.float32)

    if storage_format == "dense_triu":
        # Dense upper triangle storage
        n = int(data["n"][0])
        triu_values = dequantize(data["triu_values"])

        # Reconstruct full symmetric matrix
        adjacency = np.zeros((n, n), dtype=np.float32)
        triu_idx = np.triu_indices(n)
        off_diag_mask = triu_idx[0] != triu_idx[1]
        adjacency[triu_idx[0][off_diag_mask], triu_idx[1][off_diag_mask]] = triu_values[off_diag_mask]
        # Mirror to lower triangle
        adjacency = adjacency + adjacency.T - np.diag(np.diag(adjacency))

    elif storage_format == "sparse_triu":
        # Sparse upper triangle storage
        n = int(data["n"][0])
        row = data["row"]
        col = data["col"]
        values = dequantize(data["data"])

        # Reconstruct full symmetric matrix
        adjacency = np.zeros((n, n), dtype=np.float32)
        off_diag_mask = row != col
        adjacency[row[off_diag_mask], col[off_diag_mask]] = values[off_diag_mask]
        # Mirror to lower triangle (excluding diagonal)
        adjacency[col[off_diag_mask], row[off_diag_mask]] = values[off_diag_mask]

    elif "shape" in data:
        # Legacy full sparse format
        shape = tuple(data["shape"])
        row = data["row"]
        col = data["col"]
        values = dequantize(data["data"])

        if "upper_triangle" in data and data["upper_triangle"][0]:
            # Legacy upper triangle format
            n = shape[0]
            adjacency = np.zeros(shape, dtype=np.float32)
            off_diag_mask = row != col
            adjacency[row[off_diag_mask], col[off_diag_mask]] = values[off_diag_mask]
            adjacency[col[off_diag_mask], row[off_diag_mask]] = values[off_diag_mask]
        else:
            # Full COO sparse format
            sparse_adj = sp.coo_matrix((values, (row, col)), shape=shape)
            adjacency = sparse_adj.toarray().astype(np.float32)
    else:
        raise ValueError(f"Unknown adjacency storage format in {path}")

    logger.info("Loaded adjacency from %s: %d genes", path, len(genes))
    return adjacency, genes


def load_adjacency_sparse_edges(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load sparse upper-triangle adjacency data as an edge list.

    Parameters
    ----------
    path
        Path to the ``.npz`` file.

    Returns
    -------
    edges : np.ndarray
        Array of shape ``(n_edges, 2)`` with integer indices.
    weights : np.ndarray
        Edge weights aligned to ``edges``.
    genes : list[str]
        Gene identifiers.
    """
    data = np.load(path, allow_pickle=True)
    genes = list(data["genes"])

    storage_format = None
    if "storage_format" in data:
        storage_format = str(data["storage_format"][0])

    scale_factor = None
    if "scale_factor" in data:
        scale_factor = float(data["scale_factor"][0])

    def dequantize(values: np.ndarray) -> np.ndarray:
        if scale_factor is not None and values.dtype == np.int8:
            return (values.astype(np.float32) / 127.0) * scale_factor
        return values.astype(np.float32)

    if storage_format == "dense_triu":
        n = int(data["n"][0])
        triu_values = dequantize(data["triu_values"])
        triu_idx = np.triu_indices(n)
        nonzero_mask = triu_values != 0
        row = triu_idx[0][nonzero_mask]
        col = triu_idx[1][nonzero_mask]
        values = triu_values[nonzero_mask]
    elif storage_format == "sparse_triu":
        row = data["row"].astype(np.int64)
        col = data["col"].astype(np.int64)
        values = dequantize(data["data"])
    elif "shape" in data:
        row = data["row"].astype(np.int64)
        col = data["col"].astype(np.int64)
        values = dequantize(data["data"])
        if not ("upper_triangle" in data and data["upper_triangle"][0]):
            upper_mask = row <= col
            row = row[upper_mask]
            col = col[upper_mask]
            values = values[upper_mask]
    else:
        raise ValueError(f"Unknown adjacency storage format in {path}")

    edges = np.column_stack((row, col)).astype(np.int64)
    logger.info("Loaded sparse edges from %s: %d genes, %d edges", path, len(genes), len(edges))
    return edges, values, genes


def save_edge_list_compressed(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    include_self: bool = False,
    compress: bool = True,
    use_indices: bool = True,
) -> None:
    """Save an adjacency matrix as a compressed edge list.

    .. deprecated::
        Use :func:`save_edge_list_parquet` instead for better compression
        and performance.

    For large networks, this stores edges using integer indices with a
    separate gene lookup, reducing file size significantly.

    Parameters
    ----------
    adjacency
        Square matrix encoding weights.
    output_path
        Output path. If compress=True and path doesn't end with .gz, .gz is appended.
    genes
        Optional list of gene names.
    threshold
        Minimum absolute weight to keep an edge.
    include_self
        Whether to keep self-loops.
    compress
        Whether to gzip the output (default True).
    use_indices
        If True, store integer indices and save genes separately.
    """
    warnings.warn(
        "save_edge_list_compressed is deprecated. Use save_edge_list_parquet instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if genes is None:
        genes = list(range(adjacency.shape[0]))
    genes = list(genes)

    # Determine separator from original path (before adding .gz)
    base_path = str(path).rstrip(".gz")
    sep = "," if Path(base_path).suffix.lower() == ".csv" else "\t"

    # Collect edges (upper triangle only for symmetric matrix)
    edges = []
    for i in range(adjacency.shape[0]):
        # Include diagonal (self-loops) when include_self=True
        start_j = i if include_self else i + 1
        for j in range(start_j, adjacency.shape[1]):
            weight = adjacency[i, j]
            if abs(weight) >= threshold:
                edges.append((i, j, weight))

    # Prepare output path
    if compress and not str(path).endswith(".gz"):
        path = Path(str(path) + ".gz")

    # Write edges
    open_func = gzip.open if compress else open
    mode = "wt" if compress else "w"

    with open_func(path, mode) as f:
        if use_indices:
            f.write(f"source_idx{sep}target_idx{sep}weight\n")
            for src, tgt, weight in edges:
                f.write(f"{src}{sep}{tgt}{sep}{weight:.6g}\n")
        else:
            f.write(f"source{sep}target{sep}weight\n")
            for src, tgt, weight in edges:
                f.write(f"{genes[src]}{sep}{genes[tgt]}{sep}{weight:.6g}\n")

    # Save gene lookup if using indices
    if use_indices:
        gene_path = path.parent / (path.stem.replace(".csv", "").replace(".tsv", "") + "_genes.txt")
        if compress:
            gene_path = Path(str(gene_path).rstrip(".gz"))
        with open(gene_path, "w") as f:
            for gene in genes:
                f.write(f"{gene}\n")
        logger.info("Saved gene lookup to %s", gene_path)

    logger.info(
        "Saved compressed edge list to %s: %d edges (threshold=%.4f)",
        path,
        len(edges),
        threshold,
    )


def load_edge_list_compressed(
    path: str,
    genes_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix from a compressed edge list.

    .. deprecated::
        Use :func:`load_edge_list_parquet` instead for better performance.

    Parameters
    ----------
    path
        Path to the edge list file (can be gzipped).
    genes_path
        Optional path to gene lookup file. If not provided, attempts to
        find it automatically.

    Returns
    -------
    adjacency : np.ndarray
        Dense adjacency matrix reconstructed from edge list.
    genes : list[str]
        Gene identifiers.
    """
    warnings.warn(
        "load_edge_list_compressed is deprecated. Use load_edge_list_parquet instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(path)

    # Determine if file is compressed
    is_compressed = str(path).endswith(".gz")
    open_func = gzip.open if is_compressed else open
    mode = "rt" if is_compressed else "r"

    # Read edges
    edges = []
    with open_func(path, mode) as f:
        header = f.readline().strip()
        sep = "\t" if "\t" in header else ","
        cols = header.split(sep)
        use_indices = "source_idx" in cols

        for line in f:
            parts = line.strip().split(sep)
            src, tgt, weight = parts[0], parts[1], float(parts[2])
            if use_indices:
                edges.append((int(src), int(tgt), weight))
            else:
                edges.append((src, tgt, weight))

    # Load or infer genes
    if use_indices:
        if genes_path is None:
            # Try to find gene file automatically
            stem = path.stem.replace(".csv", "").replace(".tsv", "").replace(".gz", "")
            genes_path = path.parent / f"{stem}_genes.txt"
            if not genes_path.exists():
                # Try without .gz suffix removal
                stem = path.name.replace(".csv.gz", "").replace(".tsv.gz", "")
                genes_path = path.parent / f"{stem}_genes.txt"

        if genes_path and Path(genes_path).exists():
            with open(genes_path) as f:
                genes = [line.strip() for line in f]
        else:
            # Infer size from max index
            max_idx = max(max(e[0], e[1]) for e in edges)
            genes = [str(i) for i in range(max_idx + 1)]
    else:
        # Extract unique genes from edges
        all_genes = set()
        for src, tgt, _ in edges:
            all_genes.add(src)
            all_genes.add(tgt)
        genes = sorted(all_genes)

    # Build adjacency matrix
    n = len(genes)
    adjacency = np.zeros((n, n), dtype=np.float32)

    if use_indices:
        for src, tgt, weight in edges:
            adjacency[src, tgt] = weight
            adjacency[tgt, src] = weight  # Symmetric
    else:
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        for src, tgt, weight in edges:
            i, j = gene_to_idx[src], gene_to_idx[tgt]
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    logger.info("Loaded edge list from %s: %d genes, %d edges", path, n, len(edges))
    return adjacency, genes


def load_edge_list_compressed_edges(
    path: str,
    genes_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a compressed edge list as indexed edges and weights.

    Parameters
    ----------
    path
        Path to the edge list file (can be gzipped).
    genes_path
        Optional path to gene lookup file.

    Returns
    -------
    edges : np.ndarray
        Array of shape ``(n_edges, 2)`` with integer indices.
    weights : np.ndarray
        Edge weights aligned to ``edges``.
    genes : list[str]
        Gene identifiers.
    """
    warnings.warn(
        "load_edge_list_compressed_edges is deprecated. Use load_edge_list_parquet_edges instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    path = Path(path)
    is_compressed = str(path).endswith(".gz")
    open_func = gzip.open if is_compressed else open
    mode = "rt" if is_compressed else "r"

    sources: List[int] = []
    targets: List[int] = []
    weights: List[float] = []
    use_indices = False
    raw_edges: List[Tuple[str, str, float]] = []

    with open_func(path, mode) as f:
        header = f.readline().strip()
        sep = "\t" if "\t" in header else ","
        cols = header.split(sep)
        use_indices = "source_idx" in cols

        for line in f:
            parts = line.strip().split(sep)
            src, tgt, weight = parts[0], parts[1], float(parts[2])
            if use_indices:
                sources.append(int(src))
                targets.append(int(tgt))
                weights.append(weight)
            else:
                raw_edges.append((src, tgt, weight))

    if not use_indices:
        if genes_path is None:
            all_genes = {g for edge in raw_edges for g in edge[:2]}
            genes = sorted(all_genes)
        else:
            with open(genes_path) as f:
                genes = [line.strip() for line in f]
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        for src, tgt, weight in raw_edges:
            sources.append(gene_to_idx[src])
            targets.append(gene_to_idx[tgt])
            weights.append(weight)
    else:
        if genes_path is None:
            stem = path.stem.replace(".csv", "").replace(".tsv", "").replace(".gz", "")
            genes_path = path.parent / f"{stem}_genes.txt"
            if not genes_path.exists():
                stem = path.name.replace(".csv.gz", "").replace(".tsv.gz", "")
                genes_path = path.parent / f"{stem}_genes.txt"
        if genes_path and Path(genes_path).exists():
            with open(genes_path) as f:
                genes = [line.strip() for line in f]
        else:
            max_idx = max(max(sources, default=0), max(targets, default=0))
            genes = [str(i) for i in range(max_idx + 1)]

    edges = np.column_stack((np.array(sources, dtype=np.int64), np.array(targets, dtype=np.int64)))
    weight_arr = np.array(weights, dtype=np.float32)
    logger.info("Loaded edge list from %s: %d genes, %d edges", path, len(genes), len(edges))
    return edges, weight_arr, genes


def save_edge_list_parquet(
    adjacency: np.ndarray,
    output_path: str,
    genes: Optional[Sequence[str]] = None,
    threshold: float = 0.0,
    include_self: bool = False,
    compression: str = "zstd",
) -> None:
    """Save an adjacency matrix as a Parquet edge list.

    Parquet format provides better compression and faster read/write performance
    compared to gzipped CSV. Gene names are stored as metadata within the file,
    eliminating the need for a separate lookup file.

    Parameters
    ----------
    adjacency
        Square matrix encoding weights.
    output_path
        Output path. Should end with ``.parquet``. If not, ``.parquet`` is appended.
    genes
        Optional list of gene names.
    threshold
        Minimum absolute weight to keep an edge.
    include_self
        Whether to keep self-loops.
    compression
        Parquet compression codec: "zstd" (default, best balance),
        "snappy" (fastest), "gzip", or None for no compression.
    """
    path = Path(output_path)
    if not str(path).endswith(".parquet"):
        path = Path(str(path).replace(".csv", "").replace(".tsv", "") + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)

    if genes is None:
        genes = [str(i) for i in range(adjacency.shape[0])]
    genes = list(genes)

    # Stream edges row-by-row to avoid materializing full upper triangle.
    # For large networks (e.g., 20k genes), the upper triangle has ~200M entries;
    # streaming keeps memory proportional to the number of edges above threshold.
    src_list: List[int] = []
    tgt_list: List[int] = []
    weight_list: List[float] = []

    for i, j, w in _iter_upper_triangle(adjacency, threshold=threshold, include_diagonal=include_self):
        src_list.append(i)
        tgt_list.append(j)
        weight_list.append(w)

    # Create DataFrame with integer indices for compact storage
    df = pd.DataFrame({
        "source_idx": np.array(src_list, dtype=np.int32),
        "target_idx": np.array(tgt_list, dtype=np.int32),
        "weight": np.array(weight_list, dtype=np.float32),
    })

    # Store genes as Parquet metadata
    table = pa.Table.from_pandas(df, preserve_index=False)
    genes_json = "\n".join(genes)  # Newline-separated for compatibility
    metadata = {
        b"genes": genes_json.encode("utf-8"),
        b"n_genes": str(len(genes)).encode("utf-8"),
    }
    # Merge with existing schema metadata
    existing_metadata = table.schema.metadata or {}
    merged_metadata = {**existing_metadata, **metadata}
    table = table.replace_schema_metadata(merged_metadata)

    pq.write_table(table, path, compression=compression)

    logger.info(
        "Saved edge list to %s: %d edges (threshold=%.4f, compression=%s)",
        path,
        len(df),
        threshold,
        compression,
    )


def load_edge_list_parquet(path: str) -> Tuple[np.ndarray, List[str]]:
    """Load an adjacency matrix from a Parquet edge list.

    Parameters
    ----------
    path
        Path to the Parquet edge list file.

    Returns
    -------
    adjacency : np.ndarray
        Dense adjacency matrix reconstructed from edge list.
    genes : list[str]
        Gene identifiers extracted from file metadata.
    """
    path = Path(path)

    # Read Parquet file
    table = pq.read_table(path)
    df = table.to_pandas()

    # Extract genes from metadata
    metadata = table.schema.metadata or {}
    if b"genes" in metadata:
        genes_str = metadata[b"genes"].decode("utf-8")
        genes = genes_str.split("\n")
    else:
        # Fallback: infer from max index
        max_idx = max(df["source_idx"].max(), df["target_idx"].max())
        genes = [str(i) for i in range(max_idx + 1)]
        logger.warning("No genes metadata found in %s, using numeric indices", path)

    # Build adjacency matrix
    n = len(genes)
    adjacency = np.zeros((n, n), dtype=np.float32)

    src_idx = df["source_idx"].values
    tgt_idx = df["target_idx"].values
    weights = df["weight"].values

    adjacency[src_idx, tgt_idx] = weights
    adjacency[tgt_idx, src_idx] = weights  # Symmetric

    logger.info("Loaded edge list from %s: %d genes, %d edges", path, n, len(df))
    return adjacency, genes


def load_edge_list_parquet_edges(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load a Parquet edge list as indexed edges and weights."""
    path = Path(path)
    table = pq.read_table(path)
    df = table.to_pandas()

    metadata = table.schema.metadata or {}
    if b"genes" in metadata:
        genes_str = metadata[b"genes"].decode("utf-8")
        genes = genes_str.split("\n")
    else:
        max_idx = max(df["source_idx"].max(), df["target_idx"].max())
        genes = [str(i) for i in range(max_idx + 1)]
        logger.warning("No genes metadata found in %s, using numeric indices", path)

    edges = np.column_stack((df["source_idx"].values, df["target_idx"].values)).astype(np.int64)
    weights = df["weight"].values.astype(np.float32)
    logger.info("Loaded edge list from %s: %d genes, %d edges", path, len(genes), len(edges))
    return edges, weights, genes


def load_expression(path: str) -> pd.DataFrame:
    """Load a gene expression matrix (genes × samples)."""

    sep = _infer_separator(path)
    return pd.read_csv(path, index_col=0, sep=sep)


def create_dataloader_from_expression(path: str, batch_size: int = 128) -> Tuple[DataLoader, List[str], List[str]]:
    """Create a DataLoader from a genes × samples matrix.

    Parameters
    ----------
    path
        CSV/TSV file containing the expression matrix.
    batch_size
        Batch size for the returned DataLoader.

    Returns
    -------
    dataloader : torch.utils.data.DataLoader
        Yields ``(expression, sample_id)`` pairs with shape ``(batch, G)``.
    genes : list of str
        Gene identifiers from the index.
    samples : list of str
        Sample identifiers from the columns.
    """

    df = load_expression(path)
    tensor = torch.from_numpy(df.T.values.astype(np.float32))
    dataset = TensorDataset(tensor, torch.arange(tensor.shape[0]))

    class SampleIdWrapper(Dataset):
        def __init__(self, base: Dataset, sample_ids: Sequence[str]):
            self.base = base
            self.sample_ids = list(sample_ids)

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            x, _ = self.base[idx]
            return x, self.sample_ids[idx]

    wrapped = SampleIdWrapper(dataset, df.columns)
    dataloader = DataLoader(wrapped, batch_size=batch_size, shuffle=False)
    return dataloader, list(df.index), list(df.columns)


def run_extraction(
    model: torch.nn.Module,
    dataloader: DataLoader,
    genes: Sequence[str],
    methods: Iterable[str],
    threshold: float = 0.0,
    alpha: float = 0.01,
    output_dir: Optional[str] = None,
    create_heatmaps: bool = False,
    sparse: bool = True,
    compress: bool = True,
    target_sparsity: float = 0.01,
    quantize: Union[bool, str] = "int8",
) -> List[NetworkResults]:
    """Run requested network extraction methods.

    Parameters
    ----------
    model
        Loaded StructuredFactorVAE.
    dataloader
        Iterator over expression data.
    genes
        Gene identifiers corresponding to decoder rows.
    methods
        Iterable of methods to compute (case-insensitive).
    threshold
        Threshold applied when writing edge lists. If 0 and sparse=True,
        an adaptive threshold is computed based on target_sparsity.
    alpha
        Graphical Lasso regularization strength.
    output_dir
        Optional directory to persist results.
    create_heatmaps
        When ``True`` generate matplotlib heatmaps for adjacencies.
    sparse
        When ``True`` (default), save adjacency in sparse NPZ format and
        apply thresholding to reduce file size.
    compress
        When ``True`` (default), use gzip compression for edge lists.
    target_sparsity
        Target fraction of edges to keep when using adaptive thresholding
        (default 0.01 = top 1% of edges).
    quantize
        Quantization level: "int8" (smallest), "float16", or "float32".

    Returns
    -------
    list of NetworkResults
        One entry per computed method.
    """

    device = next(model.parameters()).device
    W = load_weights(model).to(device)
    methods = [m.lower() for m in methods]

    if W.shape[0] != len(genes):
        raise ValueError(
            f"Gene dimension mismatch: decoder has {W.shape[0]} rows but {len(genes)} genes were provided."
        )

    mu, logvar, sample_ids = extract_latents(model, dataloader, device=device)
    results: List[NetworkResults] = []

    if "w_similarity" in methods:
        adjacency = compute_W_similarity(W)
        results.append(NetworkResults("w_similarity", adjacency))
        _persist(adjacency, genes, output_dir, "w_similarity", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)

    if "latent_cov" in methods:
        logvar_mean = torch.from_numpy(logvar).to(device).mean(dim=0)
        cov, corr = compute_latent_covariance(W, logvar_mean)
        results.append(NetworkResults("latent_cov", cov, {"correlation": corr}))
        _persist(cov, genes, output_dir, "latent_cov", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)
        if output_dir:
            # Also save correlation in sparse format
            _persist(corr, genes, output_dir, "latent_cov_correlation", threshold, False, sparse, compress, target_sparsity, quantize)

    if "graphical_lasso" in methods:
        precision, covariance, adjacency = compute_graphical_lasso(mu, W, alpha=alpha)
        results.append(NetworkResults("graphical_lasso", adjacency, {"precision": precision, "covariance": covariance}))
        _persist(adjacency, genes, output_dir, "graphical_lasso", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)
        if output_dir:
            _persist(precision, genes, output_dir, "graphical_lasso_precision", threshold, False, sparse, compress, target_sparsity, quantize)

    if "laplacian" in methods:
        if getattr(model, "laplacian_matrix", None) is not None:
            adjacency = compute_laplacian_refined(W, model.laplacian_matrix.to(device))
            results.append(NetworkResults("laplacian", adjacency))
            _persist(adjacency, genes, output_dir, "laplacian", threshold, create_heatmaps, sparse, compress, target_sparsity, quantize)
        else:
            logger.warning(
                "Laplacian method requested but model has no laplacian_matrix attribute; skipping."
            )

    return results


def _persist(
    adjacency: np.ndarray,
    genes: Sequence[str],
    output_dir: Optional[str],
    prefix: str,
    threshold: float,
    create_heatmaps: bool,
    sparse: bool = True,
    compress: bool = True,
    target_sparsity: float = 0.01,
    quantize: Union[bool, str] = "int8",
) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)

    # Compute adaptive threshold for edge list (thresholded for downstream use)
    edge_threshold = threshold
    if sparse and threshold == 0.0:
        edge_threshold = compute_adaptive_threshold(adjacency, target_sparsity)
        logger.info(
            "Using adaptive threshold %.4f for %s edge list (target sparsity=%.1f%%)",
            edge_threshold,
            prefix,
            target_sparsity * 100,
        )

    if sparse:
        # Save FULL adjacency matrix (threshold=0) for clustering accuracy
        # Compression + quantization provides significant size reduction
        save_adjacency_sparse(
            adjacency,
            os.path.join(output_dir, f"{prefix}_adjacency.npz"),
            genes,
            threshold=0.0,  # Keep all edges for clustering
            compress=compress,
            quantize=quantize,
        )
        # Save THRESHOLDED edge list for downstream analysis/visualization
        # Use Parquet format for better compression and performance
        compression = "zstd" if compress else None
        save_edge_list_parquet(
            adjacency,
            os.path.join(output_dir, f"{prefix}_edges.parquet"),
            genes,
            threshold=edge_threshold,
            compression=compression,
        )
    else:
        # Legacy dense format
        save_adjacency_matrix(adjacency, os.path.join(output_dir, f"{prefix}_adjacency.csv"), genes)
        save_edge_list(adjacency, os.path.join(output_dir, f"{prefix}_edges.csv"), genes, threshold=threshold)

    if create_heatmaps:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(adjacency, ax=ax, xticklabels=False, yticklabels=False, cmap="viridis")
            ax.set_title(prefix)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{prefix}_heatmap.png"), dpi=200)
            plt.close(fig)
        except Exception as exc:  # pragma: no cover - visualization optional
            logger.warning("Could not create heatmap for %s: %s", prefix, exc)


__all__ = [
    "NetworkResults",
    "load_model",
    "load_weights",
    "compute_W_similarity",
    "compute_latent_covariance",
    "compute_graphical_lasso",
    "compute_laplacian_refined",
    "save_adjacency_matrix",
    "save_edge_list",
    "compute_adaptive_threshold",
    "save_adjacency_sparse",
    "load_adjacency_sparse",
    "load_adjacency_sparse_edges",
    "save_edge_list_compressed",  # Deprecated
    "load_edge_list_compressed",  # Deprecated
    "load_edge_list_compressed_edges",  # Deprecated
    "save_edge_list_parquet",
    "load_edge_list_parquet",
    "load_edge_list_parquet_edges",
    "load_expression",
    "create_dataloader_from_expression",
    "run_extraction",
]
