"""Network extraction utilities for BSVAE.

This module provides reusable helpers to derive gene–gene networks from
trained :class:`~bsvae.models.StructuredFactorVAE` instances using several
complimentary strategies:

1. **Decoder-loading similarity (Method A)** — cosine similarity between rows
   of the decoder weight matrix ``W``.
2. **Latent-space covariance propagation (Method B)** — propagate posterior
   uncertainty ``diag(exp(logvar_mean))`` through the decoder.
3. **Conditional independence graph (Method C)** — fit a Graphical Lasso on
   reconstructed expression ``\hat{X} = Z W^T``.
4. **Laplacian-refined network (Method D)** — constrain decoder similarity to a
   supplied Laplacian prior.

The functions here are device-agnostic and written for integration in both CLI
workflows and unit tests.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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


def compute_W_similarity(W: torch.Tensor, eps: float = 1e-8, chunk_size: Optional[int] = None) -> np.ndarray:
    """Compute cosine similarity between gene loading vectors (Method A).

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

    if not chunk_size or chunk_size >= W_norm.shape[0]:
        adjacency = torch.matmul(W_norm, W_norm.T)
        return adjacency.cpu().numpy()

    adjacency = torch.empty((W_norm.shape[0], W_norm.shape[0]), device="cpu", dtype=W_norm.dtype)
    for start in range(0, W_norm.shape[0], chunk_size):
        end = min(start + chunk_size, W_norm.shape[0])
        block = torch.matmul(W_norm[start:end], W_norm.T)
        adjacency[start:end] = block.cpu()
    return adjacency.numpy()


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


def compute_graphical_lasso(latent_samples: np.ndarray, W: torch.Tensor, alpha: float = 0.01, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    precision : np.ndarray
        Estimated precision matrix ``(G, G)``.
    covariance : np.ndarray
        Model-implied covariance matrix from the Graphical Lasso.
    adjacency : np.ndarray
        Binary adjacency where non-zero precision entries indicate edges.
    """

    latent_samples = latent_samples.astype(np.float32, copy=False)
    W_np = W.detach().float().cpu().numpy()
    Xhat = np.matmul(latent_samples, W_np.T)
    gl = GraphicalLasso(alpha=alpha, max_iter=max_iter)
    gl.fit(Xhat)
    precision = gl.precision_
    covariance = gl.covariance_
    adjacency = (np.abs(precision) > 0).astype(float)
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

    if laplacian.is_sparse:
        laplacian = laplacian.coalesce()
        rows, cols = laplacian.indices()
        weights = torch.mul(W[rows], W[cols]).sum(dim=1)
        refined = torch.zeros((W.shape[0], W.shape[0]), device=W.device, dtype=W.dtype)
        refined[rows, cols] = weights
        refined = refined + refined.T - torch.diag(torch.diag(refined))
    else:
        similarity = torch.matmul(W, W.T)
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
        df = pd.DataFrame(adjacency, index=genes, columns=genes)
        df.to_csv(path, sep=sep)
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
    dialect = "excel" if sep == "," else "excel-tab"
    abs_weights = np.abs(adjacency)
    if not include_self:
        np.fill_diagonal(abs_weights, 0.0)

    mask = abs_weights >= threshold
    sources, targets = np.nonzero(mask)
    if not len(sources):
        rows_to_write: List[Sequence[object]] = []
    else:
        genes_arr = np.asarray(genes)
        rows_to_write = np.column_stack((genes_arr[sources], genes_arr[targets], adjacency[sources, targets]))

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, dialect=dialect)
        writer.writerow(["source", "target", "weight"])
        if len(rows_to_write):
            writer.writerows(rows_to_write.tolist())


def _infer_separator(path: str) -> str:
    suffixes = [s.lower() for s in Path(path).suffixes]
    return "\t" if ".tsv" in suffixes else ","


def load_expression(path: str) -> pd.DataFrame:
    """Load a gene expression matrix (genes × samples)."""

    sep = _infer_separator(path)
    return pd.read_csv(path, index_col=0, sep=sep, dtype=np.float32)


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
    values = df.to_numpy(dtype=np.float32, copy=False).T
    tensor = torch.as_tensor(values, dtype=torch.float32)
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
    similarity_chunk_size: Optional[int] = None,
    threshold: float = 0.0,
    alpha: float = 0.01,
    output_dir: Optional[str] = None,
    create_heatmaps: bool = False,
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
    similarity_chunk_size
        Optional block size to compute decoder similarities without materializing
        the full G × G multiplication in memory.
    threshold
        Threshold applied when writing edge lists.
    alpha
        Graphical Lasso regularization strength.
    output_dir
        Optional directory to persist results.
    create_heatmaps
        When ``True`` generate matplotlib heatmaps for adjacencies.

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
    mu = mu.astype(np.float32, copy=False)
    logvar = logvar.astype(np.float32, copy=False)
    results: List[NetworkResults] = []

    if "w_similarity" in methods:
        adjacency = compute_W_similarity(W, chunk_size=similarity_chunk_size)
        results.append(NetworkResults("w_similarity", adjacency))
        _persist(adjacency, genes, output_dir, "w_similarity", threshold, create_heatmaps)

    if "latent_cov" in methods:
        logvar_mean = torch.from_numpy(logvar).to(device).mean(dim=0)
        cov, corr = compute_latent_covariance(W, logvar_mean)
        results.append(NetworkResults("latent_cov", cov, {"correlation": corr}))
        _persist(cov, genes, output_dir, "latent_cov", threshold, create_heatmaps)
        if output_dir:
            save_adjacency_matrix(corr, os.path.join(output_dir, "latent_cov_correlation.csv"), genes)

    if "graphical_lasso" in methods:
        precision, covariance, adjacency = compute_graphical_lasso(mu, W, alpha=alpha)
        results.append(NetworkResults("graphical_lasso", adjacency, {"precision": precision, "covariance": covariance}))
        _persist(adjacency, genes, output_dir, "graphical_lasso", threshold, create_heatmaps)
        if output_dir:
            save_adjacency_matrix(precision, os.path.join(output_dir, "graphical_lasso_precision.csv"), genes)

    if "laplacian" in methods and getattr(model, "laplacian_matrix", None) is not None:
        adjacency = compute_laplacian_refined(W, model.laplacian_matrix.to(device))
        results.append(NetworkResults("laplacian", adjacency))
        _persist(adjacency, genes, output_dir, "laplacian", threshold, create_heatmaps)

    return results


def _persist(adjacency: np.ndarray, genes: Sequence[str], output_dir: Optional[str], prefix: str, threshold: float, create_heatmaps: bool) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
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
    "load_expression",
    "create_dataloader_from_expression",
    "run_extraction",
]
