"""
Network extraction for GMMModuleVAE.

Two methods:
  A — Latent mean cosine similarity:   S_ij = cos(μ_i, μ_j)
  B — FAISS HNSW kNN in γ-space:      S_ij = γ_i · γ_j (after L2-normalisation)

Both operate on (μ, γ) matrices extracted from the model and save
sparse NPZ files as output.

Legacy methods (w_similarity, latent_cov, graphical_lasso, laplacian) are
removed; only GMM-based methods are supported in this version.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NetworkResults:
    """Container for a sparse adjacency matrix."""
    method: str
    adjacency: sp.csr_matrix  # sparse, shape (F, F)
    aux: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------

def extract_mu_gamma(
    model,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    disable_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract encoder means (μ) and GMM soft assignments (γ) from all features.

    Parameters
    ----------
    model : GMMModuleVAE
    dataloader : DataLoader
        Each batch yields (profiles, feature_ids).
    device : torch.device or None
    disable_progress : bool

    Returns
    -------
    mu : np.ndarray, shape (F, D)
    gamma : np.ndarray, shape (F, K)
    feature_ids : list of str
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    mu_list, gamma_list, fid_list = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, disable=disable_progress, desc="Extracting latents"):
            if isinstance(batch, (list, tuple)):
                x, feature_ids = batch[0], batch[1]
            else:
                x = batch
                feature_ids = []
            x = x.to(device)
            mu, logvar = model.encode(x)
            gamma = model.gmm_prior.posterior_weights(mu, logvar)
            mu_list.append(mu.cpu().numpy())
            gamma_list.append(gamma.cpu().numpy())
            if isinstance(feature_ids, (list, tuple)):
                fid_list.extend(feature_ids)
            else:
                fid_list.extend(feature_ids.tolist() if hasattr(feature_ids, 'tolist') else list(feature_ids))

    return (
        np.concatenate(mu_list, axis=0).astype(np.float32),
        np.concatenate(gamma_list, axis=0).astype(np.float32),
        fid_list,
    )


# ---------------------------------------------------------------------------
# Method A — Latent mean cosine similarity
# ---------------------------------------------------------------------------

def method_a_cosine(
    mu: np.ndarray,
    top_k: int = 50,
    chunk_size: int = 1000,
) -> sp.csr_matrix:
    """
    Compute sparse top-K cosine similarity network from encoder means.

    For each feature i, keep its K highest-similarity neighbours.
    Resulting matrix is symmetrised (max of A and A.T).

    Parameters
    ----------
    mu : np.ndarray, shape (F, D)
    top_k : int
        Edges to keep per feature. Default: 50.
    chunk_size : int
        Chunk size for chunked computation. Default: 1000.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (F, F)
    """
    F, D = mu.shape

    # L2-normalise
    norms = np.linalg.norm(mu, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    mu_norm = mu / norms  # (F, D)

    rows, cols, vals = [], [], []

    for start in range(0, F, chunk_size):
        end = min(start + chunk_size, F)
        chunk = mu_norm[start:end]            # (C, D)
        sim = chunk @ mu_norm.T               # (C, F)

        # Top-K per row in the chunk
        for local_i, row_sim in enumerate(sim):
            global_i = start + local_i
            row_sim[global_i] = -np.inf       # exclude self
            top_idx = np.argpartition(row_sim, -top_k)[-top_k:]
            for j in top_idx:
                s = row_sim[j]
                if s > 0:
                    rows.append(global_i)
                    cols.append(j)
                    vals.append(float(s))

    A = sp.csr_matrix(
        (vals, (rows, cols)), shape=(F, F), dtype=np.float32
    )
    # Symmetrise
    A = A.maximum(A.T)
    return A


# ---------------------------------------------------------------------------
# Method B — FAISS HNSW kNN in γ-space
# ---------------------------------------------------------------------------

def method_b_gamma_knn(
    gamma: np.ndarray,
    top_k: int = 50,
    hnsw_m: int = 32,
) -> sp.csr_matrix:
    """
    Build a kNN graph in γ-space using FAISS HNSW (cosine similarity).

    Parameters
    ----------
    gamma : np.ndarray, shape (F, K)
    top_k : int
        Number of nearest neighbours per feature. Default: 50.
    hnsw_m : int
        FAISS HNSW graph degree parameter. Default: 32.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix, shape (F, F)
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss-cpu is required for Method B: pip install faiss-cpu"
        ) from e

    F, K = gamma.shape

    # L2-normalise γ rows so inner product = cosine similarity
    norms = np.linalg.norm(gamma, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    gamma_norm = np.ascontiguousarray(gamma / norms, dtype=np.float32)

    # Build HNSW index (inner product after normalisation = cosine)
    index = faiss.IndexHNSWFlat(K, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.add(gamma_norm)

    # Query all features for their top-K+1 neighbours (includes self)
    k_query = min(top_k + 1, F)
    distances, indices = index.search(gamma_norm, k_query)

    rows, cols, vals = [], [], []
    for i in range(F):
        for rank in range(k_query):
            j = int(indices[i, rank])
            s = float(distances[i, rank])
            if j == i or j < 0:
                continue
            if s > 0:
                rows.append(i)
                cols.append(j)
                vals.append(s)

    A = sp.csr_matrix(
        (vals, (rows, cols)), shape=(F, F), dtype=np.float32
    )
    # Symmetrise
    A = A.maximum(A.T)
    return A


# ---------------------------------------------------------------------------
# NPZ persistence
# ---------------------------------------------------------------------------

def save_adjacency_npz(
    adjacency: sp.spmatrix,
    path: str,
    feature_ids: Optional[List[str]] = None,
) -> None:
    """Save a sparse adjacency matrix as compressed NPZ."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    coo = adjacency.tocoo()
    arrays = dict(
        row=coo.row.astype(np.int32),
        col=coo.col.astype(np.int32),
        data=coo.data.astype(np.float32),
        shape=np.array(coo.shape, dtype=np.int64),
    )
    if feature_ids is not None:
        arrays["feature_ids"] = np.array(feature_ids, dtype=object)
    np.savez_compressed(path, **arrays)


def load_adjacency_npz(path: str) -> Tuple[sp.csr_matrix, Optional[List[str]]]:
    """Load a sparse adjacency saved with save_adjacency_npz."""
    npz = np.load(path, allow_pickle=True)
    shape = tuple(npz["shape"])
    A = sp.csr_matrix(
        (npz["data"], (npz["row"], npz["col"])),
        shape=shape,
        dtype=np.float32,
    )
    feature_ids = list(npz["feature_ids"]) if "feature_ids" in npz else None
    return A, feature_ids


# ---------------------------------------------------------------------------
# High-level extraction pipeline
# ---------------------------------------------------------------------------

def run_extraction(
    model,
    dataloader: DataLoader,
    feature_ids: Optional[List[str]] = None,
    methods: Sequence[str] = ("mu_cosine",),
    top_k: int = 50,
    output_dir: Optional[str] = None,
    disable_progress: bool = False,
) -> List[NetworkResults]:
    """
    Extract sparse networks from a trained GMMModuleVAE.

    Parameters
    ----------
    model : GMMModuleVAE
    dataloader : DataLoader
    feature_ids : list of str or None
    methods : sequence of str
        Subset of {"mu_cosine", "gamma_knn"}.
    top_k : int
        Top-K edges per feature for both methods.
    output_dir : str or None
        If provided, save NPZ files here.
    disable_progress : bool

    Returns
    -------
    results : list of NetworkResults
    """
    device = next(model.parameters()).device
    mu, gamma, extracted_ids = extract_mu_gamma(
        model, dataloader, device=device, disable_progress=disable_progress
    )
    if feature_ids is None:
        feature_ids = extracted_ids

    results = []
    for method in methods:
        _LOG.info("Running network extraction method: %s", method)
        if method in ("mu_cosine", "a", "method_a"):
            adj = method_a_cosine(mu, top_k=top_k)
        elif method in ("gamma_knn", "b", "method_b"):
            adj = method_b_gamma_knn(gamma, top_k=top_k)
        else:
            _LOG.warning("Unknown method '%s'; skipping.", method)
            continue

        result = NetworkResults(method=method, adjacency=adj)
        results.append(result)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{method}_adjacency.npz")
            save_adjacency_npz(adj, out_path, feature_ids)
            _LOG.info("Saved %s adjacency to %s", method, out_path)

    return results


# ---------------------------------------------------------------------------
# Legacy helpers for CLI backward compatibility
# ---------------------------------------------------------------------------

def load_expression(path: str):
    """Load an expression matrix (features × samples) as a pandas DataFrame."""
    import pandas as pd
    from pathlib import Path
    suffixes = [s.lower() for s in Path(path).suffixes]
    sep = "\t" if ".tsv" in suffixes else ","
    return pd.read_csv(path, index_col=0, sep=sep)


def create_dataloader_from_expression(
    path: str,
    batch_size: int = 128,
    shuffle: bool = False,
) -> Tuple:
    """
    Create a DataLoader from an expression matrix for inference.

    Returns (dataloader, feature_ids, sample_ids).
    """
    from bsvae.utils.datasets import OmicsDataset
    dataset = OmicsDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.feature_ids, dataset.sample_ids
