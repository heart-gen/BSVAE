"""Tests for network extraction and module extraction (GMM-VAE version)."""

import tempfile
import os
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bsvae.networks.module_extraction import (
    format_module_feedback,
    extract_gmm_modules,
    compute_module_eigengenes_from_soft,
)
from bsvae.networks.extract_networks import (
    method_a_cosine,
    save_adjacency_npz,
    load_adjacency_npz,
    extract_mu_gamma,
)
from bsvae.models.gmvae import GMMModuleVAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_model(n_samples=20, n_latent=6, n_modules=4):
    return GMMModuleVAE(
        n_features=n_samples, n_latent=n_latent, n_modules=n_modules,
        hidden_dims=[16], use_batch_norm=False,
    )


def _make_loader(n_features=20, n_samples=20, batch_size=8):
    profiles = torch.randn(n_features, n_samples)
    feat_ids = [f"f{i}" for i in range(n_features)]

    class DS(torch.utils.data.Dataset):
        def __len__(self): return n_features
        def __getitem__(self, i): return profiles[i], feat_ids[i]

    return DataLoader(DS(), batch_size=batch_size, shuffle=False), feat_ids


# ---------------------------------------------------------------------------
# format_module_feedback
# ---------------------------------------------------------------------------

def test_format_module_feedback_includes_resolution():
    modules = pd.Series([0] * 80 + [1] * 120, index=[f"g{i}" for i in range(200)])
    msg = format_module_feedback("Leiden", modules, resolution=1.0)
    assert "resolution=1.0" in msg
    assert "2 modules" in msg


def test_format_module_feedback_no_details():
    modules = pd.Series([0, 1, 2], index=["a", "b", "c"])
    msg = format_module_feedback("GMM", modules)
    assert "3 modules" in msg


# ---------------------------------------------------------------------------
# Method A (μ cosine)
# ---------------------------------------------------------------------------

def test_method_a_cosine_shape():
    F, D = 30, 8
    mu = np.random.randn(F, D).astype(np.float32)
    A = method_a_cosine(mu, top_k=5, chunk_size=10)
    assert A.shape == (F, F)


def test_method_a_cosine_symmetric():
    mu = np.random.randn(20, 6).astype(np.float32)
    A = method_a_cosine(mu, top_k=5)
    diff = np.abs(A - A.T)
    assert diff.max() < 1e-5


def test_method_a_cosine_no_self_loops():
    mu = np.random.randn(15, 4).astype(np.float32)
    A = method_a_cosine(mu, top_k=3)
    assert A.diagonal().sum() == 0


# ---------------------------------------------------------------------------
# NPZ persistence
# ---------------------------------------------------------------------------

def test_save_load_adjacency_npz(tmp_path):
    import scipy.sparse as sp
    F = 10
    rows = [0, 1, 2]
    cols = [1, 2, 3]
    vals = [0.5, 0.6, 0.7]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(F, F))
    A = A.maximum(A.T)

    path = str(tmp_path / "adj.npz")
    feat_ids = [f"feat_{i}" for i in range(F)]
    save_adjacency_npz(A, path, feature_ids=feat_ids)

    A_loaded, ids_loaded = load_adjacency_npz(path)
    assert A_loaded.shape == (F, F)
    assert ids_loaded == feat_ids


def test_load_adjacency_npz_without_ids(tmp_path):
    import scipy.sparse as sp
    A = sp.eye(5, format="csr", dtype=np.float32)
    path = str(tmp_path / "adj.npz")
    save_adjacency_npz(A, path)
    A_loaded, ids = load_adjacency_npz(path)
    assert ids is None


# ---------------------------------------------------------------------------
# extract_mu_gamma
# ---------------------------------------------------------------------------

def test_extract_mu_gamma_shapes():
    model = _make_small_model(n_samples=20, n_latent=6, n_modules=4)
    loader, feat_ids = _make_loader(n_features=20, n_samples=20)
    model.eval()
    mu, gamma, ids = extract_mu_gamma(model, loader, disable_progress=True)
    assert mu.shape == (20, 6)
    assert gamma.shape == (20, 4)
    assert len(ids) == 20


def test_extract_mu_gamma_gamma_sums_to_one():
    model = _make_small_model(n_samples=20, n_latent=6, n_modules=4)
    loader, _ = _make_loader(n_features=20, n_samples=20)
    model.eval()
    _, gamma, _ = extract_mu_gamma(model, loader, disable_progress=True)
    row_sums = gamma.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# extract_gmm_modules
# ---------------------------------------------------------------------------

def test_extract_gmm_modules_shapes():
    model = _make_small_model()
    loader, feat_ids = _make_loader()
    result = extract_gmm_modules(model, loader, feature_ids=feat_ids)
    assert result["gamma"].shape == (20, 4)
    assert result["hard_assignments"].shape == (20,)
    assert len(result["feature_ids"]) == 20


def test_extract_gmm_modules_saves_to_disk(tmp_path):
    model = _make_small_model()
    loader, feat_ids = _make_loader()
    extract_gmm_modules(model, loader, feature_ids=feat_ids, output_dir=str(tmp_path))
    assert os.path.exists(str(tmp_path / "gamma.npz"))
    assert os.path.exists(str(tmp_path / "hard_assignments.npz"))


def test_extract_gmm_modules_hard_in_range():
    model = _make_small_model(n_modules=4)
    loader, _ = _make_loader()
    result = extract_gmm_modules(model, loader)
    hard = result["hard_assignments"]
    assert (hard >= 0).all() and (hard < 4).all()


# ---------------------------------------------------------------------------
# compute_module_eigengenes_from_soft
# ---------------------------------------------------------------------------

def test_compute_soft_eigengenes_shape():
    n_feat, n_samp, K = 30, 20, 4
    feat_ids = [f"feat_{i}" for i in range(n_feat)]
    sample_ids = [f"sample_{j}" for j in range(n_samp)]
    expr = pd.DataFrame(
        np.random.randn(n_feat, n_samp),
        index=feat_ids, columns=sample_ids,
    )
    gamma = np.random.dirichlet(np.ones(K), size=n_feat).astype(np.float32)
    eigengenes = compute_module_eigengenes_from_soft(expr, gamma, feat_ids)
    assert eigengenes.shape == (n_samp, K)
    assert list(eigengenes.index) == sample_ids
