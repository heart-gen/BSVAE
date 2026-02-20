"""Tests for simulation subpackage."""

import numpy as np
import pytest

from bsvae.simulation.generate import simulate_omics_data
from bsvae.simulation.metrics import (
    compute_ari,
    compute_nmi,
    benchmark_methods,
)


# ---------------------------------------------------------------------------
# simulate_omics_data
# ---------------------------------------------------------------------------

def test_simulate_shapes():
    X, gt, meta = simulate_omics_data(n_features=50, n_samples=30, n_modules=5, seed=0)
    assert X.shape == (50, 30)
    assert gt.shape == (50,)
    assert len(np.unique(gt)) == 5


def test_simulate_ground_truth_range():
    _, gt, _ = simulate_omics_data(n_features=100, n_samples=20, n_modules=10)
    assert gt.min() >= 0
    assert gt.max() <= 9  # 0-indexed, max K-1


def test_simulate_reproducibility():
    X1, gt1, _ = simulate_omics_data(n_features=50, n_samples=20, n_modules=4, seed=7)
    X2, gt2, _ = simulate_omics_data(n_features=50, n_samples=20, n_modules=4, seed=7)
    assert np.allclose(X1, X2)
    assert np.array_equal(gt1, gt2)


def test_simulate_different_seeds_differ():
    X1, _, _ = simulate_omics_data(seed=1)
    X2, _, _ = simulate_omics_data(seed=2)
    assert not np.allclose(X1, X2)


def test_simulate_metadata():
    _, _, meta = simulate_omics_data(n_features=60, n_samples=25, n_modules=6)
    assert meta["n_features"] == 60
    assert meta["n_samples"] == 25
    assert meta["n_modules"] == 6
    assert sum(meta["module_sizes"]) == 60


def test_simulate_within_module_correlation():
    """Features in the same module should be more correlated than across modules."""
    X, gt, _ = simulate_omics_data(
        n_features=200, n_samples=100, n_modules=5,
        within_module_correlation=0.9, between_module_correlation=0.0, seed=42
    )
    # Pick two within-module features from module 0
    m0_idx = np.where(gt == 0)[0][:2]
    m1_idx = np.where(gt == 1)[0][:2]
    corr_within = np.corrcoef(X[m0_idx[0]], X[m0_idx[1]])[0, 1]
    corr_between = np.corrcoef(X[m0_idx[0]], X[m1_idx[0]])[0, 1]
    # Within-module should be notably higher
    assert corr_within > corr_between + 0.1


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def test_ari_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert abs(compute_ari(labels, labels) - 1.0) < 1e-6


def test_nmi_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert abs(compute_nmi(labels, labels) - 1.0) < 1e-6


def test_ari_random():
    rng = np.random.default_rng(0)
    labels_true = rng.integers(0, 5, size=100)
    labels_pred = rng.integers(0, 5, size=100)
    ari = compute_ari(labels_true, labels_pred)
    # Random assignment should give low ARI (not necessarily close to 0 for small arrays)
    assert ari < 0.5


def test_benchmark_methods():
    gt = np.array([0, 0, 1, 1, 2, 2])
    preds = {
        "perfect": np.array([0, 0, 1, 1, 2, 2]),
        "wrong":   np.array([2, 2, 0, 0, 1, 1]),
    }
    results = benchmark_methods(gt, preds)
    assert "perfect" in results
    assert "wrong" in results
    assert abs(results["perfect"]["ari"] - 1.0) < 1e-5
    assert abs(results["perfect"]["nmi"] - 1.0) < 1e-5
    # "wrong" has different label names but same cluster structure → ARI = 1.0
    assert abs(results["wrong"]["ari"] - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Integration: simulate → GMM-VAE → ARI > 0.5 (smoke test)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_simulation_integration_gmm_vae():
    """
    Smoke-test: simulate → train GMMModuleVAE → benchmark.
    ARI threshold is intentionally loose (0.3) to avoid flakiness on short training.
    Use --run-slow or -m slow to include this test.
    """
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from bsvae.models.gmvae import GMMModuleVAE
    from bsvae.models.losses import GMMVAELoss
    from bsvae.utils.training import Trainer
    from bsvae.networks.module_extraction import extract_gmm_modules

    N_FEAT = 200
    N_SAMP = 100
    N_MOD = 10
    LATENT = 20

    X, gt, _ = simulate_omics_data(
        n_features=N_FEAT, n_samples=N_SAMP, n_modules=N_MOD, seed=0
    )

    feat_ids = [f"feat_{i}" for i in range(N_FEAT)]

    class TensorDataset(torch.utils.data.Dataset):
        def __len__(self): return N_FEAT
        def __getitem__(self, i):
            return torch.tensor(X[i], dtype=torch.float32), feat_ids[i]

    loader = DataLoader(TensorDataset(), batch_size=32, shuffle=True)

    model = GMMModuleVAE(
        n_features=N_SAMP, n_latent=LATENT, n_modules=N_MOD,
        hidden_dims=[64, 32], use_batch_norm=False,
    )

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=0.5, bal_strength=0.01),
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=10,
            transition_epochs=5,
        )
        trainer(loader, epochs=20)

        result = extract_gmm_modules(model, loader, feature_ids=feat_ids)
        pred = result["hard_assignments"]
        ari = compute_ari(gt, pred)
        assert ari > 0.3, f"ARI={ari:.3f} too low for basic sanity check"
