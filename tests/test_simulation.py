"""Tests for simulation subpackage."""

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# Integration: isoform-hierarchy simulation → BSVAE-hier training
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("switch_rate", [0.0, 0.15, 0.30])
def test_simulation_isoform_hier_training(switch_rate):
    """
    Smoke-test: isoform-hierarchy data → BSVAE-generic and BSVAE-hier training.

    Parametrized over switch_rate (0.0 = full concordance, 0.15 = realistic
    brain tissue, 0.30 = moderate switching).

    Verifies:
    1. Both training runs complete without error at each switch rate.
    2. hierarchical_loss is non-negative and > 0 when isoform pairs present.
    3. Both models produce module assignments (ARI > 0 loose threshold).
    4. Actual fraction of switched transcripts is within ±10 pp of switch_rate.

    Use --run-slow or -m slow to include this test.
    """
    import tempfile

    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    from bsvae.models.gmvae import GMMModuleVAE
    from bsvae.models.losses import GMMVAELoss, hierarchical_loss
    from bsvae.networks.module_extraction import extract_gmm_modules
    from bsvae.utils.hierarchy import group_isoforms_by_gene
    from bsvae.utils.training import Trainer

    # ── Small isoform-hierarchy dataset ──────────────────────────────────────
    N_GENES  = 24           # genes
    N_ISO    = 3            # isoforms per gene (fixed, as in Sim 6)
    N_TX     = N_GENES * N_ISO  # 72 transcripts
    N_SAMP   = 80
    N_MOD    = 4
    K_LATENT = 8
    rng = np.random.default_rng(42)

    # Gene-level module assignments (balanced)
    gene_modules = np.tile(np.arange(N_MOD), N_GENES // N_MOD)   # (N_GENES,)
    gene_of_tx   = np.repeat(np.arange(N_GENES), N_ISO)           # (N_TX,)
    tx_modules   = gene_modules[gene_of_tx].copy()                 # (N_TX,)

    # Isoform switching (mirrors simulate_isoform.py step 3)
    if switch_rate > 0.0:
        switch_mask = rng.random(N_GENES) < switch_rate
        for g in np.where(switch_mask)[0]:
            tx_start = g * N_ISO
            iso_idx  = tx_start + rng.integers(N_ISO)
            orig_mod = tx_modules[iso_idx]
            other    = [m for m in range(N_MOD) if m != orig_mod]
            if other:
                tx_modules[iso_idx] = rng.choice(other)

    n_switched = int(np.sum(tx_modules != gene_modules[gene_of_tx]))
    frac_switched = n_switched / N_TX

    # NB count simulation (GTEx-calibrated, mirrors simulate_isoform.py)
    signal_scale = 2.0
    W     = np.zeros((N_TX, N_MOD))
    for i in range(N_TX):
        W[i, tx_modules[i]] = abs(rng.normal(0, signal_scale))

    Z       = rng.normal(0, 1, (N_SAMP, N_MOD))
    alpha   = rng.normal(2.6, 1.6, N_TX)
    libsize = np.exp(rng.normal(0, 0.36, N_SAMP))
    eta     = Z @ W.T + alpha[None, :] + np.log(libsize)[:, None]
    mu_nb   = np.exp(np.clip(eta, -10, 10))

    log_theta = -0.5 + 0.3 * alpha + rng.normal(0, 0.3, N_TX)
    theta     = np.exp(log_theta).clip(0.3, 20.0)
    p_nb      = theta[None, :] / (theta[None, :] + mu_nb)
    counts    = rng.negative_binomial(theta[None, :], p_nb).astype(np.float32)

    row_sums = counts.sum(axis=1, keepdims=True)
    cpm      = counts / np.maximum(row_sums, 1.0) * 1e6
    X_log    = np.log2(cpm + 1.0).T          # (N_TX, N_SAMP)

    # ── Feature IDs and tx2gene ───────────────────────────────────────────────
    tx_ids   = [f"ENST{i:07d}.1"  for i in range(N_TX)]
    gene_ids = [f"ENSG{g:07d}.1"  for g in range(N_GENES)]
    tx2gene  = pd.Series(
        [gene_ids[g] for g in gene_of_tx],
        index=tx_ids,
        name="gene_id",
    )
    gene_groups = group_isoforms_by_gene(tx2gene, tx_ids)
    assert len(gene_groups) == N_GENES, "All genes should have 3 isoforms"

    # Switched fraction should be within ±10 pp of switch_rate (each switch
    # affects 1/3 isoforms of a gene → frac_switched ≈ switch_rate / N_ISO)
    if switch_rate == 0.0:
        assert frac_switched == 0.0, "No switching expected when switch_rate=0"
    else:
        expected_frac = switch_rate / N_ISO   # ~1/3 of affected transcripts switch
        assert abs(frac_switched - expected_frac) < 0.10, (
            f"switch_rate={switch_rate}: frac_switched={frac_switched:.3f}, "
            f"expected ~{expected_frac:.3f}"
        )

    # ── Dataset (exposes .feature_ids for Trainer._get_feature_id_to_idx) ────
    class IsoformDataset(torch.utils.data.Dataset):
        def __init__(self, X, feat_ids):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.feature_ids = feat_ids

        def __len__(self):
            return len(self.feature_ids)

        def __getitem__(self, i):
            return self.X[i], self.feature_ids[i]

    dataset = IsoformDataset(X_log, tx_ids)

    def make_loader(shuffle=True):
        return DataLoader(dataset, batch_size=36, shuffle=shuffle)

    # ── 1. Verify hierarchical_loss is non-negative on a fresh model ──────────
    model_check = GMMModuleVAE(
        n_features=N_SAMP, n_latent=K_LATENT, n_modules=N_MOD,
        hidden_dims=[32, 16], use_batch_norm=False,
    )
    model_check.eval()
    with torch.no_grad():
        x_batch, id_batch = next(iter(make_loader(shuffle=False)))
        _, mu_check, _, _, _ = model_check(x_batch)
        id_to_idx = {fid: i for i, fid in enumerate(tx_ids)}
        feat_idx  = torch.tensor([id_to_idx[fid] for fid in id_batch], dtype=torch.long)
        h_loss    = hierarchical_loss(mu_check, gene_groups, feat_idx)
    assert h_loss.item() >= 0.0, "hierarchical_loss must be non-negative"
    # A batch of 36 from 72 features (3 isoforms/gene) will contain isoform pairs
    assert h_loss.item() > 0.0, "hierarchical_loss must be > 0 when isoform pairs present"

    # ── 2. Train BSVAE-hier (with hierarchy) ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        model_hier = GMMModuleVAE(
            n_features=N_SAMP, n_latent=K_LATENT, n_modules=N_MOD,
            hidden_dims=[32, 16], use_batch_norm=False,
        )
        trainer_hier = Trainer(
            model=model_hier,
            optimizer=optim.Adam(model_hier.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=0.5, bal_strength=0.01, hier_strength=0.05),
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=5,
            transition_epochs=3,
            gene_groups=gene_groups,
        )
        trainer_hier(make_loader(), epochs=15)
    assert not model_hier.training

    # ── 3. Train BSVAE-generic (no hierarchy) ────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        model_gen = GMMModuleVAE(
            n_features=N_SAMP, n_latent=K_LATENT, n_modules=N_MOD,
            hidden_dims=[32, 16], use_batch_norm=False,
        )
        trainer_gen = Trainer(
            model=model_gen,
            optimizer=optim.Adam(model_gen.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=0.5, bal_strength=0.01, hier_strength=0.0),
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=5,
            transition_epochs=3,
        )
        trainer_gen(make_loader(), epochs=15)
    assert not model_gen.training

    # ── 4. Both models produce valid module assignments (loose ARI > 0) ───────
    loader_eval = make_loader(shuffle=False)
    for model, label in [(model_hier, "hier"), (model_gen, "generic")]:
        result = extract_gmm_modules(model, loader_eval, feature_ids=tx_ids)
        pred   = result["hard_assignments"]
        ari    = compute_ari(tx_modules, pred)
        assert ari > 0.0, (
            f"BSVAE-{label}: ARI={ari:.3f} — module assignments appear random"
        )
