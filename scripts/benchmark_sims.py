#!/usr/bin/env python
"""
Benchmark BSVAE on one representative from each simulation type.

Scenarios
---------
1. sim1_balanced     simple block-diagonal, equal-size modules (generate.py, Gaussian)
2. sim2_nb_balanced  NB count model, balanced module sizes (GTEx-calibrated)
3. sim3_nb_long_tail NB count model, long-tail sizes (GTEx-calibrated) ← key case
4. sim4_nb_nonlinear NB long-tail + nonlinear features (GTEx-calibrated)
5. sim5_nb_confound  NB long-tail + moderate confounding (GTEx-calibrated)
6. sim6_nb_dropout   NB long-tail + high dropout/stress (GTEx-calibrated)

NB calibration from GTEx v11 Whole Blood (300 samples, expressed genes CPM≥1):
  baseline_log_mean=2.638, baseline_log_sd=1.601, libsize_log_sd=0.364,
  dispersion='high' (theta=1.5), signal_scale=1.922 → within-module r≈0.45
  Bug fix: W weights use abs(N(0,scale)) so all module members co-vary positively.

Metrics: ARI and NMI from hard γ argmax assignments.
"""

from __future__ import annotations

import sys
import tempfile
import time
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from bsvae.simulation.generate import simulate_omics_data
from bsvae.simulation.scenario import _simulate_counts, DEFAULT_CONFIG
from bsvae.simulation.metrics import compute_ari, compute_nmi
from bsvae.models.gmvae import GMMModuleVAE
from bsvae.models.losses import GMMVAELoss
from bsvae.utils.training import Trainer
from bsvae.networks.module_extraction import extract_gmm_modules

# --------------------------------------------------------------------------- #
# Shared training hyper-params (small enough to be fast, big enough to matter)
# --------------------------------------------------------------------------- #
N_FEAT    = 300
N_SAMP    = 100
N_MOD     = 10
LATENT    = 20
HIDDEN    = [64, 32]
EPOCHS    = 150
WARMUP    = 40
TRANS     = 10
FREEZE_GMM = 5       # freeze GMM prior for first N GMM epochs to stabilise after K-means
LR        = 5e-4
BATCH     = 32
SEEDS     = [0, 1, 2, 3, 4]   # multiple seeds — report mean ± std

# GTEx v11 Whole Blood-calibrated NB parameters (see scripts/calibrate_nb_from_gtex.py)
# baseline_log_mean/sd estimated from expressed genes (CPM≥1) in 300 WB samples.
# signal_scale chosen to achieve within-module Pearson r≈0.45 after fixing the
# W-sign bug (abs-normal weights so module members co-vary positively).
GEN_CFG = {
    **DEFAULT_CONFIG["generator"],
    "n_features": N_FEAT,
    "n_modules": N_MOD,
    "baseline_log_mean": 2.638,
    "baseline_log_sd":   1.601,
    "libsize_log_sd":    0.364,
    "dispersion":        "high",
    "signal_scale":      1.922,
}


# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #
class ArrayDataset(Dataset):
    """Wraps a (features × samples) float32 array."""
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ids = [f"feat_{i}" for i in range(len(X))]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.ids[i]


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def _kmeans_on_mu(model: GMMModuleVAE, eval_loader: DataLoader) -> np.ndarray:
    """Extract encoder μ for all features and run K-means (like benchmarking repo)."""
    model.eval()
    all_mu = []
    with torch.no_grad():
        for xb, _ in eval_loader:
            mu, _ = model.encoder(model._maybe_normalize(xb))
            all_mu.append(mu.numpy())
    mu_all = np.concatenate(all_mu, axis=0)
    return KMeans(n_clusters=N_MOD, n_init=10, random_state=0).fit_predict(mu_all)


def train_and_score(
    X: np.ndarray,
    ground_truth: np.ndarray,
    seed: int,
    normalize_input: bool = False,
    masked_recon: bool = False,
    corr_strength: float = 0.0,
    latent_corr_strength: float = 0.0,
) -> dict:
    """Train one BSVAE config and return ARI/NMI for both γ-argmax and K-means-on-μ."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ArrayDataset(X)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=False)
    eval_loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, drop_last=False)
    n_samp  = X.shape[1]

    model = GMMModuleVAE(
        n_features=n_samp,
        n_latent=LATENT,
        n_modules=N_MOD,
        hidden_dims=HIDDEN,
        use_batch_norm=True,
        normalize_input=normalize_input,
    )

    loss_fn = GMMVAELoss(
        beta=1.0,
        free_bits=0.5,
        sep_strength=0.0,
        bal_strength=0.1,
        pi_entropy_strength=0.01,
        normalize_input=normalize_input,
        masked_recon=masked_recon,
        corr_strength=corr_strength,
        latent_corr_strength=latent_corr_strength,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=LR),
            gmm_loss_f=loss_fn,
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=WARMUP,
            transition_epochs=TRANS,
            freeze_gmm_epochs=FREEZE_GMM,
        )
        trainer(loader, epochs=EPOCHS)
        result = extract_gmm_modules(model, eval_loader, feature_ids=dataset.ids)
        pred_gamma = result["hard_assignments"]
        pred_km    = _kmeans_on_mu(model, eval_loader)

    return {
        "ari":    compute_ari(ground_truth, pred_gamma),
        "nmi":    compute_nmi(ground_truth, pred_gamma),
        "ari_km": compute_ari(ground_truth, pred_km),
        "nmi_km": compute_nmi(ground_truth, pred_km),
    }


# Three configs: baseline, normalize only, normalize + latent correlation loss
CONFIGS = [
    ("original",  dict(normalize_input=False, masked_recon=False, corr_strength=0.0, latent_corr_strength=0.0)),
    ("norm_only", dict(normalize_input=True,  masked_recon=False, corr_strength=0.0, latent_corr_strength=0.0)),
    ("norm_corr", dict(normalize_input=True,  masked_recon=False, corr_strength=0.0, latent_corr_strength=0.5)),
]


def run_scenario(X, gt, cfg_name, normalize_input=False, masked_recon=False,
                 corr_strength=0.0, latent_corr_strength=0.0, **_):
    """Run across all seeds; return (ari_gamma, ari_kmeans) arrays."""
    aris_g, aris_km, nmis_g = [], [], []
    for s in SEEDS:
        r = train_and_score(X, gt, seed=s,
                            normalize_input=normalize_input,
                            masked_recon=masked_recon,
                            corr_strength=corr_strength,
                            latent_corr_strength=latent_corr_strength)
        aris_g.append(r["ari"])
        aris_km.append(r["ari_km"])
        nmis_g.append(r["nmi"])
        print(f"      seed={s}  γ-ARI={r['ari']:.3f}  μ-KM-ARI={r['ari_km']:.3f}  NMI={r['nmi']:.3f}", flush=True)
    return np.array(aris_g), np.array(aris_km), np.array(nmis_g)


# --------------------------------------------------------------------------- #
# Simulation builders
# --------------------------------------------------------------------------- #
def sim1_balanced():
    X, gt, _ = simulate_omics_data(
        n_features=N_FEAT, n_samples=N_SAMP, n_modules=N_MOD,
        within_module_correlation=0.8, between_module_correlation=0.0,
        noise_std=0.2, seed=42,
    )
    return X, gt


def _nb_sim(**overrides):
    cfg = {**GEN_CFG, "module_size_distribution": "long_tail",
           "nonlinear_mode": "off", "dropout_mode": "off",
           "dropout_target": 0.0, "confounding": "none", **overrides}
    r = _simulate_counts(
        n_features=cfg["n_features"],
        n_samples=N_SAMP,
        n_modules=cfg["n_modules"],
        signal_scale=cfg.get("signal_scale", GEN_CFG["signal_scale"]),
        overlap_rate=cfg.get("overlap_rate", 0.1),
        confounding=cfg["confounding"],
        nonlinear_mode=cfg["nonlinear_mode"],
        dropout_mode=cfg["dropout_mode"],
        dropout_target=cfg.get("dropout_target", 0.0),
        seed=42,
        generator_cfg=cfg,
    )
    X   = r.expr_features_x_samples.values.astype(np.float32)
    gt  = r.modules_hard["module"].values.astype(int)
    return X, gt


def sim2_nb_balanced():
    return _nb_sim(module_size_distribution="balanced")

def sim3_nb_long_tail():
    return _nb_sim()

def sim4_nb_nonlinear():
    return _nb_sim(nonlinear_mode="on")

def sim5_nb_confound():
    return _nb_sim(confounding="moderate")

def sim6_nb_dropout():
    return _nb_sim(dropout_mode="high", dropout_target=0.5)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
SCENARIOS = [
    ("sim1_balanced",     sim1_balanced,   "Equal-size modules, Gaussian noise"),
    ("sim2_nb_balanced",  sim2_nb_balanced,"NB counts, balanced modules"),
    ("sim3_nb_long_tail", sim3_nb_long_tail,"NB counts, long-tail modules ★"),
    ("sim4_nb_nonlinear", sim4_nb_nonlinear,"NB long-tail + nonlinear"),
    ("sim5_nb_confound",  sim5_nb_confound, "NB long-tail + confounding"),
    ("sim6_nb_dropout",   sim6_nb_dropout,  "NB long-tail + high dropout"),
]

def main():
    print(f"\n{'='*80}", flush=True)
    print(f"BSVAE Simulation Benchmark  — GTEx-calibrated NB + Gaussian baseline", flush=True)
    print(f"  {N_FEAT} features × {N_SAMP} samples  K={N_MOD}  "
          f"{EPOCHS} epochs  warmup={WARMUP}  freeze_gmm={FREEZE_GMM}  n={len(SEEDS)} seeds", flush=True)
    print(f"{'='*80}\n", flush=True)

    # rows[scenario][cfg_name] = (ari_mean, ari_std, nmi_mean)
    results = {}

    for sim_name, sim_fn, sim_desc in SCENARIOS:
        print(f"\n--- {sim_name}  ({sim_desc}) ---", flush=True)
        X, gt = sim_fn()
        sizes = np.bincount(gt)
        print(f"  module sizes: min={sizes.min()} max={sizes.max()} "
              f"mean={sizes.mean():.1f} std={sizes.std():.1f}", flush=True)
        results[sim_name] = {}

        for cfg_name, cfg_kwargs in CONFIGS:
            t0 = time.time()
            aris_g, aris_km, nmis_g = run_scenario(X, gt, cfg_name, **cfg_kwargs)
            elapsed = time.time() - t0
            results[sim_name][cfg_name] = (aris_g.mean(), aris_g.std(),
                                           aris_km.mean(), aris_km.std(),
                                           nmis_g.mean())
            print(f"  {cfg_name:<12}  γ-ARI={aris_g.mean():.3f}±{aris_g.std():.3f}  "
                  f"μ-KM-ARI={aris_km.mean():.3f}±{aris_km.std():.3f}  "
                  f"NMI={nmis_g.mean():.3f}  ({elapsed:.0f}s)", flush=True)

    # Summary table — show both γ-argmax and K-means-on-μ ARI side by side
    ncols = len(CONFIGS)
    width = 36 + ncols * 36
    print(f"\n\n{'='*width}", flush=True)
    print(f"SUMMARY  (mean ARI over {len(SEEDS)} seeds)  |  γ=GMM posterior argmax  |  μKM=K-means on encoder μ", flush=True)
    print(f"{'='*width}", flush=True)
    hdr = f"  {'Scenario':<22}  {'SzStd':>5}  " + "  ".join(
        f"{'γ-'+c:>14}  {'μKM-'+c:>14}" for c, _ in CONFIGS)
    print(hdr, flush=True)
    print("  " + "-"*(width - 2), flush=True)
    for sim_name, sim_fn, _ in SCENARIOS:
        X, gt = sim_fn()
        std = np.bincount(gt).std()
        row = f"  {sim_name:<22}  {std:>5.1f}  "
        for cfg_name, _ in CONFIGS:
            ag_m, ag_s, akm_m, akm_s, _ = results[sim_name][cfg_name]
            row += f"  {ag_m:.3f}±{ag_s:.3f}        {akm_m:.3f}±{akm_s:.3f}  "
        print(row, flush=True)
    print(f"{'='*width}\n", flush=True)


if __name__ == "__main__":
    main()
