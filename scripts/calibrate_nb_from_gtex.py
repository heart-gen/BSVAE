#!/usr/bin/env python3
"""
calibrate_nb_from_gtex.py — Estimate NB simulation parameters from GTEx v11.

Reads Whole Blood samples from GTEx gene_reads GCT (compressed), fits
per-gene Negative Binomial parameters for expressed genes, and finds the
signal_scale that produces realistic within-module Pearson r (targeting 0.4–0.5)
in the bsvae NB simulator.

Outputs a JSON with recommended parameters for benchmark_sims.py.

Usage:
    python scripts/calibrate_nb_from_gtex.py
    python scripts/calibrate_nb_from_gtex.py --n-samples 300 --target-r 0.45 --out calibrated_params.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BENCHMARK_DIR = Path("/ocean/projects/bio250020p/kbenjamin/projects/bsvae-benchmarking")
COUNTS_GCT    = BENCHMARK_DIR / "inputs/gtex_v11/counts/GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_reads.gct.gz"
METADATA_TSV  = BENCHMARK_DIR / "inputs/gtex_v11/metadata/GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
TISSUE        = "Whole Blood"
N_SAMPLES_CAL = 300    # number of GTEx samples to use for calibration
N_GENES_CAL   = 5000   # cap on expressed genes to keep
MIN_CPM       = 1.0    # genes with mean CPM below this are excluded
SEED          = 42


# --------------------------------------------------------------------------- #
# Step 1: Load sample metadata, select tissue
# --------------------------------------------------------------------------- #
def get_tissue_samples(n: int, tissue: str = TISSUE, seed: int = SEED) -> list[str]:
    meta = pd.read_csv(METADATA_TSV, sep="\t", usecols=["SAMPID", "SMTSD"])
    wb = meta[meta["SMTSD"] == tissue]["SAMPID"].tolist()

    # Read GCT header to get available sample IDs
    with gzip.open(COUNTS_GCT, "rt") as fh:
        fh.readline(); fh.readline()
        gct_header = fh.readline().rstrip("\n").split("\t")[2:]  # skip Name, Description
    gct_set = set(gct_header)

    # Only keep samples present in both metadata and GCT
    wb_in_gct = [s for s in wb if s in gct_set]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(wb_in_gct, size=min(n, len(wb_in_gct)), replace=False).tolist()
    print(f"[cal] Tissue='{tissue}'  metadata={len(wb)}  in_GCT={len(wb_in_gct)}  selected={len(chosen)}", flush=True)
    return chosen


# --------------------------------------------------------------------------- #
# Step 2: Stream GCT, keep only selected columns
# --------------------------------------------------------------------------- #
def load_gct_subset(gct_path: Path, keep_samples: list[str]) -> pd.DataFrame:
    """Returns raw count matrix (genes × samples), columns = sample IDs."""
    keep_set = set(keep_samples)
    print(f"[cal] Reading GCT header …", flush=True)

    with gzip.open(gct_path, "rt") as fh:
        fh.readline()  # #1.2
        fh.readline()  # nrow ncol
        header = fh.readline().rstrip("\n").split("\t")

    # Find column indices for selected samples (cols 0=Name, 1=Description, then samples)
    # col 0=Name (→ index), col 1=Description (skip), then sample cols
    keep_set_local = {"Name", "Description"} | keep_set
    col_indices = [i for i, h in enumerate(header) if h in keep_set_local]

    found = sum(1 for h in header if h in keep_set)
    print(f"[cal] Matched {found}/{len(keep_samples)} requested samples in GCT header", flush=True)
    if found == 0:
        raise RuntimeError("No matching samples found — check sample IDs vs GCT header")

    # Read only those columns (skip rows 0-1 which are #1.2 and nrow/ncol)
    print(f"[cal] Loading {found} columns × 74k genes …", flush=True)
    df = pd.read_csv(
        gct_path,
        sep="\t",
        skiprows=2,
        usecols=col_indices,
        index_col=0,
        compression="gzip",
    )
    if "Description" in df.columns:
        df = df.drop(columns=["Description"])
    df = df.astype(np.float32)
    print(f"[cal] Loaded  shape={df.shape}", flush=True)
    return df


# --------------------------------------------------------------------------- #
# Step 3: QC + CPM normalisation
# --------------------------------------------------------------------------- #
def compute_cpm_and_filter(counts: pd.DataFrame, min_cpm: float = MIN_CPM,
                           n_genes_cap: int = N_GENES_CAL, seed: int = SEED):
    """Returns (filtered_counts, libsizes, expressed_genes_df)."""
    libsizes = counts.sum(axis=0).values.astype(np.float64)  # per sample
    cpm = counts.div(libsizes / 1e6, axis=1)
    mean_cpm = cpm.mean(axis=1)
    expressed = cpm.loc[mean_cpm >= min_cpm]
    print(f"[cal] Expressed genes (mean CPM ≥ {min_cpm}): {len(expressed)}/{len(counts)}", flush=True)

    if len(expressed) > n_genes_cap:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(expressed), size=n_genes_cap, replace=False)
        expressed = expressed.iloc[idx]
        print(f"[cal] Subsampled to {n_genes_cap} genes", flush=True)

    return expressed.values.astype(np.float32), libsizes, expressed


# --------------------------------------------------------------------------- #
# Step 4: Fit NB parameters (method of moments)
# --------------------------------------------------------------------------- #
def fit_nb_params(counts: np.ndarray, libsizes: np.ndarray):
    """
    Fit per-gene NB parameters.

    Returns
    -------
    baseline_log_mean : float   mean of log(mu_g / geom_mean_libsize)
    baseline_log_sd   : float   std  of above
    libsize_log_sd    : float   std of log(libsize)
    theta_median      : float   median NB dispersion (theta)
    theta_low_pct     : float   10th-pct theta (used for 'low' dispersion mode)
    """
    geom_mean_lib = np.exp(np.log(libsizes + 1).mean())
    norm_counts = counts / libsizes[None, :] * geom_mean_lib  # library-size normalised

    # Per-gene mean and variance (across samples)
    mu    = norm_counts.mean(axis=1).astype(np.float64)
    var   = norm_counts.var(axis=1, ddof=1).astype(np.float64)

    # NB: var = mu + mu^2/theta  =>  theta = mu^2 / (var - mu)
    safe  = var > mu * 1.01  # only genes with overdispersion
    theta = np.where(safe, mu**2 / np.maximum(var - mu, 1e-6), np.nan)
    theta = theta[~np.isnan(theta)]
    theta = np.clip(theta, 0.5, 200.0)

    # Baseline log mean (normalised count on log scale)
    log_mu = np.log(mu + 1e-6)
    baseline_log_mean = float(log_mu.mean())
    baseline_log_sd   = float(log_mu.std())

    libsize_log_sd = float(np.log(libsizes).std())
    theta_median   = float(np.median(theta))
    theta_lo       = float(np.percentile(theta, 10))

    print(f"\n[cal] NB parameter estimates:")
    print(f"  baseline_log_mean = {baseline_log_mean:.3f}  (current default: -2.0)")
    print(f"  baseline_log_sd   = {baseline_log_sd:.3f}   (current default: 0.6)")
    print(f"  libsize_log_sd    = {libsize_log_sd:.3f}    (current default: 0.5)")
    print(f"  theta_median      = {theta_median:.1f}      (medium=5, low=20)")
    print(f"  theta_10th_pct    = {theta_lo:.1f}", flush=True)

    return baseline_log_mean, baseline_log_sd, libsize_log_sd, theta_median, theta_lo


# --------------------------------------------------------------------------- #
# Step 5: Find signal_scale that produces target within-module r
# --------------------------------------------------------------------------- #
def _within_module_r(signal_scale: float, calibrated_cfg: dict,
                     n_features=300, n_modules=10, n_samp=100, seed=42,
                     n_pairs=500) -> float:
    """Run NB simulator and return mean within-module Pearson r."""
    from bsvae.simulation.scenario import _simulate_counts

    cfg = {**calibrated_cfg, "signal_scale": signal_scale}
    r = _simulate_counts(
        n_features=n_features,
        n_samples=n_samp,
        n_modules=n_modules,
        signal_scale=signal_scale,
        overlap_rate=0.0,
        confounding="none",
        nonlinear_mode="off",
        dropout_mode="off",
        dropout_target=0.0,
        seed=seed,
        generator_cfg=cfg,
    )
    X  = r.expr_features_x_samples.values.astype(np.float32)
    gt = r.modules_hard["module"].values.astype(int)

    rng  = np.random.default_rng(seed)
    within_r = []
    for k in np.unique(gt):
        idx = np.where(gt == k)[0]
        if len(idx) < 2:
            continue
        pairs = list(combinations(range(len(idx)), 2))
        if len(pairs) > n_pairs // len(np.unique(gt)):
            chosen = [pairs[i] for i in rng.choice(len(pairs),
                      size=n_pairs // len(np.unique(gt)), replace=False)]
        else:
            chosen = pairs
        for i, j in chosen:
            xi, xj = X[idx[i]], X[idx[j]]
            if xi.std() > 1e-6 and xj.std() > 1e-6:
                within_r.append(float(np.corrcoef(xi, xj)[0, 1]))

    return float(np.mean(within_r)) if within_r else 0.0


def find_signal_scale(calibrated_cfg: dict, target_r: float = 0.45,
                      lo: float = 0.5, hi: float = 8.0, n_iter: int = 10) -> float:
    """Binary search for signal_scale producing target within-module r."""
    print(f"\n[cal] Searching signal_scale for target r={target_r} ...", flush=True)

    # Quick scan first
    for ss in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
        r = _within_module_r(ss, calibrated_cfg)
        print(f"  signal_scale={ss:.1f}  within-r={r:.3f}", flush=True)
        if r >= target_r:
            hi = ss
            lo = max(lo, ss / 2)
            break

    # Binary search
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        r   = _within_module_r(mid, calibrated_cfg)
        print(f"  [bisect] signal_scale={mid:.3f}  r={r:.3f}", flush=True)
        if r < target_r:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 0.05:
            break

    best = (lo + hi) / 2.0
    final_r = _within_module_r(best, calibrated_cfg)
    print(f"\n[cal] signal_scale={best:.3f}  achieved r={final_r:.3f}  (target={target_r})", flush=True)
    return best


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tissue",     default=TISSUE)
    p.add_argument("--n-samples",  type=int, default=N_SAMPLES_CAL)
    p.add_argument("--min-cpm",    type=float, default=MIN_CPM)
    p.add_argument("--target-r",   type=float, default=0.45,
                   help="Target within-module Pearson r for signal_scale search")
    p.add_argument("--out",        default="scripts/gtex_calibrated_params.json")
    p.add_argument("--skip-signal-search", action="store_true",
                   help="Only estimate NB params, skip signal_scale search (faster)")
    return p.parse_args()


def main():
    args = parse_args()

    # 1. Sample IDs
    samples = get_tissue_samples(args.n_samples, tissue=args.tissue)

    # 2. Load counts
    counts_df = load_gct_subset(COUNTS_GCT, samples)

    # 3. QC + CPM
    counts_arr, libsizes, _ = compute_cpm_and_filter(
        counts_df, min_cpm=args.min_cpm
    )

    # 4. Fit NB
    blm, bls, lib_sd, theta_med, theta_lo = fit_nb_params(counts_arr, libsizes)

    # Map theta to dispersion mode:
    #   theta >= 15  → "low" (less overdispersion, more signal-preserving)
    #   5 <= theta < 15 → "medium"
    #   theta < 5   → "high"
    if theta_med >= 15:
        disp_mode = "low"
    elif theta_med >= 5:
        disp_mode = "medium"
    else:
        disp_mode = "high"
    print(f"  => dispersion mode: '{disp_mode}' (theta_median={theta_med:.1f})", flush=True)

    calibrated_cfg = {
        "family": "nb_latent_modules",
        "n_features": 300,
        "n_modules": 10,
        "module_size_distribution": "long_tail",
        "hub_fraction": 0.05,
        "hub_multiplier": 2.0,
        "baseline_log_mean": round(blm, 3),
        "baseline_log_sd":   round(bls, 3),
        "libsize_log_sd":    round(lib_sd, 3),
        "dispersion":        disp_mode,
        "signal_scale":      0.8,  # placeholder, updated below
        "overlap_rate":      0.0,
        "nonlinear_mode":    "off",
        "dropout_mode":      "off",
        "truth_edge_top_k":  25,
    }

    # 5. Find signal_scale
    if not args.skip_signal_search:
        sig_scale = find_signal_scale(calibrated_cfg, target_r=args.target_r)
        calibrated_cfg["signal_scale"] = round(float(sig_scale), 3)
    else:
        print("[cal] Skipping signal_scale search (--skip-signal-search)", flush=True)

    # 6. Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(calibrated_cfg, f, indent=2)
    print(f"\n[cal] Saved calibrated params → {out}", flush=True)
    print(json.dumps(calibrated_cfg, indent=2), flush=True)


if __name__ == "__main__":
    main()
