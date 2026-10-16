"""
kME preservation across module resolutions, stratified and matched by module size.

Module preservation statistics scale with module size (Z_density grows with
members; kME ordering is more stable in large modules), so methods that
produce modules of different sizes cannot be compared on raw cor_kME.  This
benchmark fits each method at several resolutions on cohort A of the
reference isoform scenario, scores every module's preservation in an
independent cohort B of the same truth (``module_preservation``), and reports

  * cor_kME and Z_density by module-size bin and resolution,
  * a size-adjusted cor_kME per method and resolution (linear fit on log size,
    evaluated at 50 and 150 members),
  * a size-matched paired comparison: each BSVAE module is matched to the
    WGCNA module (same seed, default resolutions) closest in size, and the
    paired difference in cor_kME is averaged,
  * the same restricted to "true" modules (≥ 70 % of members from one true
    module), to separate real modules from over-clustering remnants.

Gene coherence (the default annotation policy) is applied to every label set.

Usage:
  poetry run python scripts/benchmark_kme_preservation.py --outdir results/kme_preservation \\
    [--seeds 0,1,2] [--epochs 150] [--n-perm 200]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import run_bsvae, run_wgcna
from bsvae.networks.postprocess import enforce_gene_coherence
from bsvae.networks.quality import module_preservation
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params

logger = logging.getLogger("kme_preservation")
BASE = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=150, n_modules=10, switch_rate=0.15, alpha=1.0, null_fraction=0.10)
COHORT_SEED = 991

RESOLUTIONS = {
    "bsvae": [
        ("ocf1_cut0.40", dict(over_cluster_factor=1.0, merge_cut_height=0.40)),
        ("ocf3_cut0.25", dict(over_cluster_factor=3.0, merge_cut_height=0.25)),
        ("ocf3_cut0.40", dict(over_cluster_factor=3.0, merge_cut_height=0.40)),   # default
        ("ocf3_cut0.60", dict(over_cluster_factor=3.0, merge_cut_height=0.60)),
    ],
    "wgcna": [
        ("ds2_cut0.15", dict(deep_split=2, merge_cut_height=0.15)),
        ("ds2_cut0.25", dict(deep_split=2, merge_cut_height=0.25)),             # default
        ("ds2_cut0.40", dict(deep_split=2, merge_cut_height=0.40)),
        ("ds4_cut0.25", dict(deep_split=4, merge_cut_height=0.25)),
    ],
}
DEFAULT_RES = {"bsvae": "ocf3_cut0.40", "wgcna": "ds2_cut0.25"}
SIZE_BINS = [0, 50, 150, 10 ** 9]
SIZE_LABELS = ["<50", "50-150", ">150"]


def fit(method: str, name: str, cfg: Dict, expr: pd.DataFrame, tx2gene: Dict[str, str], seed: int, workdir: Path, epochs: int) -> np.ndarray:
    fids = [str(f) for f in expr.index]
    if method == "bsvae":
        tcfg = {"epochs": epochs, "warmup_epochs": min(30, max(2, epochs // 3)), "transition_epochs": min(10, max(1, epochs // 10))}
        res = run_bsvae(expr, n_modules=10, seed=seed, workdir=workdir, train_cfg=tcfg,
                        post_cfg={"merge_cut_height": cfg["merge_cut_height"]}, over_cluster_factor=cfg["over_cluster_factor"])
    else:
        res = run_wgcna(expr, workdir=workdir, seed=seed, deep_split=cfg["deep_split"], merge_cut_height=cfg["merge_cut_height"])
    if res.status != "ok":
        raise RuntimeError(f"{method} {name}: {res.message}")
    lab = res.labels.reindex(fids).fillna(-1).astype(int).values
    lab, _ = enforce_gene_coherence(lab, expr, fids, tx2gene)
    return lab


def purity(labels: np.ndarray, truth: np.ndarray) -> Dict[int, float]:
    out = {}
    for m in np.unique(labels[labels >= 0]):
        t = truth[labels == m]
        t = t[t >= 0]
        out[int(m)] = float(pd.Series(t).value_counts(normalize=True).iloc[0]) if len(t) else np.nan
    return out


def size_adjusted(df: pd.DataFrame, at: float) -> float:
    d = df.dropna(subset=["cor_kME"])
    if len(d) < 4 or d["n_shared"].nunique() < 2:
        return np.nan
    x = np.log(d["n_shared"].values.astype(float)); y = d["cor_kME"].values
    b, a = np.polyfit(x, y, 1)
    return float(a + b * np.log(at))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/kme_preservation")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows: List[pd.DataFrame] = []
    for seed in seeds:
        a = simulate_params(BASE, seed); b = simulate_params(dict(BASE, sample_seed=COHORT_SEED), seed)
        ea = a.expr_features_x_samples; ea.index = ea.index.astype(str)
        eb = b.expr_features_x_samples; eb.index = eb.index.astype(str)
        t2g = dict(zip(a.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), a.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
        truth = a.modules_hard.set_index("feature_id")["module"].reindex(ea.index).values
        fids = list(ea.index)
        for method, configs in RESOLUTIONS.items():
            for name, cfg in configs:
                logger.info("seed %d | %s %s", seed, method, name)
                lab = fit(method, name, cfg, ea, t2g, seed, outdir / f"seed_{seed}" / method / name, args.epochs)
                pres = module_preservation(ea, eb, lab, fids, n_perm=args.n_perm, seed=seed)
                pur = purity(lab, truth)
                pres["purity"] = pres["module"].map(pur)
                pres.insert(0, "resolution", name); pres.insert(0, "method", method); pres.insert(0, "seed", seed)
                pres["n_modules"] = int(len(pres)); pres["grey_frac"] = float((lab < 0).mean())
                rows.append(pres)
                pd.concat(rows, ignore_index=True).to_csv(outdir / "preservation_by_module.tsv", sep="\t", index=False)
    df = pd.concat(rows, ignore_index=True)
    df["size_bin"] = pd.cut(df["n_shared"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    df["true_module"] = df["purity"] >= 0.7
    pd.set_option("display.width", 250)
    print("== cor_kME / Z_density by method, resolution and size bin (all modules)")
    print(df.groupby(["method", "resolution", "size_bin"], observed=True)[["cor_kME", "Z_density"]].agg(["mean", "count"]).round(2).to_string())
    print("\n== same, true modules only (purity >= 0.7)")
    print(df[df["true_module"]].groupby(["method", "resolution", "size_bin"], observed=True)[["cor_kME", "Z_density"]].agg(["mean", "count"]).round(2).to_string())
    print("\n== size-adjusted cor_kME (log-size fit) at 50 and 150 members, modules per run, grey fraction")
    adj = []
    for (m, r), g in df.groupby(["method", "resolution"]):
        adj.append({"method": m, "resolution": r, "n_modules": g.groupby("seed")["module"].nunique().mean(),
                    "grey_frac": g["grey_frac"].mean(), "median_size": g["n_shared"].median(),
                    "cor_kME_at50": size_adjusted(g, 50), "cor_kME_at150": size_adjusted(g, 150),
                    "cor_kME_true_at100": size_adjusted(g[g["true_module"]], 100), "frac_true_modules": g["true_module"].mean()})
    adj = pd.DataFrame(adj); adj.to_csv(outdir / "size_adjusted.tsv", sep="\t", index=False)
    print(adj.round(3).to_string(index=False))
    print("\n== size-matched paired comparison at default resolutions (BSVAE module vs closest-size WGCNA module, same seed)")
    pairs = []
    for seed in seeds:
        bs = df[(df["method"] == "bsvae") & (df["resolution"] == DEFAULT_RES["bsvae"]) & (df["seed"] == seed)]
        wg = df[(df["method"] == "wgcna") & (df["resolution"] == DEFAULT_RES["wgcna"]) & (df["seed"] == seed)]
        for _, r in bs.iterrows():
            if wg.empty:
                continue
            j = (wg["n_shared"] - r["n_shared"]).abs().idxmin()
            pairs.append({"seed": seed, "bsvae_size": r["n_shared"], "wgcna_size": wg.loc[j, "n_shared"],
                          "d_cor_kME": r["cor_kME"] - wg.loc[j, "cor_kME"], "d_Z": r["Z_density"] - wg.loc[j, "Z_density"],
                          "bsvae_true": r["true_module"], "size_ratio": r["n_shared"] / max(wg.loc[j, "n_shared"], 1)})
    pairs = pd.DataFrame(pairs); pairs.to_csv(outdir / "size_matched_pairs.tsv", sep="\t", index=False)
    close = pairs[(pairs["size_ratio"] > 0.5) & (pairs["size_ratio"] < 2.0)]
    print(f"all pairs: n={len(pairs)} mean d_cor_kME={pairs['d_cor_kME'].mean():.3f} mean d_Z={pairs['d_Z'].mean():.2f}")
    print(f"pairs within 2x size: n={len(close)} mean d_cor_kME={close['d_cor_kME'].mean():.3f} mean d_Z={close['d_Z'].mean():.2f}")
    tp = close[close["bsvae_true"]]
    print(f"within 2x size, true BSVAE modules: n={len(tp)} mean d_cor_kME={tp['d_cor_kME'].mean():.3f} mean d_Z={tp['d_Z'].mean():.2f}")


if __name__ == "__main__":
    main()
