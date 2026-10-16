"""
Network preservation and reproducibility benchmark.

On one transcript-level scenario (``nb_isoform_modules``, 600 genes × 150 samples,
10 modules, 15 % switched isoforms) each method is evaluated for

  seeds      random initialisations on the same data: pairwise assigned-feature
             ARI and per-feature stability (BSVAE only; WGCNA is deterministic),
  donors     donor resampling: bootstrap resamples of the samples (with
             replacement), ARI to the full-data solution,
  cohort     an independent cohort of the same truth (new samples, same
             structure): ARI between the two solutions, and preservation of the
             reference modules in the cohort data (Z_density, cor_kME),
  tissue     a cohort in which 30 % of the modules dissolved (their genes scattered
             over the other modules; ``--tissue-mode genes`` instead moves 30 % of
             genes at random): preservation statistics of reference modules against
             the fraction of their members that moved, and ARI on the unchanged genes,

and, for every comparison, reproducibility of isoform-specific membership:
among genes called divergent (isoforms in two or more modules) in the reference
solution, the fraction that is divergent again in the replicate, separately for
genes with a truly switched isoform and genes without one.

Usage:
  poetry run python scripts/benchmark_preservation.py --outdir results/preservation \\
    [--methods bsvae,bsvae_coherent,wgcna] [--seeds 0,1,2] [--bootstraps 5] [--epochs 150]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import run_bsvae, run_wgcna
from bsvae.networks.postprocess import enforce_gene_coherence
from bsvae.networks.quality import module_preservation
from bsvae.simulation.metrics import ari_nmi_assigned, compute_ari
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params

logger = logging.getLogger("preservation")

BASE = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=150, n_modules=10, switch_rate=0.15, alpha=1.0, null_fraction=0.10)
COHORT_SEED, TISSUE_SHIFT = 991, 0.3


def simulate(seed: int, tissue_mode: str = "modules"):
    a = simulate_params(BASE, seed)
    b = simulate_params(dict(BASE, sample_seed=COHORT_SEED), seed)
    c = simulate_params(dict(BASE, sample_seed=COHORT_SEED, tissue_shift=TISSUE_SHIFT, tissue_shift_mode=tissue_mode), seed)
    out = {}
    for name, r in (("A", a), ("B", b), ("C", c)):
        expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
        t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
        out[name] = dict(expr=expr, tx2gene=t2g, truth=r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values,
                         genes=r.extra_truth["modules_gene.csv"], switch=r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id"))
    return out


def fit(method: str, expr: pd.DataFrame, tx2gene: Dict[str, str], n_modules: int, seed: int, workdir: Path, epochs: int) -> np.ndarray:
    fids = [str(f) for f in expr.index]
    if method.startswith("bsvae"):
        cfg = {"epochs": epochs, "warmup_epochs": min(30, max(2, epochs // 3)), "transition_epochs": min(10, max(1, epochs // 10))}
        res = run_bsvae(expr, n_modules=n_modules, seed=seed, workdir=workdir, train_cfg=cfg)
        if res.status != "ok":
            raise RuntimeError(res.message)
        lab = res.labels.reindex(fids).fillna(-1).astype(int).values
        if method == "bsvae_coherent":
            lab, _ = enforce_gene_coherence(lab, expr, fids, tx2gene)
        return lab
    if method == "wgcna":
        res = run_wgcna(expr, workdir=workdir, seed=seed)
        if res.status != "ok":
            raise RuntimeError(res.message)
        return res.labels.reindex(fids).fillna(-1).astype(int).values
    raise ValueError(method)


def divergent_genes(labels: np.ndarray, fids: List[str], tx2gene: Dict[str, str]) -> set:
    df = pd.DataFrame({"gene": [tx2gene.get(f) for f in fids], "lab": labels})
    df = df[(df["lab"] >= 0) & df["gene"].notna()]
    n = df.groupby("gene")["lab"].nunique()
    return set(n.index[n > 1])


def divergence_reproducibility(lab_ref, lab_rep, fids, tx2gene, switch) -> Dict[str, float]:
    d_ref, d_rep = divergent_genes(lab_ref, fids, tx2gene), divergent_genes(lab_rep, fids, tx2gene)
    sw_genes = set(switch.loc[switch["is_switched"] == 1, "gene_id"])
    out = {}
    for name, genes in (("switched", d_ref & sw_genes), ("unswitched", d_ref - sw_genes)):
        out[f"div_repro_{name}"] = float(len(genes & d_rep) / len(genes)) if genes else np.nan
        out[f"n_div_ref_{name}"] = len(genes)
    return out


def stability(labels_list: List[np.ndarray]) -> float:
    """Per-feature fraction of run pairs that agree on co-membership with a random partner; here: mean pairwise ARI."""
    pairs = [compute_ari(a[(a >= 0) & (b >= 0)], b[(a >= 0) & (b >= 0)]) for a, b in itertools.combinations(labels_list, 2)]
    return float(np.mean(pairs)) if pairs else np.nan


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/preservation")
    p.add_argument("--methods", default="bsvae,bsvae_coherent,wgcna")
    p.add_argument("--seeds", default="0,1,2", help="Data/model seeds (each is one replicate).")
    p.add_argument("--bootstraps", type=int, default=5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--comparisons", default="seeds,donors,cohort,tissue")
    p.add_argument("--tissue-mode", choices=["modules", "genes"], default="modules")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows, pres_rows = [], []
    comps = set(args.comparisons.split(","))
    for seed in seeds:
        sims = simulate(seed, args.tissue_mode)
        A, B, C = sims["A"], sims["B"], sims["C"]
        fids = [str(f) for f in A["expr"].index]
        shifted = set(C["genes"].loc[C["genes"]["tissue_shifted"] == 1, "gene_id"].astype(str))
        unshifted_mask = np.array([A["tx2gene"][f] not in shifted for f in fids])
        for method in methods:
            wd = outdir / f"seed_{seed}" / method
            logger.info("seed %d | %s: reference fit", seed, method)
            ref = fit(method, A["expr"], A["tx2gene"], 10, seed, wd / "ref", args.epochs)
            truth_ari = ari_nmi_assigned(A["truth"], ref)["ari_assigned"]
            # (a) random initialisations
            if method.startswith("bsvae") and "seeds" in comps:
                inits = [ref] + [fit(method, A["expr"], A["tx2gene"], 10, 1000 * (seed + 1) + i, wd / f"init_{i}", args.epochs) for i in range(2)]
                row = dict(seed=seed, method=method, comparison="seeds", ari=stability(inits), truth_ari=truth_ari)
                reps = [divergence_reproducibility(ref, l, fids, A["tx2gene"], A["switch"]) for l in inits[1:]]
                row.update(pd.DataFrame(reps).mean().to_dict()); rows.append(row)
            # (b) donor resampling
            rng = np.random.default_rng(seed)
            boots = []
            for b in range(int(args.bootstraps) if "donors" in comps else 0):
                cols = rng.choice(A["expr"].shape[1], size=A["expr"].shape[1], replace=True)
                xb = A["expr"].iloc[:, cols]; xb.columns = [f"{c}_b{i}" for i, c in enumerate(xb.columns)]
                lab = fit(method, xb, A["tx2gene"], 10, seed, wd / f"boot_{b}", args.epochs)
                boots.append(lab)
                row = dict(seed=seed, method=method, comparison="donors", replicate=b, truth_ari=truth_ari,
                           ari=ari_nmi_assigned(ref, lab)["ari_assigned"])
                row.update(divergence_reproducibility(ref, lab, fids, A["tx2gene"], A["switch"])); rows.append(row)
            # (c) independent cohort
            if "cohort" in comps:
              labB = fit(method, B["expr"], B["tx2gene"], 10, seed, wd / "cohort", args.epochs)
              row = dict(seed=seed, method=method, comparison="cohort", truth_ari=truth_ari, ari=ari_nmi_assigned(ref, labB)["ari_assigned"],
                         truth_ari_cohort=ari_nmi_assigned(B["truth"], labB)["ari_assigned"])
              row.update(divergence_reproducibility(ref, labB, fids, A["tx2gene"], A["switch"])); rows.append(row)
              pres = module_preservation(A["expr"], B["expr"], ref, fids, n_perm=args.n_perm, seed=seed)
              pres.insert(0, "comparison", "cohort"); pres.insert(0, "method", method); pres.insert(0, "seed", seed)
              pres["frac_shifted"] = 0.0; pres_rows.append(pres)
            # (d) tissue with partial change
            if "tissue" in comps:
              labC = fit(method, C["expr"], C["tx2gene"], 10, seed, wd / "tissue", args.epochs)
              m = unshifted_mask
              row = dict(seed=seed, method=method, comparison="tissue", truth_ari=truth_ari,
                         ari=ari_nmi_assigned(ref, labC)["ari_assigned"], ari_unshifted=ari_nmi_assigned(ref[m], labC[m])["ari_assigned"],
                         truth_ari_cohort=ari_nmi_assigned(C["truth"], labC)["ari_assigned"])
              row.update(divergence_reproducibility(ref, labC, fids, A["tx2gene"], A["switch"])); rows.append(row)
              pres = module_preservation(A["expr"], C["expr"], ref, fids, n_perm=args.n_perm, seed=seed)
              frac = {}
              for k in pres["module"]:
                  members = [f for f, l in zip(fids, ref) if l == k]
                  frac[k] = float(np.mean([A["tx2gene"][f] in shifted for f in members])) if members else np.nan
              pres.insert(0, "comparison", "tissue"); pres.insert(0, "method", method); pres.insert(0, "seed", seed)
              pres["frac_shifted"] = pres["module"].map(frac); pres_rows.append(pres)
            pd.DataFrame(rows).to_csv(outdir / "results.tsv", sep="\t", index=False)
            pd.concat(pres_rows, ignore_index=True).to_csv(outdir / "preservation.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows); pres = pd.concat(pres_rows, ignore_index=True) if pres_rows else pd.DataFrame(columns=["comparison", "method", "frac_shifted", "Z_density", "cor_kME", "n_shared"])
    pd.set_option("display.width", 250)
    cols = [c for c in ["ari", "ari_unshifted", "truth_ari", "truth_ari_cohort", "div_repro_switched", "div_repro_unswitched", "n_div_ref_switched", "n_div_ref_unswitched"] if c in df.columns]
    summ = df.groupby(["comparison", "method"])[cols].mean()
    summ.to_csv(outdir / "summary.tsv", sep="\t")
    print(summ.round(3).to_string())
    print()
    pres["shift_bin"] = pd.cut(pres["frac_shifted"].fillna(0), [-0.01, 0.1, 0.5, 1.0], labels=["<=10%", "10-50%", ">50%"])
    print(pres.groupby(["comparison", "method", "shift_bin"], observed=True)[["Z_density", "cor_kME", "n_shared"]].agg(["mean", "count"]).round(2).to_string())


if __name__ == "__main__":
    main()
