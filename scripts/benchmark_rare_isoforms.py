"""
Rare-isoform benchmark: how well switched isoforms are recovered as a function of their
abundance, what the noise ceiling is, and which single-matrix remedies help.

Scenarios (transcript-level, 600 genes × --n-samples): hard and partial switching, each
at the default minor-isoform share (dominant fraction 0.6) and at a rare share (0.9, minor
isoforms ~8 % of their gene).  Variants of BSVAE:

  base       log-normalised input (default pipeline, raw labels for the label metrics)
  masked     --masked-recon (zeros excluded from the reconstruction loss)
  filter     features below --filter-min-mean log2 expression dropped before training
  pearson    negative-binomial Pearson residuals of the raw counts as input (theta 100)

plus WGCNA (signed) as reference.  For every switched isoform the script records whether
its label is its true module, whether it is placed apart from its dominant sibling, whether
the residual scan calls it, and its abundance tercile; it also reports the oracle ceiling:
the correlation of each switched isoform's sibling residual with the true factor of its
target module, and how often that oracle ranks the target module first.

Usage:
  poetry run python scripts/benchmark_rare_isoforms.py --outdir results/rare \\
    [--seeds 0,1,2] [--n-samples 150] [--epochs 150] [--variants base,masked,filter,pearson,wgcna]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import run_bsvae, run_wgcna
from bsvae.simulation.metrics import evaluate_labels, normalize_labels
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import isoform_residuals, isoform_specific_calls
from bsvae.utils.preprocess import pearson_residuals

logger = logging.getLogger("rare")


def scenarios(n_samples: int) -> Dict[str, Dict]:
    iso = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=n_samples, n_modules=10, switch_rate=0.15)
    return {
        "hard": dict(iso, alpha=1.0, dominant_isoform_frac=0.6),
        "hard_rare": dict(iso, alpha=1.0, dominant_isoform_frac=0.9),
        "partial": dict(iso, alpha=0.5, dominant_isoform_frac=0.6),
        "partial_rare": dict(iso, alpha=0.5, dominant_isoform_frac=0.9),
    }


def majority_map(pred: np.ndarray, true: np.ndarray) -> Dict[int, int]:
    ct = pd.crosstab(pred[pred >= 0], true[pred >= 0])
    return ct.idxmax(axis=1).to_dict() if len(ct) else {}


def per_isoform(labels: pd.Series, truth: pd.Series, st: pd.DataFrame, expr: pd.DataFrame, calls: pd.DataFrame | None) -> pd.DataFrame:
    fids = truth.index
    t = truth.values; p = labels.reindex(fids).fillna(-1).astype(int).values
    m = majority_map(p, t); mapped = np.array([m.get(x, -1) if x >= 0 else -1 for x in p])
    s = st.reindex(fids)
    dom = s[s["isoform_rank"] == 0]
    dom_lab = dict(zip(dom["gene_id"], p[np.isin(fids, dom.index)]))
    dom_p = np.array([dom_lab.get(g, -1) for g in s["gene_id"].values])
    out = pd.DataFrame({
        "gene_id": s["gene_id"].values, "is_switched": s["is_switched"].values == 1,
        "minor": (s["isoform_rank"].values > 0) & (s["is_null"].values == 0),
        "mean_expr": expr.reindex(fids).mean(axis=1).values,
        "correct": (mapped == t) & (p >= 0), "separated": (p >= 0) & (dom_p >= 0) & (p != dom_p), "grey": p < 0,
    }, index=fids)
    if calls is not None:
        out["called"] = calls["called"].reindex(fids).fillna(False).values
    return out


def summarise(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keys = ["scenario", "variant", "stratum"]
    cols = [c for c in df.columns if c not in keys + ["seed"]]
    return df.groupby(keys)[cols].mean()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/rare")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--variants", default="base,masked,filter,pearson,wgcna")
    p.add_argument("--filter-min-mean", type=float, default=1.0)
    p.add_argument("--scenarios", default="hard,hard_rare,partial,partial_rare")
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]
    tcfg = {"epochs": args.epochs, "warmup_epochs": min(30, max(2, args.epochs // 3)), "transition_epochs": min(10, max(1, args.epochs // 10))}
    rows: List[Dict] = []; ceiling: List[Dict] = []
    for sname, gen in scenarios(args.n_samples).items():
        if sname not in args.scenarios.split(","):
            continue
        for seed in seeds:
            r = simulate_params(gen, seed)
            expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
            counts = r.counts_features_x_samples; counts.index = counts.index.astype(str)
            t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
            truth = r.modules_hard.set_index("feature_id")["module"]; truth.index = truth.index.astype(str)
            st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id"); st.index = st.index.astype(str)
            Z = r.module_latents.reindex(expr.columns); Zc = (Z - Z.mean()) / Z.std(ddof=0)
            # oracle ceiling from the sibling residual against the true factors
            resid, _ = isoform_residuals(expr, t2g)
            sw_ids = [t for t in resid.index if st.loc[t, "is_switched"] == 1]
            R = resid.loc[sw_ids]; Rc = R.sub(R.mean(axis=1), axis=0).div(R.std(axis=1, ddof=0) + 1e-12, axis=0)
            cor = Rc.values @ Zc.values / Zc.shape[0]
            tgt = st.loc[sw_ids, "tx_module"].values.astype(int)
            me_sw = expr.reindex(sw_ids).mean(axis=1)
            q = pd.qcut(me_sw.rank(method="first"), 3, labels=["low", "mid", "high"]).values
            for b in ["low", "mid", "high"]:
                m = q == b
                ceiling.append(dict(scenario=sname, seed=seed, stratum=b, n=int(m.sum()), mean_expr=float(me_sw.values[m].mean()),
                                    oracle_abs_cor=float(np.abs(cor[m, tgt[m]]).mean()),
                                    oracle_rank1=float((np.abs(cor[m]).argmax(axis=1) == tgt[m]).mean())))
            for variant in variants:
                wd = outdir / sname / f"seed_{seed}" / variant
                if variant == "wgcna":
                    res = run_wgcna(expr, workdir=wd, seed=seed)
                    if res.status != "ok":
                        logger.warning("WGCNA failed: %s", res.message); continue
                    table = per_isoform(res.labels, truth, st, expr, None)
                else:
                    X, cfg = expr, dict(tcfg)
                    if variant == "masked":
                        cfg["masked_recon"] = True
                    elif variant == "filter":
                        X = expr.loc[expr.mean(axis=1) >= args.filter_min_mean]
                    elif variant == "pearson":
                        X = pearson_residuals(counts, theta=100.0)
                    res = run_bsvae(X, n_modules=10, seed=seed, workdir=wd, train_cfg=cfg, device=dev)
                    if res.status != "ok":
                        raise RuntimeError(res.message)
                    lab = res.labels.reindex(X.index).fillna(-1).astype(int).values
                    calls = isoform_specific_calls(X if variant != "pearson" else expr, t2g, lab, list(X.index))
                    table = per_isoform(res.labels, truth, st, expr, calls)
                ari = evaluate_labels(truth.values, normalize_labels(res.labels.reindex(truth.index).fillna(-1).astype(int).values))["ari_assigned"]
                sw = table[table["is_switched"]]
                qq = pd.qcut(sw["mean_expr"].rank(method="first"), 3, labels=["low", "mid", "high"])
                for stratum, sub in [("all", sw)] + [(b, sw[qq == b]) for b in ["low", "mid", "high"]]:
                    row = dict(scenario=sname, seed=seed, variant=variant, stratum=stratum, n=len(sub), ari_assigned=ari,
                               mean_expr=float(sub["mean_expr"].mean()), correct=float(sub["correct"].mean()),
                               separated=float(sub["separated"].mean()), grey=float(sub["grey"].mean()))
                    if "called" in sub:
                        row["called"] = float(sub["called"].mean())
                    rows.append(row)
                ns = table[table["minor"] & ~table["is_switched"]]
                row = dict(scenario=sname, seed=seed, variant=variant, stratum="unswitched_minor", n=len(ns), ari_assigned=ari,
                           mean_expr=float(ns["mean_expr"].mean()), correct=np.nan, separated=float(ns["separated"].mean()), grey=float(ns["grey"].mean()))
                if "called" in ns:
                    row["called"] = float(ns["called"].mean())
                rows.append(row)
                logger.info("%s seed %d %s: ari %.3f switched correct %.2f called %s", sname, seed, variant, ari,
                            rows[-2]["correct"] if rows[-2]["stratum"] == "high" else float("nan"), rows[-1].get("called"))
                pd.DataFrame(rows).to_csv(outdir / "results.tsv", sep="\t", index=False)
                pd.DataFrame(ceiling).to_csv(outdir / "oracle_ceiling.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print("== oracle ceiling (sibling residual vs true target factor)")
    print(pd.DataFrame(ceiling).groupby(["scenario", "stratum"])[["n", "mean_expr", "oracle_abs_cor", "oracle_rank1"]].mean().round(3).to_string())
    print("\n== recovery by variant and abundance stratum")
    print(summarise(rows).round(3).to_string())


if __name__ == "__main__":
    main()
