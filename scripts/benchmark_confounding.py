"""
Abundance-confounding benchmark: when does a method recover transcript-specific
regulation rather than variation in total gene expression?

Scenario sets (transcript-level ``nb_isoform_modules`` family, 600 genes × 150 samples):

  A  abundance_only   no switching, no usage regulation; gene-level signal 1.0 / 1.9 / 3.0.
                      Any gene whose isoforms land in different modules is a false split.
  B  usage_null       usage-regulated isoforms in genes with NO gene-level signal: the
                      gene total is flat, only the isoform split varies (pure
                      transcript-specific regulation).
  C  usage_ratio      usage regulation in signal genes at strength 0.5 / 1.0 / 2.0
                      against gene-level signal 1.9: recovery vs the ratio of
                      isoform-specific to gene-level variation.
  D  switch_vs_usage  the same isoforms regulated by an uncompensated abundance
                      switch (gene total changes) or a compensated usage shift (gene
                      total unchanged): does a method need the total to move?

Usage:
  poetry run python scripts/benchmark_confounding.py --outdir results/confounding \\
    [--methods bsvae,bsvae_coherent,wgcna,bsvae_isoform] [--seeds 0,1,2] [--epochs 150] [--sets A,B,C,D]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bsvae.benchmark.harness import load_truth, run_grid, summarize
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS

ISO = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=150, n_modules=10, switch_rate=0.0, usage_rate=0.0)

SETS = {
    "A_abundance_only": {"generator": dict(ISO, null_fraction=0.10), "grid": {"signal_scale": [1.0, 1.9, 3.0]}},
    "B_usage_null": {"generator": dict(ISO, null_fraction=0.35, usage_rate=0.6, usage_from="null", usage_strength=1.0),
                     "grid": {"usage_strength": [0.5, 1.0, 2.0]}},
    "C_usage_ratio": {"generator": dict(ISO, null_fraction=0.10, usage_rate=0.15, usage_from="signal", signal_scale=1.9),
                      "grid": {"usage_strength": [0.5, 1.0, 2.0]}},
    "D_switch_vs_usage": {"generator": dict(ISO, null_fraction=0.10, signal_scale=1.9),
                          "grid": {"mechanism": ["switch", "usage"]}},
}


def _expand_D(cfg):
    """The D set is two configs rather than a grid value the generator knows."""
    base = dict(cfg["generator"])
    return {
        "switch": dict(base, switch_rate=0.15, alpha=1.0),
        "usage": dict(base, usage_rate=0.15, usage_from="signal", usage_strength=1.0),
    }


def stratified_false_splits(outdir: Path, scenario: str) -> pd.DataFrame:
    """Per-run false-split and divergence rates by isoform abundance tercile (from saved labels)."""
    rows = []
    for run_dir in sorted(outdir.glob("scenarios/*/rep_*")):
        truth = load_truth(run_dir)
        if "switch_table" not in truth:
            continue
        expr = pd.read_csv(truth["expr_path"], sep="\t", index_col=0); expr.index = expr.index.astype(str)
        st = truth["switch_table"].set_index("transcript_id")
        me = expr.mean(axis=1)
        for mpath in sorted(run_dir.glob("methods/*/seed_*/modules.csv")):
            method = mpath.parts[-3]
            if method == "bsvae_isoform":
                continue
            pred = pd.read_csv(mpath).set_index("feature_id")["module"].reindex(expr.index).fillna(-1).astype(int)
            multi = st[(st["n_isoforms"] >= 2) & (st["is_null"] == 0)] if "n_isoforms" in st else st[st["is_null"] == 0]
            minor = multi[multi["isoform_rank"] > 0]
            q = pd.qcut(me.loc[minor.index].rank(method="first"), 3, labels=["low", "mid", "high"])
            for b in ["low", "mid", "high"]:
                idx = minor.index[(q == b).values]
                dom = st.loc[st.index.isin(st.loc[idx, "gene_id"].map(lambda g: st.index[(st["gene_id"] == g) & (st["isoform_rank"] == 0)][0]))]
                dom_lab = {g: int(pred[t]) for g, t in zip(dom["gene_id"], dom.index)}
                p = pred.loc[idx].values
                d = np.array([dom_lab.get(g, -1) for g in st.loc[idx, "gene_id"]])
                sw = st.loc[idx, "is_switched"].values == 1
                sep = (p >= 0) & (d >= 0) & (p != d)
                rows.append({"scenario": scenario, "scenario_id": run_dir.parts[-2], "method": method, "seed": mpath.parts[-2],
                             "tercile": b, "mean_expr": float(me.loc[idx].mean()),
                             "false_divergence": float(sep[~sw].mean()) if (~sw).any() else np.nan,
                             "true_divergence": float(sep[sw].mean()) if sw.any() else np.nan,
                             "grey": float((p < 0).mean())})
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/confounding")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--methods", default="bsvae,bsvae_coherent,wgcna,bsvae_isoform")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    names = [s.strip() for s in args.sets.split(",") if s.strip()]
    unknown = [n for n in names if n not in SETS]
    if unknown:
        sys.exit(f"Unknown set(s): {unknown}; choose from {list(SETS)}")
    methods = [m.strip() for m in args.methods.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    frames, strat = [], []
    for name in names:
        cfg = SETS[name]
        configs = _expand_D(cfg) if name.startswith("D_") else {"": cfg}
        for sub, c in configs.items():
            gen = c if name.startswith("D_") else c["generator"]
            grid = {} if name.startswith("D_") else c["grid"]
            conf = {"base_seed": 13, "reps": args.reps, "generator": gen, "grid": grid, "stress": [],
                    "outputs": {"emit_wgcna_matrix": False, "emit_gnvae_folds": False}}
            out = Path(args.outdir) / (f"{name}_{sub}" if sub else name)
            df = run_grid(conf, out, methods=methods, seeds=seeds,
                          bsvae_cfg={"epochs": args.epochs, "no_cuda": args.no_cuda})
            label = f"{name}:{sub}" if sub else name
            df.insert(0, "scenario", label)
            frames.append(df)
            strat.append(stratified_false_splits(out, label))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(f"{args.outdir}/results_all.tsv", sep="\t", index=False)
    factor_cols = [c for c in ["p_signal_scale", "p_usage_strength"] if c in all_df.columns]
    summary = summarize(all_df, by=("scenario", *factor_cols, "method"))
    summary.to_csv(f"{args.outdir}/summary_all.tsv", sep="\t", index=False)
    strat_df = pd.concat(strat, ignore_index=True)
    strat_df.to_csv(f"{args.outdir}/false_splits_by_abundance.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    show = [c for c in ["scenario", *factor_cols, "method", "ari_assigned_mean", "gene_ari_mean", "false_split_rate_mean",
                        "split_detection_rate_mean", "switched_recall_mean", "switched_precision_mean"] if c in summary.columns]
    print(summary[show].round(3).to_string(index=False))
    print()
    print(strat_df.groupby(["scenario", "method", "tercile"], observed=True)[["mean_expr", "false_divergence", "true_divergence", "grey"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
