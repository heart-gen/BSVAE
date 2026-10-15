#!/usr/bin/env python
"""
Benchmark BSVAE against WGCNA on representative simulation scenarios.

Thin driver over ``bsvae.benchmark.harness.run_grid``; sizes are chosen so the
whole suite runs on a laptop in well under an hour.  For the manuscript-scale
grid use ``bsvae-benchmark run --config <grid.yaml>`` on an HPC node.

Scenarios
---------
sim1_balanced        NB counts, balanced module sizes
sim2_long_tail       NB counts, long-tail sizes with 10% null features   <- key case
sim3_nonlinear       long-tail + gene-specific nonlinear links
sim4_confound        long-tail + gene-specific batch confounding
sim5_dropout         long-tail + high dropout
sim6_isoform_hier    transcripts in genes, hard isoform switching (alpha=1)
sim7_isoform_mix     transcripts in genes, partial switching (alpha=0.5)

Usage
-----
python scripts/benchmark_sims.py --outdir results/bench_sims [--seeds 0,1,2]
    [--methods bsvae,wgcna] [--epochs 150] [--no-cuda] [--scenarios sim2_long_tail,sim6_isoform_hier]
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from bsvae.benchmark.harness import run_grid, summarize
from bsvae.simulation.scenario import DEFAULT_CONFIG, ISOFORM_GENERATOR_DEFAULTS

N_FEAT, N_SAMP, N_MOD = 1500, 150, 10
N_GENES = 600

NB = dict(DEFAULT_CONFIG["generator"], n_features=N_FEAT, n_samples=N_SAMP, n_modules=N_MOD,
          overlap_rate=0.0, confounding="none", nonlinear_mode="off", dropout_mode="off")
ISO = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=N_GENES, n_samples=N_SAMP, n_modules=N_MOD)

SCENARIOS = {
    "sim1_balanced": dict(NB, module_size_distribution="balanced"),
    "sim2_long_tail": dict(NB, module_size_distribution="long_tail", null_fraction=0.10),
    "sim3_nonlinear": dict(NB, module_size_distribution="long_tail", null_fraction=0.10, nonlinear_mode="on"),
    "sim4_confound": dict(NB, module_size_distribution="long_tail", null_fraction=0.10, confounding="moderate"),
    "sim5_dropout": dict(NB, module_size_distribution="long_tail", null_fraction=0.10,
                         dropout_mode="high", dropout_target=0.5),
    "sim6_isoform_hier": dict(ISO, switch_rate=0.15, alpha=1.0),
    "sim7_isoform_mix": dict(ISO, switch_rate=0.15, alpha=0.5),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/bench_sims")
    p.add_argument("--scenarios", default=",".join(SCENARIOS), help="Comma list of scenario names.")
    p.add_argument("--methods", default="bsvae,wgcna")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--reps", type=int, default=1, help="Data replicates per scenario.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--no-postprocess", action="store_true", default=False)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s: %(message)s")
    names = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        sys.exit(f"Unknown scenario(s): {unknown}; choose from {list(SCENARIOS)}")

    frames = []
    for name in names:
        cfg = {"base_seed": 13, "reps": args.reps, "generator": SCENARIOS[name], "grid": {},
               "stress": [], "outputs": {"emit_wgcna_matrix": False, "emit_gnvae_folds": False}}
        df = run_grid(
            cfg, f"{args.outdir}/{name}",
            methods=[m.strip() for m in args.methods.split(",")],
            seeds=[int(s) for s in args.seeds.split(",")],
            bsvae_cfg={"epochs": args.epochs, "no_cuda": args.no_cuda},
            postprocess=not args.no_postprocess,
        )
        df.insert(0, "scenario", name)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(f"{args.outdir}/results_all.tsv", sep="\t", index=False)
    summary = summarize(all_df, by=("scenario", "method"))
    summary.to_csv(f"{args.outdir}/summary_all.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    show = [c for c in ["scenario", "method", "n_runs", "ari_all_mean", "ari_assigned_mean",
                        "frac_pred_grey_mean", "K_ratio_mean", "gene_ari_mean",
                        "split_detection_rate_mean", "runtime_s_mean"] if c in summary.columns]
    print(summary[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
