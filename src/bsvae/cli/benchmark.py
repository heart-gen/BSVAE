"""
bsvae-benchmark — run BSVAE and WGCNA head-to-head on a simulation grid.

Usage
-----
bsvae-benchmark run --config sim.yaml --outdir results/bench --methods bsvae,wgcna
bsvae-benchmark summarize --results results/bench/results.tsv
bsvae-benchmark check-env
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from bsvae.cli._common import ast_eval, setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bsvae-benchmark", description=__doc__.split("Usage")[0].strip())
    subs = p.add_subparsers(dest="command", required=True)

    r = subs.add_parser("run", help="Generate scenarios and run the methods.")
    r.add_argument("--config", required=True, help="Scenario config (JSON or YAML).")
    r.add_argument("--outdir", required=True)
    r.add_argument("--methods", default="bsvae,wgcna", help="Comma list (default: bsvae,wgcna).")
    r.add_argument("--reps", type=int, default=None, help="Replicates per scenario (default: config reps).")
    r.add_argument("--rep-ids", default=None,
                   help="Comma list of replicate indices to run (e.g. one per SLURM task); overrides --reps.")
    r.add_argument("--seeds", default="0", help="Comma list of method seeds (default: 0).")
    r.add_argument("--scenario-ids", default=None, help="Comma list to restrict scenarios.")
    r.add_argument("--k-mode", choices=["oracle", "sweep"], default="oracle")
    r.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    r.add_argument("--no-cuda", action="store_true", default=False)
    r.add_argument("--log-level", default="info")
    # BSVAE overrides
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--warmup-epochs", type=int, default=None)
    r.add_argument("--transition-epochs", type=int, default=None)
    r.add_argument("--latent-dim", type=int, default=None)
    r.add_argument("--hidden-dims", type=ast_eval, default=None)
    r.add_argument("--beta", type=float, default=None)
    r.add_argument("--latent-corr-strength", type=float, default=None)
    r.add_argument("--hier-strength", type=float, default=None)
    r.add_argument("--isoform-stratified", action="store_true", default=None)
    r.add_argument("--bal-type", choices=["uniform", "entropy", "none"], default=None)
    r.add_argument("--no-normalize-input", dest="normalize_input", action="store_false", default=None)
    # Post-processing
    r.add_argument("--no-postprocess", action="store_true", default=False)
    r.add_argument("--consensus-runs", type=int, default=3,
                   help="Runs per seed for the bsvae_consensus method (default: 3).")
    r.add_argument("--over-cluster-factor", type=float, default=None,
                   help="Train the GMM with this multiple of the expected module count and let "
                        "eigengene merging consolidate (default: 3; 1 disables over-clustering).")
    r.add_argument("--min-gamma", type=float, default=None)
    r.add_argument("--min-module-size", type=int, default=None)
    r.add_argument("--merge-cut-height", type=float, default=None)
    r.add_argument("--kme-reassign", action="store_true", default=None)
    # WGCNA
    r.add_argument("--wgcna-network-type", choices=["signed", "unsigned", "signed hybrid"], default=None)
    r.add_argument("--wgcna-soft-power", type=int, default=None)
    r.add_argument("--wgcna-min-module-size", type=int, default=None)
    r.add_argument("--wgcna-merge-cut-height", type=float, default=None)
    r.add_argument("--wgcna-threads", type=int, default=None)

    s = subs.add_parser("summarize", help="Mean ± sd per scenario and method.")
    s.add_argument("--results", required=True, help="results.tsv written by `run`.")
    s.add_argument("--by", default="scenario_id,method")
    s.add_argument("--output", default=None, help="Write the summary TSV here (default: stdout).")

    subs.add_parser("check-env", help="Report Rscript/WGCNA/torch availability.")
    return p


def _opt(args, names):
    return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}


def handle_run(args, logger):
    from bsvae.benchmark.harness import run_grid

    bsvae_cfg = _opt(args, ["epochs", "warmup_epochs", "transition_epochs", "latent_dim", "hidden_dims",
                            "beta", "latent_corr_strength", "hier_strength", "isoform_stratified",
                            "bal_type", "normalize_input"])
    if args.no_cuda:
        bsvae_cfg["no_cuda"] = True
    post_cfg = _opt(args, ["min_gamma", "min_module_size", "merge_cut_height", "kme_reassign"])
    wgcna_cfg = {}
    for cli_name, key in [("wgcna_network_type", "network_type"), ("wgcna_soft_power", "soft_power"),
                          ("wgcna_min_module_size", "min_module_size"),
                          ("wgcna_merge_cut_height", "merge_cut_height"), ("wgcna_threads", "n_threads")]:
        v = getattr(args, cli_name, None)
        if v is not None:
            wgcna_cfg[key] = v
    df = run_grid(
        args.config, args.outdir,
        methods=[m.strip() for m in args.methods.split(",") if m.strip()],
        reps=args.reps,
        rep_ids=[int(x) for x in args.rep_ids.split(",")] if args.rep_ids else None,
        seeds=[int(s) for s in args.seeds.split(",") if s.strip()],
        k_mode=args.k_mode,
        bsvae_cfg=bsvae_cfg, wgcna_cfg=wgcna_cfg, post_cfg=post_cfg,
        postprocess=not args.no_postprocess, resume=args.resume,
        scenario_ids=[s.strip() for s in args.scenario_ids.split(",")] if args.scenario_ids else None,
        over_cluster_factor=args.over_cluster_factor, consensus_runs=args.consensus_runs,
    )
    logger.info("Benchmark complete: %d rows → %s/results.tsv", len(df), args.outdir)


def handle_summarize(args, logger):
    import pandas as pd
    from bsvae.benchmark.harness import summarize

    df = pd.read_csv(args.results, sep="\t")
    out = summarize(df, by=[b.strip() for b in args.by.split(",")])
    if args.output:
        out.to_csv(args.output, sep="\t", index=False)
        logger.info("Summary written to %s", args.output)
    else:
        pd.set_option("display.width", 200)
        print(out.to_string(index=False))


def handle_check_env(args, logger):
    import shutil
    import torch
    from bsvae.benchmark.runners import wgcna_available

    info = {
        "Rscript": shutil.which("Rscript"),
        "WGCNA": wgcna_available(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
    }
    print(json.dumps(info, indent=2))


def cli(argv=None):
    args = build_parser().parse_args(argv)
    logger = setup_logging(getattr(args, "log_level", "info"), name=__name__)
    if args.command == "run":
        handle_run(args, logger)
    elif args.command == "summarize":
        handle_summarize(args, logger)
    elif args.command == "check-env":
        handle_check_env(args, logger)


if __name__ == "__main__":
    cli()
