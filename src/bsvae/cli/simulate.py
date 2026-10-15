"""
bsvae-simulate — Generate synthetic omics data and benchmark module methods.

Subcommands
-----------
  generate           Simulate omics data with known block-diagonal module structure.
  benchmark          ARI/NMI/enrichment vs WGCNA/GNVAE on simulated data.
  init-config        Write starter simulation grid config.
  generate-grid      Generate all scenarios/replicates defined in config.
  generate-scenario  Generate one scenario/replicate.
  validate-grid      Validate generated grid output structure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsvae-simulate",
        description="Simulate omics data and benchmark module discovery methods.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    gp = subs.add_parser("generate", help="Simulate omics data with known module structure.")
    gp.add_argument("--output", required=True, help="Output CSV path (features × samples).")
    gp.add_argument("--n-features", type=int, default=500)
    gp.add_argument("--n-samples", type=int, default=200)
    gp.add_argument("--n-modules", type=int, default=10)
    gp.add_argument("--within-corr", type=float, default=0.8,
                    help="Within-module correlation (default: 0.8).")
    gp.add_argument("--between-corr", type=float, default=0.0,
                    help="Between-module correlation (default: 0.0).")
    gp.add_argument("--noise-std", type=float, default=0.2)
    gp.add_argument("--seed", type=int, default=13)
    gp.add_argument("--save-ground-truth", help="Path to save ground-truth module labels CSV.")

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    bp = subs.add_parser("benchmark", help="Benchmark module discovery (ARI/NMI).")
    bp.add_argument("--dataset", required=True, help="Expression matrix to train on.")
    bp.add_argument("--ground-truth", required=True, help="CSV with ground-truth module labels.")
    bp.add_argument("--model-path", required=True, help="Trained GMMModuleVAE directory.")
    bp.add_argument("--expr", default=None,
                    help="Expression matrix for eigengene merging (default: --dataset).")
    bp.add_argument("--no-postprocess", action="store_true", default=False,
                    help="Score raw argmax(γ) instead of post-processed assignments.")
    bp.add_argument("--min-gamma", type=float, default=0.5)
    bp.add_argument("--min-module-size", type=int, default=10)
    bp.add_argument("--merge-cut-height", type=float, default=0.25)
    bp.add_argument("--no-merge", action="store_true", default=False)
    bp.add_argument("--output", required=True, help="Output JSON with benchmark metrics.")
    bp.add_argument("--batch-size", type=int, default=128)
    bp.add_argument("--no-cuda", action="store_true", default=False)

    # ------------------------------------------------------------------
    # init-config
    # ------------------------------------------------------------------
    cp = subs.add_parser("init-config", help="Write a starter simulation config file.")
    cp.add_argument("--output", default="sim.yaml", help="Destination config path.")

    # ------------------------------------------------------------------
    # generate-grid
    # ------------------------------------------------------------------
    gp2 = subs.add_parser("generate-grid", help="Generate a scenario grid from config.")
    gp2.add_argument("--config", required=True, help="Simulation config path.")
    gp2.add_argument("--outdir", required=True, help="Output directory for scenario grid.")
    gp2.add_argument("--reps", type=int, default=None, help="Override replicates per scenario.")
    gp2.add_argument("--base-seed", type=int, default=None, help="Override base seed for all runs.")

    # ------------------------------------------------------------------
    # generate-scenario
    # ------------------------------------------------------------------
    sp = subs.add_parser("generate-scenario", help="Generate a single scenario/replicate from config.")
    sp.add_argument("--config", required=True, help="Simulation config path.")
    sp.add_argument("--scenario-id", required=True, help="Scenario identifier from config expansion.")
    sp.add_argument("--rep", type=int, default=0, help="Replicate index.")
    sp.add_argument("--outdir", required=True, help="Output directory for scenario grid.")
    sp.add_argument("--base-seed", type=int, default=None, help="Override base seed for this run.")

    # ------------------------------------------------------------------
    # validate-grid
    # ------------------------------------------------------------------
    vp = subs.add_parser("validate-grid", help="Validate scenario grid output files.")
    vp.add_argument("--grid-dir", required=True, help="Scenario grid output directory.")

    return parser


def handle_generate(args, logger: logging.Logger) -> None:
    from bsvae.simulation.generate import simulate_omics_data
    import pandas as pd
    import numpy as np

    logger.info(
        "Simulating %d features × %d samples, %d modules",
        args.n_features, args.n_samples, args.n_modules,
    )
    X, ground_truth, metadata = simulate_omics_data(
        n_features=args.n_features,
        n_samples=args.n_samples,
        n_modules=args.n_modules,
        within_module_correlation=args.within_corr,
        between_module_correlation=args.between_corr,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    # Build feature IDs: feature_0001, feature_0002, …
    feat_ids = [f"feature_{i:04d}" for i in range(args.n_features)]
    sample_ids = [f"sample_{j:04d}" for j in range(args.n_samples)]

    df = pd.DataFrame(X, index=feat_ids, columns=sample_ids)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if out.suffix == ".tsv" else ","
    df.to_csv(out, sep=sep)
    logger.info("Saved simulated data to %s", out)

    if args.save_ground_truth:
        gt_df = pd.DataFrame({
            "feature_id": feat_ids,
            "module": ground_truth,
        })
        gt_df.to_csv(args.save_ground_truth, index=False)
        logger.info("Saved ground-truth labels to %s", args.save_ground_truth)

    logger.info("Simulation metadata: %s", metadata)


def handle_benchmark(args, logger: logging.Logger) -> None:
    import json
    import numpy as np
    import pandas as pd
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import create_dataloader_from_expression
    from bsvae.networks.module_extraction import extract_gmm_modules
    from bsvae.simulation.metrics import compute_ari, compute_nmi

    # Load ground truth
    gt_df = pd.read_csv(args.ground_truth)
    gt_labels = gt_df["module"].values

    # Load model and extract assignments
    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, feature_ids, _ = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )
    result = extract_gmm_modules(model, dataloader, feature_ids=feature_ids)
    pred_labels = result["hard_assignments"]
    if not getattr(args, "no_postprocess", False):
        from bsvae.networks.postprocess import postprocess_assignments
        from bsvae.networks.extract_networks import load_expression
        expr = load_expression(args.expr or args.dataset)
        post = postprocess_assignments(
            result["gamma"], feature_ids, expr,
            min_gamma=args.min_gamma, min_module_size=args.min_module_size,
            merge_cut_height=0.0 if args.no_merge else args.merge_cut_height,
        )
        pred_labels = post.labels

    # Align by feature_id (always); fall back to positional order only when
    # the truth file carries no IDs and the lengths agree.
    if "feature_id" in gt_df.columns:
        id_to_gt = dict(zip(gt_df["feature_id"].astype(str), gt_labels))
        keep = [i for i, fid in enumerate(feature_ids) if str(fid) in id_to_gt]
        if len(keep) < 0.9 * len(feature_ids):
            raise ValueError(
                f"Only {len(keep)}/{len(feature_ids)} dataset features match the ground-truth "
                "feature_id column; check that the two files describe the same features."
            )
        gt_labels = np.array([id_to_gt[str(feature_ids[i])] for i in keep])
        pred_labels = pred_labels[keep]
    elif len(gt_labels) != len(feature_ids):
        raise ValueError(
            "Ground truth has no feature_id column and its length differs from the dataset."
        )

    from bsvae.simulation.metrics import evaluate_labels
    metrics = evaluate_labels(np.asarray(gt_labels), np.asarray(pred_labels))
    metrics["n_features"] = int(len(gt_labels))
    ari, nmi = metrics["ari_all"], metrics["nmi_all"]
    logger.info(
        "ARI=%.4f  NMI=%.4f  (assigned-only ARI=%.4f, grey=%d, n=%d features)",
        ari, nmi, metrics["ari_assigned"], metrics["n_pred_grey"], len(gt_labels),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Benchmark metrics saved to %s", out)


def handle_init_config(args, logger: logging.Logger) -> None:
    from bsvae.simulation.scenario import write_starter_config

    write_starter_config(args.output)
    logger.info("Starter simulation config written to %s", args.output)


def handle_generate_grid(args, logger: logging.Logger) -> None:
    from bsvae.simulation.scenario import load_config, generate_grid

    config = load_config(args.config)
    summary = generate_grid(
        config=config,
        outdir=args.outdir,
        reps=args.reps,
        base_seed=args.base_seed,
    )
    logger.info(
        "Grid complete: %d scenarios × %d reps = %d runs",
        summary["n_scenarios"],
        summary["n_reps"],
        summary["n_runs"],
    )
    logger.info("Manifest: %s", summary["manifest"])


def handle_generate_scenario(args, logger: logging.Logger) -> None:
    from bsvae.simulation.scenario import load_config, generate_scenario

    config = load_config(args.config)
    run_dir = generate_scenario(
        config=config,
        scenario_id=args.scenario_id,
        rep=args.rep,
        outdir=args.outdir,
        base_seed=args.base_seed,
    )
    logger.info("Scenario generated: %s", run_dir)


def handle_validate_grid(args, logger: logging.Logger) -> None:
    from bsvae.simulation.scenario import validate_grid

    result = validate_grid(args.grid_dir)
    logger.info(
        "Grid validation: checked %d/%d rows, all required files present=%s",
        result["checked_rows"],
        result["manifest_rows"],
        result["all_required_files_present"],
    )


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    if args.command == "generate":
        handle_generate(args, logger)
    elif args.command == "benchmark":
        handle_benchmark(args, logger)
    elif args.command == "init-config":
        handle_init_config(args, logger)
    elif args.command == "generate-grid":
        handle_generate_grid(args, logger)
    elif args.command == "generate-scenario":
        handle_generate_scenario(args, logger)
    elif args.command == "validate-grid":
        handle_validate_grid(args, logger)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    cli()
