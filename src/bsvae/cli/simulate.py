"""
bsvae-simulate — Generate synthetic omics data and benchmark module methods.

Subcommands
-----------
  generate     Simulate omics data with known block-diagonal module structure.
  benchmark    ARI/NMI/enrichment vs WGCNA/GNVAE on simulated data.
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
    gp.add_argument("--seed", type=int, default=42)
    gp.add_argument("--save-ground-truth", help="Path to save ground-truth module labels CSV.")

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    bp = subs.add_parser("benchmark", help="Benchmark module discovery (ARI/NMI).")
    bp.add_argument("--dataset", required=True, help="Expression matrix to train on.")
    bp.add_argument("--ground-truth", required=True, help="CSV with ground-truth module labels.")
    bp.add_argument("--model-path", required=True, help="Trained GMMModuleVAE directory.")
    bp.add_argument("--output", required=True, help="Output JSON with benchmark metrics.")
    bp.add_argument("--batch-size", type=int, default=128)
    bp.add_argument("--no-cuda", action="store_true", default=False)

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

    # Align by feature_ids
    gt_feat_ids = gt_df["feature_id"].values if "feature_id" in gt_df.columns else None
    if gt_feat_ids is not None and len(gt_feat_ids) == len(feature_ids):
        id_to_gt = dict(zip(gt_feat_ids, gt_labels))
        gt_aligned = np.array([id_to_gt.get(fid, -1) for fid in feature_ids])
        mask = gt_aligned >= 0
        gt_labels = gt_aligned[mask]
        pred_labels = pred_labels[mask]

    ari = compute_ari(gt_labels, pred_labels)
    nmi = compute_nmi(gt_labels, pred_labels)

    metrics = {"ari": ari, "nmi": nmi, "n_features": int(len(gt_labels))}
    logger.info("ARI=%.4f  NMI=%.4f  (n=%d features)", ari, nmi, len(gt_labels))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Benchmark metrics saved to %s", out)


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    if args.command == "generate":
        handle_generate(args, logger)
    elif args.command == "benchmark":
        handle_benchmark(args, logger)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    cli()
