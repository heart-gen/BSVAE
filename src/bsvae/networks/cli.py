"""Command line interface for network extraction and latent export."""
from __future__ import annotations

import argparse
import logging
from typing import List

from bsvae.networks.extract_networks import (
    create_dataloader_from_expression,
    load_model,
    run_extraction,
)
from bsvae.networks.latent_export import extract_latents, save_latents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bsvae-networks", description="Network and latent export utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract-networks", help="Compute gene–gene networks from a trained model.")
    extract_parser.add_argument("--model-path", required=True, help="Directory with specs.json/model.pt or checkpoint path")
    extract_parser.add_argument("--dataset", required=True, help="Gene expression matrix (genes × samples)")
    extract_parser.add_argument("--output-dir", required=True, help="Directory to write adjacency matrices and edge lists")
    extract_parser.add_argument(
        "--methods",
        nargs="+",
        default=["w_similarity"],
        help="Methods to run: w_similarity (default), latent_cov, graphical_lasso, laplacian",
    )
    extract_parser.add_argument("--batch-size", type=int, default=128)
    extract_parser.add_argument("--threshold", type=float, default=0.0, help="Edge weight threshold when saving edge lists")
    extract_parser.add_argument("--alpha", type=float, default=0.01, help="Graphical Lasso regularization strength")
    extract_parser.add_argument("--heatmaps", action="store_true", help="Save heatmap visualizations of adjacencies")

    latent_parser = subparsers.add_parser("export-latents", help="Export encoder mu/logvar for a dataset.")
    latent_parser.add_argument("--model-path", required=True, help="Directory with specs.json/model.pt or checkpoint path")
    latent_parser.add_argument("--dataset", required=True, help="Gene expression matrix (genes × samples)")
    latent_parser.add_argument("--output", required=True, help="Destination .csv or .h5ad for mu/logvar")
    latent_parser.add_argument("--batch-size", type=int, default=128)

    return parser


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def handle_extract_networks(args, logger: logging.Logger) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    dataloader, genes, _ = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)

    logger.info("Running methods: %s", ", ".join(args.methods))
    results = run_extraction(
        model=model,
        dataloader=dataloader,
        genes=genes,
        methods=args.methods,
        threshold=args.threshold,
        alpha=args.alpha,
        output_dir=args.output_dir,
        create_heatmaps=args.heatmaps,
    )
    logger.info("Completed extraction; saved results to %s", args.output_dir)


def handle_export_latents(args, logger: logging.Logger) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    dataloader, genes, sample_ids = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)

    mu, logvar, sample_ids = extract_latents(model, dataloader)
    save_latents(mu, logvar, sample_ids, args.output)
    logger.info("Saved mu/logvar to %s", args.output)


def cli(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    if args.command == "extract-networks":
        handle_extract_networks(args, logger)
    elif args.command == "export-latents":
        handle_export_latents(args, logger)
    else:  # pragma: no cover - safeguarded by argparse
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    cli()
