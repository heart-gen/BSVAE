"""
bsvae-networks — Network extraction, module assignment, latent export.

Subcommands
-----------
  extract-networks   Method A (μ cosine) or Method B (FAISS γ-kNN)
  extract-modules    GMM soft/hard assignments + soft-weighted eigengenes
  export-latents     Save μ, logσ², γ as NPZ
  latent-analysis    UMAP, clustering, covariate correlation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    return logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bsvae-networks",
        description="Network extraction and latent analysis for GMMModuleVAE.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # extract-networks
    # ------------------------------------------------------------------
    ep = subs.add_parser("extract-networks", help="Build feature-feature networks from trained model.")
    ep.add_argument("--model-path", required=True)
    ep.add_argument("--dataset", required=True)
    ep.add_argument("--output-dir", required=True)
    ep.add_argument(
        "--methods", nargs="+", default=["mu_cosine"],
        choices=["mu_cosine", "gamma_knn"],
        help="Methods to run (default: mu_cosine).",
    )
    ep.add_argument("--top-k", type=int, default=50, help="Top-K edges per feature (default: 50).")
    ep.add_argument("--batch-size", type=int, default=128)
    ep.add_argument("--no-cuda", action="store_true", default=False)

    # ------------------------------------------------------------------
    # extract-modules
    # ------------------------------------------------------------------
    mp = subs.add_parser("extract-modules", help="Extract GMM module assignments.")
    mp.add_argument("--model-path", required=True)
    mp.add_argument("--dataset", required=True)
    mp.add_argument("--output-dir", required=True)
    mp.add_argument("--batch-size", type=int, default=128)
    mp.add_argument("--no-cuda", action="store_true", default=False)
    mp.add_argument("--expr", help="Expression matrix for eigengene computation (features × samples).")
    mp.add_argument("--soft-eigengenes", action="store_true",
                    help="Compute soft-weighted eigengenes (requires --expr).")
    mp.add_argument("--use-leiden", action="store_true",
                    help="Also run Leiden clustering as fallback comparison.")
    mp.add_argument("--leiden-resolution", type=float, default=1.0)

    # ------------------------------------------------------------------
    # export-latents
    # ------------------------------------------------------------------
    lp = subs.add_parser("export-latents", help="Export μ, logσ², γ for all features.")
    lp.add_argument("--model-path", required=True)
    lp.add_argument("--dataset", required=True)
    lp.add_argument("--output", required=True, help="Output path (.npz).")
    lp.add_argument("--batch-size", type=int, default=128)
    lp.add_argument("--no-cuda", action="store_true", default=False)

    # ------------------------------------------------------------------
    # latent-analysis
    # ------------------------------------------------------------------
    la = subs.add_parser("latent-analysis", help="UMAP, clustering, covariate correlation on μ.")
    la.add_argument("--model-path", required=True)
    la.add_argument("--dataset", required=True)
    la.add_argument("--output-dir", required=True)
    la.add_argument("--batch-size", type=int, default=128)
    la.add_argument("--no-cuda", action="store_true", default=False)
    la.add_argument("--covariates", help="TSV/CSV with sample covariates.")
    la.add_argument("--kmeans-k", type=int, default=0)
    la.add_argument("--gmm-k", type=int, default=0)
    la.add_argument("--umap", action="store_true")
    la.add_argument("--tsne", action="store_true")
    la.add_argument("--tsne-perplexity", type=float, default=30.0)

    return parser


# ------------------------------------------------------------------
# Handlers
# ------------------------------------------------------------------

def handle_extract_networks(args, logger: logging.Logger) -> None:
    import torch
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import run_extraction, create_dataloader_from_expression

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, feature_ids, _ = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )
    logger.info("Running methods: %s | top_k=%d", args.methods, args.top_k)
    run_extraction(
        model=model,
        dataloader=dataloader,
        feature_ids=feature_ids,
        methods=args.methods,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
    logger.info("Network extraction complete → %s", args.output_dir)


def handle_extract_modules(args, logger: logging.Logger) -> None:
    import numpy as np
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import create_dataloader_from_expression
    from bsvae.networks.module_extraction import (
        extract_gmm_modules,
        compute_module_eigengenes_from_soft,
    )

    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, feature_ids, _ = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )
    result = extract_gmm_modules(
        model=model,
        dataloader=dataloader,
        feature_ids=feature_ids,
        output_dir=args.output_dir,
    )
    logger.info(
        "Extracted %d-module GMM assignments for %d features",
        model.n_modules, len(feature_ids),
    )

    if args.soft_eigengenes and args.expr:
        import pandas as pd
        expr = pd.read_csv(args.expr, index_col=0, sep="\t" if args.expr.endswith(".tsv") else ",")
        eigengenes = compute_module_eigengenes_from_soft(
            expr, result["gamma"], feature_ids
        )
        out = Path(args.output_dir) / "soft_eigengenes.csv"
        eigengenes.to_csv(out)
        logger.info("Soft-weighted eigengenes saved to %s", out)

    if args.use_leiden:
        from bsvae.networks.module_extraction import leiden_modules, save_modules
        from bsvae.networks.extract_networks import (
            method_a_cosine,
            extract_mu_gamma,
        )
        device = None
        mu, gamma, _ = extract_mu_gamma(model, dataloader, device=device)
        adj = method_a_cosine(mu, top_k=50)
        import pandas as pd
        adj_df = pd.DataFrame(
            adj.toarray(), index=feature_ids, columns=feature_ids
        )
        modules = leiden_modules(adj_df, resolution=args.leiden_resolution)
        save_modules(modules, Path(args.output_dir) / "leiden_modules.csv")
        logger.info("Leiden fallback modules saved.")


def handle_export_latents(args, logger: logging.Logger) -> None:
    import numpy as np
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import extract_mu_gamma, create_dataloader_from_expression

    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, feature_ids, _ = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )
    mu, gamma, extracted_ids = extract_mu_gamma(model, dataloader)
    if not feature_ids:
        feature_ids = extracted_ids

    # Also get logvar
    import torch
    device = next(model.parameters()).device
    model.eval()
    logvar_list = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            _, logvar = model.encode(x)
            logvar_list.append(logvar.cpu().numpy())
    import numpy as np
    logvar = np.concatenate(logvar_list, axis=0).astype(np.float32)

    out = args.output
    if not out.endswith(".npz"):
        out = out + ".npz"
    np.savez_compressed(
        out,
        mu=mu,
        logvar=logvar,
        gamma=gamma,
        feature_ids=np.array(feature_ids, dtype=object),
    )
    logger.info("Latents saved to %s", out)


def handle_latent_analysis(args, logger: logging.Logger) -> None:
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import create_dataloader_from_expression
    from bsvae.latent.latent_analysis import (
        extract_latents,
        kmeans_on_mu,
        gmm_on_z,
        umap_mu,
        tsne_mu,
        correlate_with_covariates,
        save_latent_results,
    )
    import pandas as pd
    import numpy as np

    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, _, sample_ids = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )

    mu, logvar, z = extract_latents(model, dataloader)

    clusters = None
    if args.kmeans_k:
        clusters = kmeans_on_mu(mu, k=args.kmeans_k)
    elif args.gmm_k:
        clusters, _ = gmm_on_z(z, n_components=args.gmm_k)

    embedding = None
    if args.umap:
        embedding = umap_mu(mu)
    elif args.tsne:
        embedding = tsne_mu(mu, perplexity=args.tsne_perplexity)

    correlation_df = None
    if args.covariates:
        sep = "\t" if args.covariates.endswith(".tsv") else ","
        cov_df = pd.read_csv(args.covariates, sep=sep, index_col=0)
        cov_df = cov_df.reindex(sample_ids)
        if not cov_df.empty:
            correlation_df = correlate_with_covariates(
                pd.DataFrame(mu, index=sample_ids), cov_df
            )

    save_latent_results(
        mu=mu, logvar=logvar, sample_ids=sample_ids,
        output_dir=args.output_dir,
        cluster_labels=clusters,
        embedding=embedding,
        correlation_df=correlation_df,
    )
    logger.info("Latent analysis saved to %s", args.output_dir)


def cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    if args.command == "extract-networks":
        handle_extract_networks(args, logger)
    elif args.command == "extract-modules":
        handle_extract_modules(args, logger)
    elif args.command == "export-latents":
        handle_export_latents(args, logger)
    elif args.command == "latent-analysis":
        handle_latent_analysis(args, logger)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    cli()
