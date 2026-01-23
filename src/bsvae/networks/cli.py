"""Command line interface for network extraction and latent export."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from bsvae.networks.extract_networks import (
    create_dataloader_from_expression,
    load_expression,
    load_model,
    run_extraction,
    save_adjacency_matrix,
    save_adjacency_sparse,
)
from bsvae.latent.latent_export import extract_latents, save_latents
from bsvae.networks.module_extraction import (
    compute_module_eigengenes,
    leiden_modules,
    load_adjacency,
    save_eigengenes,
    save_modules,
    spectral_modules,
)
from bsvae.latent.latent_analysis import (
    correlate_with_covariates,
    extract_latents as analysis_extract_latents,
    gmm_on_z,
    kmeans_on_mu,
    save_latent_results,
    tsne_mu,
    umap_mu,
)


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
    extract_parser.add_argument("--threshold", type=float, default=0.0, help="Edge weight threshold when saving edge lists (0=auto)")
    extract_parser.add_argument("--alpha", type=float, default=0.01, help="Graphical Lasso regularization strength")
    extract_parser.add_argument("--heatmaps", action="store_true", help="Save heatmap visualizations of adjacencies")
    extract_parser.add_argument(
        "--sparse",
        action="store_true",
        default=True,
        help="Save adjacency in sparse NPZ format (default: True)",
    )
    extract_parser.add_argument(
        "--no-sparse",
        action="store_true",
        help="Disable sparse storage, use dense CSV format",
    )
    extract_parser.add_argument(
        "--compress",
        action="store_true",
        default=True,
        help="Use compression for output files (zstd for Parquet, default: True)",
    )
    extract_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression for edge list output",
    )
    extract_parser.add_argument(
        "--target-sparsity",
        type=float,
        default=0.01,
        help="Target fraction of edges to keep when using adaptive threshold (default: 0.01 = 1%%)",
    )
    extract_parser.add_argument(
        "--quantize",
        choices=["int8", "float16", "float32"],
        default="int8",
        help="Quantization level for adjacency values: int8 (smallest, ~50MB for 20k genes), "
        "float16 (balanced), float32 (no quantization). Default: int8",
    )

    latent_parser = subparsers.add_parser("export-latents", help="Export encoder mu/logvar for a dataset.")
    latent_parser.add_argument("--model-path", required=True, help="Directory with specs.json/model.pt or checkpoint path")
    latent_parser.add_argument("--dataset", required=True, help="Gene expression matrix (genes × samples)")
    latent_parser.add_argument("--output", required=True, help="Destination .csv or .h5ad for mu/logvar")
    latent_parser.add_argument("--batch-size", type=int, default=128)

    module_parser = subparsers.add_parser("extract-modules", help="Cluster adjacency matrices into gene modules.")
    module_parser.add_argument("--adjacency", help="Path to adjacency matrix (.csv/.tsv)")
    module_parser.add_argument("--model-path", help="Directory with specs.json/model.pt or checkpoint path")
    module_parser.add_argument("--dataset", help="Gene expression matrix (genes × samples) for adjacency computation")
    module_parser.add_argument("--expr", required=True, help="Expression matrix (genes × samples) for eigengenes")
    module_parser.add_argument("--output-dir", required=True, help="Directory to write module outputs")
    module_parser.add_argument("--cluster-method", choices=["leiden", "spectral"], default="leiden")
    module_parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter")
    module_parser.add_argument("--n-clusters", type=int, help="Number of clusters for spectral clustering")
    module_parser.add_argument("--n-components", type=int, help="Number of eigenvectors for spectral clustering")
    module_parser.add_argument("--batch-size", type=int, default=128)
    module_parser.add_argument(
        "--adjacency-mode",
        choices=["wgcna-signed", "signed"],
        default="wgcna-signed",
        help=(
            "How to handle negative edge weights before clustering. "
            "'wgcna-signed' (default) clips negatives to zero, matching WGCNA signed networks. "
            "'signed' preserves negative edges but is not yet supported for Leiden clustering."
        ),
    )
    module_parser.add_argument(
        "--network-method",
        default="w_similarity",
        help="Network extraction method to run when computing adjacency (default: w_similarity)",
    )

    latent_analysis_parser = subparsers.add_parser("latent-analysis", help="Sample-level latent space analysis.")
    latent_analysis_parser.add_argument("--model-path", required=True, help="Directory with specs.json/model.pt or checkpoint path")
    latent_analysis_parser.add_argument("--dataset", required=True, help="Gene expression matrix (genes × samples)")
    latent_analysis_parser.add_argument("--covariates", help="Optional covariate table indexed by sample_id")
    latent_analysis_parser.add_argument("--output-dir", required=True, help="Directory to write latent analysis results")
    latent_analysis_parser.add_argument("--batch-size", type=int, default=128)
    latent_analysis_parser.add_argument("--kmeans-k", type=int, default=0, help="Run K-means with k clusters")
    latent_analysis_parser.add_argument("--gmm-k", type=int, default=0, help="Run Gaussian Mixture with k components")
    latent_analysis_parser.add_argument("--umap", action="store_true", help="Compute UMAP embedding of mu")
    latent_analysis_parser.add_argument("--tsne", action="store_true", help="Compute t-SNE embedding of mu")
    latent_analysis_parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="Perplexity for t-SNE")

    return parser


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def handle_extract_networks(args, logger: logging.Logger) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    dataloader, genes, _ = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)

    # Handle --no-sparse and --no-compress flags
    sparse = args.sparse and not getattr(args, "no_sparse", False)
    compress = args.compress and not getattr(args, "no_compress", False)

    logger.info("Running methods: %s", ", ".join(args.methods))
    logger.info(
        "Storage options: sparse=%s, compress=%s, quantize=%s, target_sparsity=%.2f%%",
        sparse, compress, args.quantize, args.target_sparsity * 100,
    )

    results = run_extraction(
        model=model,
        dataloader=dataloader,
        genes=genes,
        methods=args.methods,
        threshold=args.threshold,
        alpha=args.alpha,
        output_dir=args.output_dir,
        create_heatmaps=args.heatmaps,
        sparse=sparse,
        compress=compress,
        target_sparsity=args.target_sparsity,
        quantize=args.quantize,
    )
    logger.info("Completed extraction; saved results to %s", args.output_dir)


def handle_export_latents(args, logger: logging.Logger) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    dataloader, genes, sample_ids = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)

    mu, logvar, sample_ids = extract_latents(model, dataloader)
    save_latents(mu, logvar, sample_ids, args.output)
    logger.info("Saved mu/logvar to %s", args.output)


def _infer_separator(path: str) -> str:
    return "\t" if Path(path).suffix.lower() == ".tsv" else ","


def _load_covariates(path: str, sample_ids: Sequence[str]) -> pd.DataFrame:
    sep = _infer_separator(path)
    df = pd.read_csv(path, sep=sep)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    elif len(df.columns) > 0:
        df = df.set_index(df.columns[0])
    df = df.reindex(sample_ids)
    return df


def handle_extract_modules(args, logger: logging.Logger) -> None:
    if not args.adjacency and (not args.model_path or not args.dataset):
        raise ValueError("Either --adjacency or both --model-path and --dataset must be provided")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    genes: Optional[List[str]] = None
    adjacency_path: Optional[Path] = None

    if args.adjacency:
        adjacency, genes = load_adjacency(args.adjacency)
        adjacency_path = Path(args.adjacency)
    else:
        logger.info("Computing adjacency using method %s", args.network_method)
        model = load_model(args.model_path)
        dataloader, genes, _ = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)
        results = run_extraction(
            model=model,
            dataloader=dataloader,
            genes=genes,
            methods=[args.network_method],
            output_dir=None,
        )
        if not results:
            raise RuntimeError("No adjacency matrices were produced")
        adjacency = results[0].adjacency
        # Save in sparse compressed format by default
        adjacency_path = Path(args.output_dir) / f"{args.network_method}_adjacency.npz"
        save_adjacency_sparse(adjacency, adjacency_path.as_posix(), genes, threshold=0.0, compress=True)
        logger.info("Saved computed adjacency to %s", adjacency_path)

    # Convert adjacency to DataFrame with gene labels so clustering functions
    # return modules indexed by gene name instead of numeric indices
    adjacency_df = pd.DataFrame(adjacency, index=genes, columns=genes)

    if args.cluster_method == "leiden":
        logger.info(
            "Running Leiden clustering (resolution=%.2f, adjacency_mode=%s)",
            args.resolution,
            args.adjacency_mode,
        )
        modules = leiden_modules(
            adjacency_df,
            resolution=args.resolution,
            adjacency_mode=args.adjacency_mode,
        )
    else:
        modules = spectral_modules(adjacency_df, n_clusters=args.n_clusters, n_components=args.n_components)

    expr_df = load_expression(args.expr)
    eigengenes = compute_module_eigengenes(expr_df, modules)

    save_modules(modules, Path(args.output_dir) / "modules.csv")
    save_eigengenes(eigengenes, Path(args.output_dir) / "eigengenes.csv")
    logger.info("Module extraction complete; outputs saved to %s", args.output_dir)


def handle_latent_analysis(args, logger: logging.Logger) -> None:
    logger.info("Loading model from %s", args.model_path)
    model = load_model(args.model_path)
    dataloader, genes, sample_ids = create_dataloader_from_expression(args.dataset, batch_size=args.batch_size)

    mu, logvar, z = analysis_extract_latents(model, dataloader)

    clusters: Optional[np.ndarray] = None
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
        cov_df = _load_covariates(args.covariates, sample_ids)
        if cov_df.empty:
            logger.warning("Covariate table is empty after aligning to samples")
        else:
            correlation_df = correlate_with_covariates(pd.DataFrame(mu, index=sample_ids), cov_df)

    save_latent_results(
        mu=mu,
        logvar=logvar,
        sample_ids=sample_ids,
        output_dir=args.output_dir,
        cluster_labels=clusters,
        embedding=embedding,
        correlation_df=correlation_df,
    )
    logger.info("Latent analysis outputs written to %s", args.output_dir)


def cli(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging()

    if args.command == "extract-networks":
        handle_extract_networks(args, logger)
    elif args.command == "export-latents":
        handle_export_latents(args, logger)
    elif args.command == "extract-modules":
        handle_extract_modules(args, logger)
    elif args.command == "latent-analysis":
        handle_latent_analysis(args, logger)
    else:  # pragma: no cover - safeguarded by argparse
        parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    cli()
