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
    leiden_modules_from_graph,
    leiden_modules,
    load_adjacency,
    optimize_resolution_modularity,
    prepare_leiden_graph,
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
    module_parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution parameter (ignored if --resolutions or --resolution-auto is used)")
    module_parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        help="Multiple resolution values for sweep (e.g., --resolutions 1.0 2.0 5.0 10.0). "
        "Creates separate output subdirectories for each resolution.",
    )
    module_parser.add_argument(
        "--resolution-auto",
        action="store_true",
        help="Automatically select resolution by maximizing modularity (no ground truth needed). "
        "Can be combined with --resolutions to also run fixed resolutions.",
    )
    module_parser.add_argument(
        "--resolution-min",
        type=float,
        default=0.5,
        help="Minimum resolution for auto-optimization search (default: 0.5)",
    )
    module_parser.add_argument(
        "--resolution-max",
        type=float,
        default=15.0,
        help="Maximum resolution for auto-optimization search (default: 15.0)",
    )
    module_parser.add_argument(
        "--resolution-steps",
        type=int,
        default=30,
        help="Number of resolution values to test during auto-optimization (default: 30). "
        "Ignored when --resolution-two-phase is used.",
    )
    module_parser.add_argument(
        "--resolution-two-phase",
        action="store_true",
        help="Use memory-efficient two-phase search: coarse sweep to find ballpark, "
        "then fine sweep in narrower range. Reduces peak memory usage.",
    )
    module_parser.add_argument(
        "--resolution-coarse-steps",
        type=int,
        default=10,
        help="Number of steps in coarse phase of two-phase search (default: 10)",
    )
    module_parser.add_argument(
        "--resolution-fine-steps",
        type=int,
        default=10,
        help="Number of steps in fine phase of two-phase search (default: 10)",
    )
    module_parser.add_argument(
        "--resolution-fine-range",
        type=float,
        default=0.3,
        help="Fraction of original range for fine search, centered on best coarse "
        "resolution (default: 0.3 = 30%% of original range)",
    )
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
    module_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during resolution auto-optimization",
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


def _run_single_clustering(
    adjacency_df: pd.DataFrame,
    expr_df: pd.DataFrame,
    output_dir: Path,
    resolution: float,
    adjacency_mode: str,
    cluster_method: str,
    n_clusters: Optional[int],
    n_components: Optional[int],
    logger: logging.Logger,
    resolution_label: Optional[str] = None,
    graph=None,
    genes: Optional[Sequence[str]] = None,
    compute_eigengenes: bool = True,
    precomputed_modules: Optional[pd.Series] = None,
) -> int:
    """Run clustering at a single resolution and save results.

    Parameters
    ----------
    precomputed_modules:
        Optional pre-computed module assignments (e.g., from resolution
        optimization) to avoid redundant Leiden computation.

    Returns the number of modules detected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if precomputed_modules is not None:
        modules = precomputed_modules
        logger.info(
            "Using precomputed modules (resolution=%.3f, %d modules)",
            resolution,
            modules.nunique(),
        )
    elif cluster_method == "leiden":
        logger.info(
            "Running Leiden clustering (resolution=%.3f, adjacency_mode=%s)",
            resolution,
            adjacency_mode,
        )
        if graph is not None and genes is not None:
            modules = leiden_modules_from_graph(graph, genes, resolution)
        else:
            modules = leiden_modules(
                adjacency_df,
                resolution=resolution,
                adjacency_mode=adjacency_mode,
            )
    else:
        modules = spectral_modules(
            adjacency_df,
            n_clusters=n_clusters,
            n_components=n_components,
            adjacency_mode=adjacency_mode,
        )

    n_modules = modules.nunique()

    save_modules(modules, output_dir / "modules.csv")
    if compute_eigengenes:
        eigengenes = compute_module_eigengenes(expr_df, modules)
        save_eigengenes(eigengenes, output_dir / "eigengenes.csv")
    else:
        logger.info("Skipping eigengene computation for resolution %.3f", resolution)

    # Save resolution metadata for reproducibility
    if cluster_method == "leiden":
        metadata = {
            "resolution": resolution,
            "resolution_label": resolution_label or str(resolution),
            "n_modules": n_modules,
            "adjacency_mode": adjacency_mode,
        }
        metadata_path = output_dir / "clustering_metadata.json"
        import json
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info(
        "Resolution %.3f: %d modules detected, saved to %s",
        resolution,
        n_modules,
        output_dir,
    )
    return n_modules


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
    expr_df = load_expression(args.expr)
    leiden_graph = None
    leiden_genes = None
    if args.cluster_method == "leiden":
        leiden_graph, leiden_genes = prepare_leiden_graph(
            adjacency_df,
            adjacency_mode=args.adjacency_mode,
        )

    # Determine which resolutions to run
    # (resolution, label, output_dir, precomputed_modules)
    resolutions_to_run: List[tuple[float, str, Path, Optional[pd.Series]]] = []

    # Handle resolution auto-optimization
    if args.resolution_auto and args.cluster_method == "leiden":
        two_phase = getattr(args, "resolution_two_phase", False)
        if two_phase:
            logger.info("Running two-phase resolution auto-optimization (modularity-based)")
        else:
            logger.info("Running resolution auto-optimization (modularity-based)")
        best_res, best_qual, auto_modules = optimize_resolution_modularity(
            adjacency_df,
            adjacency_mode=args.adjacency_mode,
            resolution_min=args.resolution_min,
            resolution_max=args.resolution_max,
            n_steps=args.resolution_steps,
            graph=leiden_graph,
            return_modules=True,
            progress=getattr(args, "progress", False),
            two_phase=two_phase,
            coarse_steps=getattr(args, "resolution_coarse_steps", 10),
            fine_steps=getattr(args, "resolution_fine_steps", 10),
            fine_range_fraction=getattr(args, "resolution_fine_range", 0.3),
        )
        auto_output = Path(args.output_dir) / "res_auto"
        resolutions_to_run.append((best_res, "auto", auto_output, auto_modules))
        logger.info("Auto-selected resolution: %.3f (modularity=%.4f)", best_res, best_qual)

    # Handle resolution sweep
    if args.resolutions and args.cluster_method == "leiden":
        for res in args.resolutions:
            # Format resolution for directory name (e.g., 1.0 -> "1.0", 10.0 -> "10.0")
            res_label = f"{res:.1f}".replace(".", "_")
            res_output = Path(args.output_dir) / f"res_{res_label}"
            resolutions_to_run.append((res, f"fixed_{res}", res_output, None))

    # If no sweep or auto, use single resolution (original behavior)
    if not resolutions_to_run:
        resolutions_to_run.append((args.resolution, "single", Path(args.output_dir), None))

    # Run clustering for each resolution
    results_summary = []
    multiple_resolutions = len(resolutions_to_run) > 1
    for resolution, label, output_dir, precomputed_modules in resolutions_to_run:
        compute_eigengenes = not (label == "auto" and multiple_resolutions)
        n_modules = _run_single_clustering(
            adjacency_df=adjacency_df,
            expr_df=expr_df,
            output_dir=output_dir,
            resolution=resolution,
            adjacency_mode=args.adjacency_mode,
            cluster_method=args.cluster_method,
            n_clusters=args.n_clusters,
            n_components=args.n_components,
            logger=logger,
            resolution_label=label,
            graph=leiden_graph,
            genes=leiden_genes,
            compute_eigengenes=compute_eigengenes,
            precomputed_modules=precomputed_modules,
        )
        results_summary.append({
            "resolution": resolution,
            "label": label,
            "n_modules": n_modules,
            "output_dir": str(output_dir),
        })

    # Save summary if multiple resolutions were run
    if len(resolutions_to_run) > 1:
        import json
        summary_path = Path(args.output_dir) / "resolution_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        logger.info("Resolution sweep summary saved to %s", summary_path)

        # Also save as TSV for easy viewing
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(
            Path(args.output_dir) / "resolution_sweep_summary.tsv",
            sep="\t",
            index=False,
        )

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
