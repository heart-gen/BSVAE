"""
bsvae-networks — Network extraction, module assignment, latent export.

Subcommands
-----------
  extract-networks   Method A (μ cosine) or Method B (FAISS γ-kNN)
  extract-modules    GMM soft/hard assignments + soft-weighted eigengenes
  consensus-modules  Consensus (majority vote) over several extract-modules runs
  isoform-residuals  Gene-programme-removed transcript matrix (residuals)
  isoform-specific   Call isoforms whose residual belongs to another module
  target-genes       Test isoforms of candidate genes / GWAS loci against the module solution
  module-quality     kME summaries, enrichment, preservation in a second dataset
  regress-covariates Remove known sample covariates before training
  export-latents     Save μ, logσ², γ as NPZ
  latent-analysis    UMAP, clustering, covariate correlation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


def _read_table(path: str, index_col: int = 0):
    """Read a CSV/TSV (optionally gzipped) table with the separator inferred from the suffixes."""
    import pandas as pd
    suffixes = [s.lower() for s in Path(path).suffixes]
    sep = "\t" if ".tsv" in suffixes else ","
    return pd.read_csv(path, index_col=index_col, sep=sep)


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
    mp.add_argument("--no-postprocess", action="store_true", default=False,
                    help="Write raw argmax(γ) assignments only (no grey class, min size, merging).")
    mp.add_argument("--min-gamma", type=float, default=0.5,
                    help="Features whose top γ is below this are unassigned/grey (default: 0.5).")
    mp.add_argument("--min-module-size", type=int, default=10,
                    help="Dissolve modules with fewer features (default: 10, as WGCNA).")
    mp.add_argument("--merge-cut-height", type=float, default=0.4,
                    help="Merge modules whose eigengene dissimilarity (1 - cor) is below this "
                         "(default: 0.4; WGCNA uses 0.25 on its tree cut, but GMM components that "
                         "split one module have noisier eigengenes). Requires --expr.")
    mp.add_argument("--no-merge", action="store_true", default=False,
                    help="Skip eigengene-correlation merging.")
    mp.add_argument("--kme-reassign", action="store_true", default=False,
                    help="Reassign features by module membership (kME) after merging. Requires --expr.")
    mp.add_argument("--min-kme", type=float, default=0.3)
    mp.add_argument("--kme-margin", type=float, default=0.1)
    mp.add_argument("--tx2gene", type=str, default=None,
                    help="TSV (transcript_id, gene_id) for isoform-to-gene aggregation.")
    mp.add_argument("--aggregate-to-gene", action="store_true", default=False,
                    help="Aggregate isoform-level γ to gene level by averaging across "
                         "isoforms per gene. Saves gamma_gene.npz and "
                         "hard_assignments_gene.npz. Requires --tx2gene.")

    # ------------------------------------------------------------------
    # consensus-modules
    # ------------------------------------------------------------------
    cm = subs.add_parser("consensus-modules",
                         help="Consensus of several extract-modules runs (e.g. different seeds).")
    cm.add_argument("--modules", nargs="+", required=True,
                    help="Two or more modules.csv files (feature_id, module) from extract-modules.")
    cm.add_argument("--output-dir", required=True)
    cm.add_argument("--min-agreement", type=float, default=0.5,
                    help="Features whose consensus label is supported by fewer than this fraction "
                         "of runs are unassigned (default: 0.5).")
    cm.add_argument("--min-module-size", type=int, default=10)

    # ------------------------------------------------------------------
    # isoform-residuals
    # ------------------------------------------------------------------
    ir = subs.add_parser("isoform-residuals",
                         help="Remove each isoform's gene-level programme; cluster the residuals for "
                              "isoform-specific modules.")
    ir.add_argument("--dataset", required=True, help="Transcript matrix (transcripts × samples, log scale).")
    ir.add_argument("--tx2gene", required=True)
    ir.add_argument("--output", required=True, help="Residual matrix path (.tsv.gz / .csv).")
    ir.add_argument("--stats", default=None, help="Optional per-transcript statistics table.")
    ir.add_argument("--min-isoforms", type=int, default=2)
    ir.add_argument("--no-leave-one-out", action="store_true", default=False,
                    help="Regress each isoform on the full gene aggregate (including itself) instead of its siblings.")
    ir.add_argument("--keep-tx-version", action="store_true", default=False)

    # ------------------------------------------------------------------
    # isoform-specific
    # ------------------------------------------------------------------
    isp = subs.add_parser("isoform-specific",
                          help="Call isoforms whose gene-programme-removed profile belongs to another module.")
    isp.add_argument("--dataset", required=True, help="Transcript matrix (transcripts × samples).")
    isp.add_argument("--tx2gene", required=True)
    isp.add_argument("--modules", required=True, help="modules.csv from extract-modules on the same matrix.")
    isp.add_argument("--output", required=True, help="Calls table (.csv / .tsv).")
    isp.add_argument("--min-kme", type=float, default=0.3,
                     help="Minimum residual kME to another module (default: 0.3).")
    isp.add_argument("--alpha", type=float, default=0.05,
                     help="BH false-discovery rate for the residual correlation test (default: 0.05); "
                          "0 or a negative value disables the test.")
    isp.add_argument("--no-residualize-eigengenes", action="store_true", default=False,
                     help="Correlate residuals with the raw eigengenes instead of eigengenes residualised "
                          "on the gene's own module (more false calls among unswitched isoforms).")
    isp.add_argument("--no-leave-one-out", action="store_true", default=False,
                     help="Regress each isoform on the full gene aggregate (including itself) instead of its siblings.")
    isp.add_argument("--keep-tx-version", action="store_true", default=False)

    # ------------------------------------------------------------------
    # target-genes
    # ------------------------------------------------------------------
    tg = subs.add_parser("target-genes",
                         help="Test isoforms of candidate genes or GWAS loci for module membership against a "
                              "transcriptome-wide module solution.")
    tg.add_argument("--dataset", required=True, help="Transcript matrix (transcripts × samples, log scale).")
    tg.add_argument("--tx2gene", required=True)
    tg.add_argument("--modules", required=True, help="modules.csv from extract-modules on the same matrix.")
    tg.add_argument("--output-dir", required=True)
    tg.add_argument("--genes", default=None, help="Gene list (one Ensembl ID or symbol per line).")
    tg.add_argument("--transcripts", default=None, help="Transcript list (one ID per line).")
    tg.add_argument("--bed", default=None, help="BED file of loci (index SNPs or credible-set intervals).")
    tg.add_argument("--annotation", default=None,
                    help="Gene annotation (GTF/GFF, or BED-like chrom start end gene_id [gene_name] [strand]); "
                         "required with --bed, optional with --genes to resolve symbols.")
    tg.add_argument("--rule", choices=["overlap", "window", "closest"], default="window",
                    help="Locus-to-gene rule for --bed (default: window).")
    tg.add_argument("--window", type=int, default=500_000, help="Window (bp) on each side of a locus (default: 500000).")
    tg.add_argument("--n-closest", type=int, default=3, help="Genes per locus for --rule closest (default: 3).")
    tg.add_argument("--null", choices=["permutation", "matched", "analytic"], default="permutation")
    tg.add_argument("--n-perm", type=int, default=1000)
    tg.add_argument("--n-matched", type=int, default=200)
    tg.add_argument("--min-kme", type=float, default=0.2,
                    help="Effect-size floor (default 0.2; 0.3 reproduces the transcriptome-wide caller).")
    tg.add_argument("--alpha", type=float, default=0.05, help="BH FDR within the target set.")
    tg.add_argument("--counts", default=None, help="Raw count matrix (same features × samples) for reliability-corrected kME.")
    tg.add_argument("--covariates", default=None, help="Sample table (samples × columns) for --condition.")
    tg.add_argument("--condition", default=None, help="Two-level column of --covariates for a case/control interaction test.")
    tg.add_argument("--n-neighbours", type=int, default=20)
    tg.add_argument("--no-residualize-eigengenes", action="store_true", default=False)
    tg.add_argument("--no-leave-one-out", action="store_true", default=False)
    tg.add_argument("--keep-tx-version", action="store_true", default=False)
    tg.add_argument("--seed", type=int, default=0)

    # ------------------------------------------------------------------
    # regress-covariates
    # ------------------------------------------------------------------
    rc = subs.add_parser("regress-covariates",
                         help="Remove known sample covariates (batch, sex, ...) from every feature profile.")
    rc.add_argument("--dataset", required=True, help="Expression matrix (features × samples).")
    rc.add_argument("--covariates", required=True, help="TSV/CSV with one row per sample (index = sample ID).")
    rc.add_argument("--columns", default=None, help="Comma list of covariate columns to use (default: all).")
    rc.add_argument("--output", required=True, help="Corrected matrix (.tsv.gz / .csv).")

    # ------------------------------------------------------------------
    # module-quality
    # ------------------------------------------------------------------
    mq = subs.add_parser("module-quality",
                         help="kME summaries, optional annotation enrichment and preservation in a second dataset.")
    mq.add_argument("--modules", required=True, help="modules.csv (feature_id, module) from extract-modules.")
    mq.add_argument("--expr", required=True, help="Expression matrix the modules were defined on (features × samples).")
    mq.add_argument("--output-dir", required=True)
    mq.add_argument("--test-expr", default=None,
                    help="Second expression matrix (features × samples) to test module preservation in.")
    mq.add_argument("--n-perm", type=int, default=200)
    mq.add_argument("--annotations", default=None,
                    help="GMT file (set name, description, members...) for per-module Fisher enrichment.")
    mq.add_argument("--min-overlap", type=int, default=3)

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
    la.add_argument("--covariates", help="TSV/CSV with sample covariates (rows = sample IDs).")
    la.add_argument("--expr", help="Expression matrix (features × samples) used to compute module "
                                   "eigengenes; required for --covariates correlations.")
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


def _aggregate_gamma_to_gene(
    gamma: "np.ndarray",
    feature_ids: List[str],
    tx2gene_path: str,
    weights: "Optional[np.ndarray]" = None,
    labels: "Optional[np.ndarray]" = None,
) -> "tuple[np.ndarray, list[str]]":
    """
    Aggregate isoform-level γ (F_tx, K) to gene level.

    Isoforms are weighted by ``weights`` (default: their maximum γ, so
    confident isoforms dominate; pass ``None`` for a plain mean) and rows are
    renormalised to sum to 1.  Features missing from the mapping map to
    themselves.  When ``labels`` are given, the function also returns, as the
    third element, the list of genes whose isoforms are assigned to two or
    more different modules (isoform-divergent genes).
    """
    import numpy as np
    from collections import defaultdict
    from bsvae.utils.hierarchy import load_tx2gene, strip_feature_version

    tx2gene = load_tx2gene(path=tx2gene_path, feature_ids=feature_ids)
    fid_to_gene = {}
    for fid in feature_ids:
        key = strip_feature_version(fid)
        g = tx2gene.get(key, None)
        fid_to_gene[fid] = str(g) if g is not None else fid

    gene_to_idxs: dict = defaultdict(list)
    for i, fid in enumerate(feature_ids):
        gene_to_idxs[fid_to_gene[fid]].append(i)

    if weights is None:
        w = np.ones(gamma.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).clip(min=1e-6)

    gene_ids = list(gene_to_idxs.keys())
    gamma_gene = np.zeros((len(gene_ids), gamma.shape[1]), dtype=np.float32)
    divergent = []
    for j, gid in enumerate(gene_ids):
        idxs = gene_to_idxs[gid]
        ww = w[idxs][:, None]
        gamma_gene[j] = (gamma[idxs] * ww).sum(axis=0) / ww.sum()
        if labels is not None:
            mods = {int(m) for m in labels[idxs] if int(m) >= 0}
            if len(mods) >= 2:
                divergent.append((gid, sorted(mods)))

    row_sums = gamma_gene.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    gamma_gene /= row_sums
    if labels is not None:
        return gamma_gene, gene_ids, divergent
    return gamma_gene, gene_ids


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

    expr = None
    if args.expr:
        expr = _read_table(args.expr)
        expr.index = expr.index.astype(str)

    gamma_final = result["gamma"]
    labels_final = None
    if not args.no_postprocess:
        from bsvae.networks.postprocess import postprocess_assignments, save_postprocess
        if expr is None and (not args.no_merge or args.kme_reassign):
            logger.warning(
                "--expr not given: eigengene merging and kME reassignment are skipped; "
                "only the confidence cutoff and minimum module size are applied."
            )
        post = postprocess_assignments(
            result["gamma"], feature_ids, expr,
            min_gamma=args.min_gamma,
            min_module_size=args.min_module_size,
            merge_cut_height=0.0 if args.no_merge else args.merge_cut_height,
            kme_reassign_=args.kme_reassign,
            min_kme=args.min_kme,
            kme_margin=args.kme_margin,
        )
        save_postprocess(post, args.output_dir, feature_ids)
        gamma_final = post.gamma
        labels_final = post.labels
        logger.info(
            "Post-processed: %d modules, %d grey (%.1f%%)",
            post.n_modules, post.n_grey, 100.0 * post.n_grey / max(len(feature_ids), 1),
        )

    if args.aggregate_to_gene:
        if not args.tx2gene:
            logger.warning("--aggregate-to-gene requires --tx2gene; skipping.")
        else:
            agg = _aggregate_gamma_to_gene(
                gamma_final, result["feature_ids"], args.tx2gene,
                weights=gamma_final.max(axis=1), labels=labels_final,
            )
            gamma_gene, gene_ids = agg[0], agg[1]
            hard_gene = gamma_gene.argmax(axis=1).astype(np.int32)
            out = Path(args.output_dir)
            if labels_final is not None:
                import pandas as pd
                divergent = agg[2]
                pd.DataFrame(
                    {"gene_id": [g for g, _ in divergent],
                     "modules": [";".join(map(str, m)) for _, m in divergent]}
                ).to_csv(out / "gene_divergent.csv", index=False)
                logger.info("%d genes have isoforms in ≥2 modules (gene_divergent.csv)", len(divergent))
            np.savez_compressed(
                out / "gamma_gene.npz",
                gamma=gamma_gene,
                feature_ids=np.array(gene_ids, dtype=object),
            )
            np.savez_compressed(
                out / "hard_assignments_gene.npz",
                hard_assignments=hard_gene,
                feature_ids=np.array(gene_ids, dtype=object),
            )
            logger.info(
                "Gene-level aggregation: %d transcripts → %d genes",
                len(result["feature_ids"]), len(gene_ids),
            )

    if args.soft_eigengenes and expr is not None:
        eigengenes = compute_module_eigengenes_from_soft(
            expr, result["gamma"], feature_ids
        )
        out = Path(args.output_dir) / "soft_eigengenes.csv"
        eigengenes.to_csv(out)
        logger.info("Soft-weighted eigengenes saved to %s", out)

    if args.use_leiden:
        from bsvae.networks.module_extraction import leiden_modules, save_modules
        from bsvae.networks.extract_networks import method_a_cosine
        # Sparse path: never densify the F × F adjacency.
        adj = method_a_cosine(result["mu"], top_k=50).tocoo()
        upper = adj.row < adj.col
        edges = np.column_stack((adj.row[upper], adj.col[upper]))
        weights = adj.data[upper]
        modules = leiden_modules(
            edges=edges, weights=weights, genes=list(feature_ids),
            resolution=args.leiden_resolution,
        )
        save_modules(modules, Path(args.output_dir) / "leiden_modules.csv")
        logger.info("Leiden fallback modules saved.")


def handle_isoform_specific(args, logger: logging.Logger) -> None:
    import pandas as pd
    from bsvae.utils.hierarchy import load_tx2gene
    from bsvae.utils.isoform import isoform_specific_calls

    expr = _read_table(args.dataset)
    mods = pd.read_csv(args.modules)
    fids = [str(f) for f in expr.index]
    labels = mods.set_index(mods["feature_id"].astype(str))["module"].reindex(fids).fillna(-1).astype(int).values
    tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=fids, keep_version=args.keep_tx_version, logger=logger)
    calls = isoform_specific_calls(
        expr, tx2gene, labels, fids, min_kme=args.min_kme, keep_version=args.keep_tx_version,
        alpha=args.alpha if args.alpha > 0 else None,
        residualize_eigengenes=not args.no_residualize_eigengenes,
        leave_one_out=not args.no_leave_one_out,
    )
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    calls.to_csv(out, sep="\t" if out.suffix == ".tsv" else ",")
    logger.info("Isoform-specific calls written to %s (%d called)", out, int(calls["called"].sum()))


def handle_target_genes(args, logger: logging.Logger) -> None:
    import pandas as pd
    from bsvae.networks.targets import (
        loci_to_genes, locus_summary, read_bed, read_gene_annotation, read_id_list,
        resolve_target_transcripts, target_gene_tests,
    )
    from bsvae.utils.hierarchy import load_tx2gene
    from bsvae.utils.isoform import map_transcripts_to_genes

    if not (args.genes or args.transcripts or args.bed):
        raise SystemExit("target-genes needs at least one of --genes, --transcripts, --bed")
    if args.bed and not args.annotation:
        raise SystemExit("--bed requires --annotation (GTF or BED-like gene table)")
    expr = _read_table(args.dataset)
    mods = pd.read_csv(args.modules)
    fids = [str(f) for f in expr.index]
    labels = mods.set_index(mods["feature_id"].astype(str))["module"].reindex(fids).fillna(-1).astype(int).values
    tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=fids, keep_version=args.keep_tx_version, logger=logger)
    tx_gene = map_transcripts_to_genes(fids, tx2gene, args.keep_tx_version)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    annotation = read_gene_annotation(args.annotation, args.keep_tx_version) if args.annotation else None
    genes = read_id_list(args.genes, args.keep_tx_version) if args.genes else []
    loci_genes = None
    if args.bed:
        loci = read_bed(args.bed)
        loci_genes = loci_to_genes(loci, annotation, rule=args.rule, window=args.window, n_closest=args.n_closest)
        loci_genes.to_csv(out_dir / "loci_genes.tsv", sep="\t", index=False)
        logger.info("%d loci → %d genes (rule=%s)", loci["locus"].nunique(), loci_genes["gene_id"].nunique(), args.rule)
        genes = list(dict.fromkeys(genes + loci_genes["gene_id"].tolist()))
    transcripts = read_id_list(args.transcripts, args.keep_tx_version) if args.transcripts else []
    targets, unmatched = resolve_target_transcripts(tx_gene, genes, transcripts, annotation, args.keep_tx_version)
    if unmatched:
        logger.warning("%d target identifiers not found in the expression matrix / tx2gene: %s%s",
                       len(unmatched), ", ".join(unmatched[:10]), " ..." if len(unmatched) > 10 else "")
    if len(targets) == 0:
        raise SystemExit("No target transcripts present in the expression matrix")
    logger.info("Testing %d isoforms of %d target genes", len(targets), targets.nunique())

    counts = _read_table(args.counts) if args.counts else None
    condition = None
    if args.condition:
        if not args.covariates:
            raise SystemExit("--condition requires --covariates")
        cov = _read_table(args.covariates)
        condition = cov[args.condition]
    res = target_gene_tests(
        expr, tx2gene, labels, fids, targets, null=args.null, n_perm=args.n_perm, n_matched=args.n_matched,
        min_kme=args.min_kme, alpha=args.alpha, counts=counts, condition=condition, n_neighbours=args.n_neighbours,
        residualize_eigengenes=not args.no_residualize_eigengenes, leave_one_out=not args.no_leave_one_out,
        keep_version=args.keep_tx_version, seed=args.seed,
    )
    res["isoforms"].to_csv(out_dir / "target_isoforms.tsv", sep="\t")
    res["genes"].to_csv(out_dir / "target_genes.tsv", sep="\t")
    res["neighbours"].to_csv(out_dir / "target_neighbours.tsv", sep="\t", index=False)
    if loci_genes is not None:
        locus_summary(res["isoforms"], loci_genes, labels).to_csv(out_dir / "target_loci.tsv", sep="\t", index=False)
    logger.info("Targeted-gene results written to %s (%d of %d isoforms called)",
                out_dir, int(res["isoforms"]["called"].sum()), len(res["isoforms"]))


def handle_regress_covariates(args, logger: logging.Logger) -> None:
    from bsvae.utils.preprocess import regress_covariates

    expr = _read_table(args.dataset)
    cov = _read_table(args.covariates)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    out_df = regress_covariates(expr, cov, columns=cols)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if ".tsv" in [s.lower() for s in out.suffixes] else ","
    out_df.to_csv(out, sep=sep, compression="gzip" if out.suffix == ".gz" else None)
    logger.info("Covariate-corrected matrix written to %s", out)


def handle_module_quality(args, logger: logging.Logger) -> None:
    import numpy as np
    import pandas as pd
    from bsvae.networks.quality import module_enrichment, module_preservation, module_quality

    mods = pd.read_csv(args.modules)
    fids = [str(f) for f in mods["feature_id"]]
    labels = mods["module"].astype(int).values
    expr = _read_table(args.expr)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    q = module_quality(expr, labels, fids)
    q.to_csv(out / "module_quality.csv", index=False)
    logger.info("Module quality for %d modules written to %s", len(q), out / "module_quality.csv")
    if args.test_expr:
        test = _read_table(args.test_expr)
        pres = module_preservation(expr, test, labels, fids, n_perm=args.n_perm)
        pres.to_csv(out / "module_preservation.csv", index=False)
        logger.info("Preservation: %d/%d modules with Z_density > 2", int((pres["Z_density"] > 2).sum()), len(pres))
    if args.annotations:
        sets = {}
        with open(args.annotations) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    sets[parts[0]] = parts[2:]
        enr = module_enrichment(labels, fids, sets, min_overlap=args.min_overlap)
        enr.to_csv(out / "module_enrichment.csv", index=False)
        logger.info("Enrichment: %d module-set pairs tested", len(enr))


def handle_isoform_residuals(args, logger: logging.Logger) -> None:
    from bsvae.utils.hierarchy import load_tx2gene
    from bsvae.utils.isoform import isoform_residuals

    expr = _read_table(args.dataset)
    tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=list(expr.index.astype(str)),
                           keep_version=args.keep_tx_version, logger=logger)
    resid, stats = isoform_residuals(expr, tx2gene, min_isoforms=args.min_isoforms,
                                     keep_version=args.keep_tx_version, leave_one_out=not args.no_leave_one_out)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if ".tsv" in [s.lower() for s in out.suffixes] else ","
    resid.to_csv(out, sep=sep, compression="gzip" if out.suffix == ".gz" else None)
    logger.info("Residual matrix (%d × %d) written to %s", *resid.shape, out)
    if args.stats:
        stats.to_csv(args.stats, sep="\t" if args.stats.endswith((".tsv", ".tsv.gz")) else ",")
        logger.info("Per-transcript statistics written to %s", args.stats)


def handle_consensus_modules(args, logger: logging.Logger) -> None:
    import numpy as np
    import pandas as pd
    from bsvae.networks.postprocess import consensus_assignments, save_consensus

    if len(args.modules) < 2:
        raise SystemExit("consensus-modules needs at least two modules.csv files")
    frames = [pd.read_csv(p) for p in args.modules]
    ids = [str(f) for f in frames[0]["feature_id"]]
    runs = []
    for p, df in zip(args.modules, frames):
        s = df.set_index(df["feature_id"].astype(str))["module"].reindex(ids)
        if s.isna().any():
            raise SystemExit(f"{p} does not cover the same features as {args.modules[0]}")
        runs.append(s.astype(int).values)
    res = consensus_assignments(runs, min_agreement=args.min_agreement, min_module_size=args.min_module_size)
    save_consensus(res, args.output_dir, ids)
    logger.info(
        "Consensus of %d runs: %d modules, %d grey, mean agreement %.3f, run ARI to consensus %s",
        res.n_runs, len(set(res.labels[res.labels >= 0])), int((res.labels < 0).sum()),
        float(res.agreement.mean()), [round(a, 3) for a in res.run_ari],
    )


def handle_export_latents(args, logger: logging.Logger) -> None:
    import numpy as np
    from bsvae.utils.modelIO import load_model
    from bsvae.networks.extract_networks import create_dataloader_from_expression

    model = load_model(args.model_path, is_gpu=not args.no_cuda)
    dataloader, feature_ids, _ = create_dataloader_from_expression(
        args.dataset, batch_size=args.batch_size
    )
    # Single pass: μ, logσ² and γ together.
    import torch
    device = next(model.parameters()).device
    model.eval()
    mu_list, logvar_list, gamma_list = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device)
            mu_b, logvar_b = model.encode(x)
            gamma_b = model.gmm_prior.posterior_weights(mu_b, logvar_b)
            mu_list.append(mu_b.cpu().numpy())
            logvar_list.append(logvar_b.cpu().numpy())
            gamma_list.append(gamma_b.cpu().numpy())
    mu = np.concatenate(mu_list, axis=0).astype(np.float32)
    logvar = np.concatenate(logvar_list, axis=0).astype(np.float32)
    gamma = np.concatenate(gamma_list, axis=0).astype(np.float32)

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
    import numpy as np
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
    dataloader, feature_ids, _ = create_dataloader_from_expression(
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
        # μ is per feature while covariates are per sample, so correlate the
        # covariates with module eigengenes (samples × modules) instead.
        if not args.expr:
            logger.warning(
                "--covariates requires --expr to compute module eigengenes; skipping correlations."
            )
        else:
            from bsvae.networks.extract_networks import extract_mu_gamma
            from bsvae.networks.module_extraction import compute_module_eigengenes
            cov_df = _read_table(args.covariates)
            cov_df.index = cov_df.index.astype(str)
            expr_df = _read_table(args.expr)
            expr_df.index = expr_df.index.astype(str)
            _, gamma, _ = extract_mu_gamma(model, dataloader, disable_progress=True)
            hard = pd.Series(gamma.argmax(axis=1), index=[str(f) for f in feature_ids], name="module")
            eigengenes = compute_module_eigengenes(expr_df, hard)
            eigengenes.index = eigengenes.index.astype(str)
            cov_aligned = cov_df.reindex(eigengenes.index)
            n_match = int(cov_aligned.notna().any(axis=1).sum())
            if n_match < 2:
                logger.warning(
                    "No sample IDs shared between covariates and expression columns; skipping."
                )
            else:
                eigengenes.columns = [f"ME{c}" for c in eigengenes.columns]
                correlation_df = correlate_with_covariates(eigengenes, cov_aligned)

    save_latent_results(
        mu=mu, logvar=logvar, row_ids=feature_ids,
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
    elif args.command == "isoform-specific":
        handle_isoform_specific(args, logger)
    elif args.command == "target-genes":
        handle_target_genes(args, logger)
    elif args.command == "regress-covariates":
        handle_regress_covariates(args, logger)
    elif args.command == "module-quality":
        handle_module_quality(args, logger)
    elif args.command == "isoform-residuals":
        handle_isoform_residuals(args, logger)
    elif args.command == "consensus-modules":
        handle_consensus_modules(args, logger)
    elif args.command == "export-latents":
        handle_export_latents(args, logger)
    elif args.command == "latent-analysis":
        handle_latent_analysis(args, logger)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    cli()
