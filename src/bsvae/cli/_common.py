"""
Shared argument groups and builders for the BSVAE CLIs.

``bsvae-train`` and ``bsvae-sweep-k`` (and the benchmark harness) build the
model, losses, data loader and trainer through the helpers here, so every
entry point exposes the same flags with the same defaults and the K selected
by a sweep is selected with the model the user will actually train.
"""

from __future__ import annotations

import argparse
import ast
import logging
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from bsvae.models import GMMModuleVAE
from bsvae.models.losses import GMMVAELoss, WarmupLoss
from bsvae.utils.datasets import OmicsDataset
from bsvae.utils.training import Trainer


def add_sample_subset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--samples", default=None,
                   help="Use these samples only (file with one sample id per line); for cross-fitting, train "
                        "one model per fold on the other folds' samples and extract its modules with the same subset.")
    p.add_argument("--exclude-samples", default=None,
                   help="Use every sample except these (file with one sample id per line).")


def read_id_file(path: str) -> set:
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


def sample_mask(sample_ids: Sequence[str], keep_path: Optional[str], drop_path: Optional[str]):
    """Boolean mask over ``sample_ids`` from keep and/or drop id files (None, None → all True)."""
    import numpy as np

    ids = [str(s) for s in sample_ids]
    mask = np.ones(len(ids), dtype=bool)
    if keep_path:
        keep = read_id_file(keep_path); mask &= np.array([s in keep for s in ids])
    if drop_path:
        drop = read_id_file(drop_path); mask &= np.array([s not in drop for s in ids])
    if mask.sum() < 3:
        raise ValueError(f"Sample subset leaves {int(mask.sum())} samples")
    return mask


def subset_samples(dataset: OmicsDataset, keep_path: Optional[str], drop_path: Optional[str], logger=None) -> OmicsDataset:
    """Restrict an OmicsDataset to a sample subset given by id lists (keep and/or drop)."""
    if keep_path is None and drop_path is None:
        return dataset
    mask = sample_mask(dataset.sample_ids, keep_path, drop_path)
    ids = [str(s) for s in dataset.sample_ids]
    data = dataset.data.numpy()[:, mask]
    if logger is not None:
        logger.info("Sample subset: %d of %d samples kept", int(mask.sum()), len(ids))
    return OmicsDataset(data=data, feature_ids=dataset.feature_ids, sample_ids=[s for s, m in zip(ids, mask) if m],
                        logger=logger, return_index=dataset.return_index)


def ast_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def setup_logging(level: str = "info", name: str = "bsvae") -> logging.Logger:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(numeric)
    return logger


# ---------------------------------------------------------------------------
# Argument groups
# ---------------------------------------------------------------------------

def add_general_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("general")
    g.add_argument("--outdir", type=str, default="results", help="Directory for outputs.")
    g.add_argument("--seed", type=int, default=13)
    g.add_argument("--no-cuda", action="store_true", default=False)
    g.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    g.add_argument("--no-progress-bar", action="store_true", default=False)
    g.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0).")
    g.add_argument("--amp", action="store_true", default=False,
                   help="Mixed-precision (bfloat16) forward pass on CUDA.")


def add_model_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--n-modules", "-K", type=int, default=20,
                   help="Number of GMM components / biological modules (default: 20).")
    g.add_argument("--latent-dim", "-z", type=int, default=32,
                   help="Latent space dimensionality D (default: 32).")
    g.add_argument("--hidden-dims", type=ast_eval, default=[512, 256, 128],
                   help="Encoder hidden layer sizes as Python list (default: [512,256,128]).")
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--no-batch-norm", action="store_false", dest="use_batch_norm")
    g.add_argument("--use-batch-norm", action="store_true", dest="use_batch_norm", default=True)
    g.add_argument("--sigma-min", type=float, default=0.3,
                   help="GMM component σ floor (default: 0.3).")
    g.add_argument("--normalize-input", action="store_true", default=False,
                   help="Z-score each feature profile across samples before encoding. "
                        "Recommended for count-derived data (log2-CPM, etc.).")
    g.add_argument("--diag-covariance", action="store_true", default=False,
                   help="Learn a variance per latent dimension for each GMM component "
                        "(diagonal instead of isotropic covariance).")


def add_schedule_args(p: argparse.ArgumentParser, checkpoint_default: int = 10) -> None:
    g = p.add_argument_group("schedule")
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--checkpoint-every", type=int, default=checkpoint_default)
    g.add_argument("--warmup-epochs", type=int, default=20,
                   help="Phase 1 length: standard N(0,I) VAE (default: 20).")
    g.add_argument("--transition-epochs", type=int, default=10,
                   help="GMM loss ramp-in length (default: 10).")
    g.add_argument("--freeze-gmm-epochs", type=int, default=0,
                   help="Freeze GMM prior params for first N GMM epochs (default: 0).")
    g.add_argument("--collapse-min-usage", type=float, default=0.002,
                   help="Absolute usage floor: a component whose mean γ over all features "
                        "falls below this fraction is revived by splitting (default: 0.002).")
    g.add_argument("--collapse-threshold", type=float, default=None,
                   help="Deprecated (was a relative floor threshold/K); ignored.")
    g.add_argument("--collapse-noise-scale", type=float, default=0.5,
                   help="Noise scale for dead-component revival (σ multiples, default: 0.5).")
    g.add_argument("--eval-batch-size", type=int, default=None)


def add_loss_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("loss")
    g.add_argument("--beta", type=float, default=1.0,
                   help="KL weight (default: 1.0). With per-element KL scaling this is a true "
                        "ELBO weight.")
    g.add_argument("--no-kl-per-element", action="store_false", dest="kl_per_element",
                   default=True,
                   help="Legacy KL scaling (sum over latent dims vs per-element MSE), which "
                        "makes the effective β grow with the number of samples.")
    g.add_argument("--free-bits", type=float, default=0.0,
                   help="Per-dim KL lower bound in nats (default: 0.0). "
                        "Non-zero values hurt GMM cluster recovery; leave at 0.0 for normal use.")
    g.add_argument("--kl-warmup-epochs", type=int, default=0,
                   help="Epochs for β linear ramp, counted from the start of Phase 2 (default: 0).")
    g.add_argument("--kl-anneal-mode", choices=["linear", "cyclical"], default="linear")
    g.add_argument("--kl-cycle-length", type=int, default=50)
    g.add_argument("--sep-strength", type=float, default=0.1,
                   help="λ_sep: σ-scaled separation loss weight (default: 0.1).")
    g.add_argument("--sep-alpha", type=float, default=2.0,
                   help="α: margin multiplier for separation loss (default: 2.0).")
    g.add_argument("--bal-strength", type=float, default=0.1,
                   help="λ_bal: γ-usage balance loss weight (default: 0.1).")
    g.add_argument("--bal-type", choices=["uniform", "entropy", "none"], default="entropy",
                   help="Balance loss mode (default: 'entropy', which discourages dead components "
                        "without forcing equal module sizes). 'uniform' pushes equal sizes.")
    g.add_argument("--bal-scale-gmm", action="store_true", default=False,
                   help="Scale balance loss by gmm_weight during the transition ramp.")
    g.add_argument("--bal-ema-blend", type=float, default=0.5,
                   help="Balance loss blend α between EMA and batch (default: 0.5).")
    g.add_argument("--pi-entropy-strength", type=float, default=0.0,
                   help="λ_pi: KL(π || uniform) penalty weight (default: 0.0).")
    g.add_argument("--masked-recon", action="store_true", default=False,
                   help="Compute reconstruction loss only on non-zero input entries.")
    g.add_argument("--corr-strength", type=float, default=0.0,
                   help="λ_corr: reconstruction-space pairwise correlation loss weight.")
    g.add_argument("--latent-corr-strength", type=float, default=0.0,
                   help="λ_lcorr: latent-space correlation alignment loss weight "
                        "(recommended 0.1–1.0 with --normalize-input).")
    g.add_argument("--contrastive-strength", type=float, default=0.0,
                   help="λ_nce: size-invariant neighbour-contrastive latent alignment (default: 0.0=disabled).")
    g.add_argument("--contrastive-k", type=int, default=5, help="Positives per anchor (default: 5).")
    g.add_argument("--contrastive-tau", type=float, default=0.1, help="Contrastive temperature (default: 0.1).")
    g.add_argument("--contrastive-min-corr", type=float, default=0.3,
                   help="Minimum Pearson r for a positive pair (default: 0.3).")
    # Hierarchical / isoform options
    g.add_argument("--hier-strength", type=float, default=0.0,
                   help="λ_hier: hierarchical isoform loss weight (default: 0.0=disabled).")
    g.add_argument("--hier-margin", type=float, default=1.0,
                   help="Hinge margin (latent units) below which isoform pairs are not pulled "
                        "together (default: 1.0).")
    g.add_argument("--no-hier-corr-weight", action="store_false", dest="hier_corr_weight",
                   default=True,
                   help="Disable weighting isoform pairs by their positive expression correlation.")
    g.add_argument("--tx2gene", type=str, default=None,
                   help="TSV/CSV with (transcript_id, gene_id); header optional, Ensembl version "
                        "suffixes are stripped on both sides.")
    g.add_argument("--keep-tx-version", action="store_true", default=False,
                   help="Do not strip version suffixes when matching tx2gene to feature IDs.")
    g.add_argument("--isoform-stratified", action="store_true", default=False,
                   help="Build batches gene-by-gene so all isoforms of a gene co-occur "
                        "(GeneGroupedBatchSampler). Requires --tx2gene.")
    g.add_argument("--p-multi", type=float, default=0.5,
                   help="Probability that a batch is filled from complete gene groups "
                        "(default: 0.5). Only active with --isoform-stratified.")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def model_kwargs(args, n_features: int) -> Dict:
    return dict(
        n_features=n_features,
        n_latent=args.latent_dim,
        n_modules=args.n_modules,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        sigma_min=args.sigma_min,
        normalize_input=args.normalize_input,
        diag_covariance=getattr(args, "diag_covariance", False),
    )


def build_model(args, n_features: int) -> GMMModuleVAE:
    return GMMModuleVAE(**model_kwargs(args, n_features))


def gmm_loss_kwargs(args) -> Dict:
    return dict(
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
        sep_strength=args.sep_strength,
        sep_alpha=args.sep_alpha,
        bal_strength=args.bal_strength,
        bal_type=args.bal_type,
        bal_scale_gmm=args.bal_scale_gmm,
        pi_entropy_strength=args.pi_entropy_strength,
        bal_ema_blend=args.bal_ema_blend,
        hier_strength=args.hier_strength,
        hier_margin=args.hier_margin,
        hier_corr_weight=args.hier_corr_weight,
        normalize_input=args.normalize_input,
        masked_recon=args.masked_recon,
        corr_strength=args.corr_strength,
        latent_corr_strength=args.latent_corr_strength,
        kl_per_element=args.kl_per_element,
        contrastive_strength=getattr(args, "contrastive_strength", 0.0),
        contrastive_k=getattr(args, "contrastive_k", 5),
        contrastive_tau=getattr(args, "contrastive_tau", 0.1),
        contrastive_min_corr=getattr(args, "contrastive_min_corr", 0.3),
    )


def build_losses(args):
    gmm_loss = GMMVAELoss(**gmm_loss_kwargs(args))
    warmup_loss = WarmupLoss(
        beta=args.beta,
        kl_warmup_epochs=args.warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
        normalize_input=args.normalize_input,
        masked_recon=args.masked_recon,
        kl_per_element=args.kl_per_element,
    )
    return gmm_loss, warmup_loss


def build_gene_groups(args, feature_ids: Sequence[str], logger: logging.Logger) -> Dict[str, List[int]]:
    """Load tx2gene and group dataset indices by gene (validated)."""
    needs = getattr(args, "hier_strength", 0.0) > 0 or getattr(args, "isoform_stratified", False)
    if not (args.tx2gene and needs):
        return {}
    import bsvae.utils.hierarchy as hier_mod

    tx2gene = hier_mod.load_tx2gene(
        path=args.tx2gene, feature_ids=feature_ids,
        keep_version=getattr(args, "keep_tx_version", False), logger=logger,
    )
    if getattr(args, "keep_tx_version", False):
        gene_groups = hier_mod.group_isoforms_by_gene(tx2gene, feature_ids, keep_version=True)
    else:
        gene_groups = hier_mod.group_isoforms_by_gene(tx2gene, feature_ids)
    logger.info("Hierarchical groups: %d multi-isoform genes", len(gene_groups))

    n_total = len(feature_ids)
    for gene_id, idxs in gene_groups.items():
        bad = [i for i in idxs if i < 0 or i >= n_total]
        if bad:
            raise ValueError(
                f"gene_groups['{gene_id}'] contains out-of-bounds indices {bad} "
                f"(dataset has {n_total} features)"
            )
    return gene_groups


def build_train_loader(
    dataset: OmicsDataset,
    args,
    gene_groups: Dict[str, List[int]],
    logger: logging.Logger,
) -> DataLoader:
    """Training DataLoader; gene-grouped batches when --isoform-stratified."""
    pin_memory = torch.cuda.is_available() and not getattr(args, "no_cuda", False)
    num_workers = int(getattr(args, "num_workers", 0) or 0)
    if getattr(args, "isoform_stratified", False) and gene_groups:
        from bsvae.utils.hierarchy import GeneGroupedBatchSampler
        sampler = GeneGroupedBatchSampler(
            n_features=len(dataset),
            gene_groups=gene_groups,
            batch_size=args.batch_size,
            p_multi=args.p_multi,
            drop_last=True,
            seed=args.seed,
        )
        logger.info(
            "GeneGroupedBatchSampler: %d multi-isoform genes, p_multi=%.2f",
            len(sampler.groups), args.p_multi,
        )
        return DataLoader(
            dataset, batch_sampler=sampler, pin_memory=pin_memory, num_workers=num_workers,
        )
    if getattr(args, "isoform_stratified", False) and not gene_groups:
        logger.warning(
            "--isoform-stratified requested but no multi-isoform gene groups found; "
            "using standard random sampler."
        )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=pin_memory, num_workers=num_workers,
    )


def build_trainer(
    args, model, optimizer, gmm_loss, warmup_loss, device, logger, save_dir, gene_groups,
) -> Trainer:
    return Trainer(
        model=model,
        optimizer=optimizer,
        gmm_loss_f=gmm_loss,
        warmup_loss_f=warmup_loss,
        device=device,
        logger=logger,
        save_dir=save_dir,
        is_progress_bar=not getattr(args, "no_progress_bar", False),
        warmup_epochs=args.warmup_epochs,
        transition_epochs=args.transition_epochs,
        freeze_gmm_epochs=args.freeze_gmm_epochs,
        gene_groups=gene_groups,
        collapse_min_usage=args.collapse_min_usage,
        collapse_noise_scale=args.collapse_noise_scale,
        seed=args.seed,
        collapse_threshold=getattr(args, "collapse_threshold", None),
        amp=getattr(args, "amp", False),
    )


def training_metadata(args, n_features: int) -> Dict:
    """Serialisable copy of the CLI namespace plus the model input size."""
    meta = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    meta["n_features"] = n_features
    return meta
