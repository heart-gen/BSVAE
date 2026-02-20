"""
bsvae-train — Train a GMMModuleVAE.

Usage
-----
bsvae-train NAME --dataset PATH [options]
"""

from __future__ import annotations

import ast
import logging
import sys
from os.path import join

import torch
from torch import optim

from bsvae.models import GMMModuleVAE
from bsvae.models.losses import GMMVAELoss, WarmupLoss
from bsvae.utils.datasets import get_omics_dataloader
from bsvae.utils.helpers import (
    FormatterNoDuplicate,
    create_safe_directory,
    get_device,
    get_n_params,
    set_seed,
)
from bsvae.utils.modelIO import save_model, load_model, load_metadata
from bsvae.utils.training import Trainer, Evaluator


def _ast_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def parse_args(cli_args=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="bsvae-train",
        description="Train a GMMModuleVAE for biological module discovery.",
        formatter_class=FormatterNoDuplicate,
    )

    # --- Required ---
    p.add_argument("name", type=str, help="Experiment name (output subdirectory).")
    p.add_argument("--dataset", required=True, help="Expression matrix (features × samples). CSV/TSV/HDF5/h5ad.")

    # --- General ---
    p.add_argument("--outdir", type=str, default="results", help="Directory for outputs.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    p.add_argument("--no-progress-bar", action="store_true", default=False)

    # --- Training ---
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=20,
                   help="Phase 1 length: standard N(0,I) VAE (default: 20).")
    p.add_argument("--transition-epochs", type=int, default=10,
                   help="GMM loss ramp-in length (default: 10).")

    # --- Model architecture ---
    p.add_argument("--n-modules", "-K", type=int, default=20,
                   help="Number of GMM components / biological modules (default: 20).")
    p.add_argument("--latent-dim", "-z", type=int, default=32,
                   help="Latent space dimensionality D (default: 32).")
    p.add_argument("--hidden-dims", type=_ast_eval, default=[512, 256, 128],
                   help="Encoder hidden layer sizes as Python list (default: [512,256,128]).")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-batch-norm", action="store_false", dest="use_batch_norm")
    p.add_argument("--use-batch-norm", action="store_true", dest="use_batch_norm", default=True)
    p.add_argument("--sigma-min", type=float, default=0.3,
                   help="GMM component σ floor (default: 0.3).")

    # --- Loss hyperparameters ---
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (default: 1.0).")
    p.add_argument("--free-bits", type=float, default=0.5,
                   help="Per-dim KL lower bound in nats (default: 0.5).")
    p.add_argument("--kl-warmup-epochs", type=int, default=0,
                   help="Epochs for β linear ramp within Phase 2 (default: 0).")
    p.add_argument("--kl-anneal-mode", choices=["linear", "cyclical"], default="linear")
    p.add_argument("--kl-cycle-length", type=int, default=50)
    p.add_argument("--sep-strength", type=float, default=0.1,
                   help="λ_sep: σ-scaled separation loss weight (default: 0.1).")
    p.add_argument("--sep-alpha", type=float, default=2.0,
                   help="α: margin multiplier for separation loss (default: 2.0).")
    p.add_argument("--bal-strength", type=float, default=0.01,
                   help="λ_bal: γ-usage balance loss weight (default: 0.01).")
    p.add_argument("--hier-strength", type=float, default=0.0,
                   help="λ_hier: hierarchical isoform loss weight (default: 0.0=disabled).")
    p.add_argument("--tx2gene", type=str, default=None,
                   help="TSV with (transcript_id, gene_id) for hierarchical loss.")

    # --- Evaluation ---
    p.add_argument("--no-eval", action="store_true", default=False,
                   help="Skip evaluation pass after training.")
    p.add_argument("--eval-batch-size", type=int, default=None)

    return p.parse_args(cli_args)


def setup_logging(level: str) -> logging.Logger:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def main(args):
    logger = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(use_gpu=not args.no_cuda)
    exp_dir = join(args.outdir, args.name)

    create_safe_directory(exp_dir, logger=logger)

    # --- Data ---
    train_loader = get_omics_dataloader(
        path=args.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        logger=logger,
    )
    n_features = train_loader.dataset.n_samples   # profile length = number of samples
    feature_ids = train_loader.dataset.feature_ids
    logger.info("Dataset: %d features × %d samples", len(feature_ids), n_features)

    # --- Hierarchy ---
    gene_groups = {}
    if args.hier_strength > 0 and args.tx2gene:
        from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene
        tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=feature_ids)
        gene_groups = group_isoforms_by_gene(tx2gene, feature_ids)
        logger.info("Hierarchical groups: %d multi-isoform genes", len(gene_groups))

    # --- Model ---
    model = GMMModuleVAE(
        n_features=n_features,
        n_latent=args.latent_dim,
        n_modules=args.n_modules,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        sigma_min=args.sigma_min,
    ).to(device)
    logger.info(
        "Model: %d features, %d latent dims, %d modules | params=%d",
        n_features, args.latent_dim, args.n_modules, get_n_params(model),
    )

    # --- Loss functions ---
    gmm_loss = GMMVAELoss(
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
        sep_strength=args.sep_strength,
        sep_alpha=args.sep_alpha,
        bal_strength=args.bal_strength,
        hier_strength=args.hier_strength,
    )
    warmup_loss = WarmupLoss(
        beta=args.beta,
        kl_warmup_epochs=args.warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Train ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        gmm_loss_f=gmm_loss,
        warmup_loss_f=warmup_loss,
        device=device,
        logger=logger,
        save_dir=exp_dir,
        is_progress_bar=not args.no_progress_bar,
        warmup_epochs=args.warmup_epochs,
        transition_epochs=args.transition_epochs,
        gene_groups=gene_groups,
    )
    trainer(train_loader, epochs=args.epochs, checkpoint_every=args.checkpoint_every)

    # --- Save ---
    metadata = vars(args)
    metadata["n_features"] = n_features
    save_model(trainer.model, exp_dir, metadata=metadata)
    logger.info("Model saved to %s", exp_dir)

    # --- Eval ---
    if not args.no_eval:
        model_eval = load_model(exp_dir, is_gpu=not args.no_cuda)
        eval_bs = args.eval_batch_size or (args.batch_size // 2)
        eval_loader = get_omics_dataloader(
            path=args.dataset,
            batch_size=eval_bs,
            shuffle=False,
            drop_last=False,
            logger=logger,
        )
        evaluator = Evaluator(
            model_eval, gmm_loss, device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_progress_bar,
        )
        evaluator(eval_loader)


def cli():
    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    cli()
