"""
bsvae-train — Train a GMMModuleVAE.

Usage
-----
bsvae-train NAME --dataset PATH [options]
"""

from __future__ import annotations

import sys
from os.path import join

from torch import optim

from bsvae.cli._common import (
    add_general_args,
    add_loss_args,
    add_model_args,
    add_schedule_args,
    build_gene_groups,
    build_losses,
    build_model,
    build_train_loader,
    build_trainer,
    setup_logging,
    training_metadata,
)
from bsvae.utils.datasets import OmicsDataset, get_omics_dataloader
from bsvae.utils.helpers import FormatterNoDuplicate, create_safe_directory, get_device, get_n_params, set_seed
from bsvae.utils.modelIO import save_model, load_model
from bsvae.utils.training import Evaluator


def parse_args(cli_args=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="bsvae-train",
        description="Train a GMMModuleVAE for biological module discovery.",
        formatter_class=FormatterNoDuplicate,
    )
    p.add_argument("name", type=str, help="Experiment name (output subdirectory).")
    p.add_argument("--dataset", required=True, help="Expression matrix (features × samples). CSV/TSV/HDF5/h5ad.")
    p.add_argument("--epochs", type=int, default=100)
    add_general_args(p)
    add_schedule_args(p, checkpoint_default=10)
    add_model_args(p)
    add_loss_args(p)
    p.add_argument("--no-eval", action="store_true", default=False,
                   help="Skip the training-set evaluation pass after training.")
    return p.parse_args(cli_args)


def main(args):
    logger = setup_logging(args.log_level, name=__name__)
    set_seed(args.seed)
    device = get_device(use_gpu=not args.no_cuda)
    exp_dir = join(args.outdir, args.name)
    create_safe_directory(exp_dir, logger=logger)

    # --- Data ---
    dataset = OmicsDataset(args.dataset, logger=logger, return_index=True)
    n_features = dataset.n_samples   # profile length = number of samples
    feature_ids = dataset.feature_ids
    logger.info("Dataset: %d features × %d samples", len(feature_ids), n_features)

    gene_groups = build_gene_groups(args, feature_ids, logger)
    train_loader = build_train_loader(dataset, args, gene_groups, logger)

    # --- Model / losses / trainer ---
    model = build_model(args, n_features).to(device)
    logger.info(
        "Model: %d features, %d latent dims, %d modules | params=%d",
        n_features, args.latent_dim, args.n_modules, get_n_params(model),
    )
    gmm_loss, warmup_loss = build_losses(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = build_trainer(
        args, model, optimizer, gmm_loss, warmup_loss, device, logger, exp_dir, gene_groups,
    )
    trainer(train_loader, epochs=args.epochs, checkpoint_every=args.checkpoint_every)

    # --- Save ---
    save_model(trainer.model, exp_dir, metadata=training_metadata(args, n_features))
    logger.info("Model saved to %s", exp_dir)

    # --- Training-set evaluation (no held-out split in bsvae-train) ---
    if not args.no_eval:
        model_eval = load_model(exp_dir, is_gpu=not args.no_cuda)
        eval_bs = args.eval_batch_size or max(1, args.batch_size // 2)
        eval_loader = get_omics_dataloader(
            path=args.dataset, batch_size=eval_bs, shuffle=False, drop_last=False,
            logger=logger, dataset=OmicsDataset.from_frame(_as_frame(dataset)),
        )
        evaluator = Evaluator(
            model_eval, gmm_loss, device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_progress_bar,
        )
        comps = evaluator(eval_loader, return_components=True)
        logger.info("Training-set evaluation (same features as training): %s", comps)


def _as_frame(dataset: OmicsDataset):
    import pandas as pd
    return pd.DataFrame(dataset.data.numpy(), index=dataset.feature_ids, columns=dataset.sample_ids)


def cli():
    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    cli()
