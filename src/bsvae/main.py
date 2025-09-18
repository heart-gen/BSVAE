#!/usr/bin/env python

import sys
import logging
import argparse
from os.path import join, dirname
from configparser import ConfigParser

import torch
from torch import optim

from bsvae.models import StructuredFactorVAE
from bsvae.models.losses import get_loss_f
from bsvae.utils.datasets import get_dataloaders
from bsvae.utils.helpers import (
    set_seed,
    get_device,
    get_n_param,
    create_safe_directory,
    FormatterNoDuplicate,
    get_config_section,
    update_namespace_,
)
from bsvae.utils.modelIO import save_model, load_model, load_metadata
from bsvae.trainer import Trainer
from bsvae.evaluator import Evaluator

RES_DIR = "results"

def parse_arguments(cli_args):
    """Parse CLI and config file arguments."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=str, default=None,
                            help="Path to config file (hyperparam.ini).")
    config_args, _ = pre_parser.parse_known_args(cli_args)

    # Load defaults from config
    config_path = config_args.config if config_args.config else join(dirname(__file__), "hyperparam.ini")
    default_config = get_config_section([config_path], "Custom")

    parser = argparse.ArgumentParser(
        description="Training and evaluation for StructuredFactorVAE.",
        formatter_class=FormatterNoDuplicate,
        parents=[pre_parser],
    )

    # General
    parser.add_argument("name", type=str, help="Name for storing/loading the model.")
    parser.add_argument("-L", "--log-level", default=default_config["log_level"],
                        choices=list(logging._levelToName.values()))
    parser.add_argument("--no-cuda", action="store_true", default=default_config["no_cuda"])
    parser.add_argument("--seed", type=int, default=default_config["seed"])

    # Training
    parser.add_argument("--checkpoint-every", type=int, default=default_config["checkpoint_every"])
    parser.add_argument("--epochs", type=int, default=default_config["epochs"])
    parser.add_argument("--batch-size", type=int, default=default_config["batch_size"])
    parser.add_argument("--lr", type=float, default=default_config["lr"])
    parser.add_argument("--gene-expression-filename", default=None,
                        help="CSV with gene expression (genes x samples).")
    parser.add_argument("--gene-expression-dir", default=None,
                        help="Directory with train/test splits (X_train.csv, X_test.csv).")

    # Model
    parser.add_argument("--latent-dim", type=int, default=default_config["latent_dim"])
    parser.add_argument("--hidden-dims", type=lambda x: eval(x) if isinstance(x, str) else x,
                        default=default_config.get("hidden_dims", [512, 256]))
    parser.add_argument("--dropout", type=float, default=default_config.get("dropout", 0.1))
    parser.add_argument("--init-sd", type=float, default=default_config.get("init_sd", 0.02))
    parser.add_argument("--learn-var", action="store_true",
                        default=default_config.get("learn_var", False))

    # Loss
    parser.add_argument("--loss", default=default_config["loss"], choices=["VAE", "beta"])
    parser.add_argument("--beta", type=float, default=default_config.get("beta", 1.0))
    parser.add_argument("--l1-strength", type=float, default=default_config.get("l1_strength", 1e-3))
    parser.add_argument("--lap-strength", type=float, default=default_config.get("lap_strength", 1e-4))

    # Evaluation
    parser.add_argument("--is-eval-only", action="store_true", default=default_config["is_eval_only"])
    parser.add_argument("--no-test", action="store_true", default=default_config["no_test"])
    parser.add_argument("--eval-batchsize", type=int, default=default_config["eval_batchsize"])

    args = parser.parse_args(cli_args)

    # Validate dataset input
    if bool(args.gene_expression_filename) == bool(args.gene_expression_dir):
        parser.error("Specify exactly one of --gene-expression-filename or --gene-expression-dir.")

    parser.set_defaults(**vars(args))
    return parser.parse_args(cli_args)


def setup_logging(log_level):
    log_fmt = "%(asctime)s %(levelname)s - %(funcName)s: %(message)s"
    formatter = logging.Formatter(log_fmt, "%H:%M:%S")
    logger = logging.getLogger()
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(log_level.upper())
    return logger


def main(args):
    logger = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(is_gpu=not args.no_cuda)
    exp_dir = join(RES_DIR, args.name)
    logger.info(f"Experiment directory: {exp_dir}")

    dataset_kwargs = {
        "gene_expression_filename": args.gene_expression_filename,
        "gene_expression_dir": args.gene_expression_dir,
    }

    # Training
    if not args.is_eval_only:
        create_safe_directory(exp_dir, logger=logger)

        train_loader = get_dataloaders(
            dataset="genenet",  # renamed dataset
            batch_size=args.batch_size,
            logger=logger,
            train=True,
            drop_last=True,
            **{k: v for k, v in dataset_kwargs.items() if v is not None},
        )
        logger.info(f"Train samples: {len(train_loader.dataset)}")

        model = StructuredFactorVAE(
            n_genes=train_loader.dataset.n_genes,
            n_latent=args.latent_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            init_sd=args.init_sd,
            learn_var=args.learn_var,
        )
        logger.info(f"Num parameters: {get_n_param(model)}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)

        loss_f = get_loss_f(args.loss, beta=args.beta,
                            l1_strength=args.l1_strength,
                            lap_strength=args.lap_strength)

        trainer = Trainer(
            model, optimizer, loss_f,
            device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_cuda,
        )
        trainer(train_loader, epochs=args.epochs,
                checkpoint_every=args.checkpoint_every)
        save_model(trainer.model, exp_dir, metadata=vars(args))

    # Evaluation
    if not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        test_loader = get_dataloaders(
            dataset="genenet",
            batch_size=args.eval_batchsize,
            shuffle=False,
            logger=logger,
            train=False,
            drop_last=True,
            **{k: v for k, v in dataset_kwargs.items() if v is not None},
        )
        loss_f = get_loss_f(args.loss, beta=args.beta,
                            l1_strength=args.l1_strength,
                            lap_strength=args.lap_strength)
        evaluator = Evaluator(
            model, loss_f, device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_cuda,
        )
        evaluator(test_loader)


def cli():
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    cli()
