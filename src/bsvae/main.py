#!/usr/bin/env python

import sys
import logging
import argparse
from os.path import join, dirname
from configparser import ConfigParser

import torch
from torch import optim

from bsvae.models import StructuredFactorVAE
from bsvae.models.losses import BaseLoss
from bsvae.utils.datasets import get_dataloaders
from bsvae.utils.helpers import (
    set_seed,
    get_device,
    get_n_params,
    create_safe_directory,
    FormatterNoDuplicate,
    get_config_section,
    update_namespace_,
)
from bsvae.utils import Trainer, Evaluator
from bsvae.utils.ppi import load_ppi_laplacian
from bsvae.utils.modelIO import save_model, load_model, load_metadata

def load_config(config_path: str, section: str = "Custom") -> dict:
    """Load hyperparameters from .ini file."""
    parser = ConfigParser()
    parser.read(config_path)
    if section not in parser:
        raise ValueError(f"Config section [{section}] not found in {config_path}")
    return {k: ast_literal_eval(v) for k, v in parser[section].items()}


def ast_literal_eval(val):
    """Helper to parse lists, ints, floats from strings safely."""
    import ast
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def parse_arguments(cli_args):
    """Parse CLI args, with defaults pulled from hyperparam.ini."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=str,
                            default=join(dirname(__file__), "hyperparam.ini"),
                            help="Path to hyperparam.ini")
    pre_parser.add_argument("--section", type=str, default="Custom",
                            help="Section of .ini to load")
    config_args, _ = pre_parser.parse_known_args(cli_args)

    # Load defaults from config
    config = load_config(config_args.config, config_args.section)

    parser = argparse.ArgumentParser(
        description="Training and evaluation for StructuredFactorVAE.",
        formatter_class=FormatterNoDuplicate,
        parents=[pre_parser],
    )

    # General
    parser.add_argument("name", type=str, help="Experiment name.")
    parser.add_argument("--outdir", type=str,
                        default=config.get("outdir", "results"),
                        help="Directory for experiment outputs (default: results).")
    parser.add_argument("--seed", type=int, default=config.get("seed", 13))
    parser.add_argument("--no-cuda", action="store_true",
                        default=config.get("no_cuda", False))
    parser.add_argument("--log-level", type=str, default=config.get("log_level", "info"),
                        choices=["debug", "info", "warning", "error", "critical"],
                        help="Logging verbosity (default: info)")

    # Training
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 100))
    parser.add_argument("--batch-size", type=int,
                        default=config.get("batch_size", 64))
    parser.add_argument("--lr", type=float, default=config.get("lr", 5e-4))
    parser.add_argument("--checkpoint-every", type=int,
                        default=config.get("checkpoint_every", 10))

    # Model
    parser.add_argument("--latent-dim", "-z", type=int,
                        default=config.get("latent_dim", 10))
    parser.add_argument("--hidden-dims", "-Z", type=ast_literal_eval,
                        default=config.get("hidden_dims", [128, 64]))
    parser.add_argument("--dropout", type=float,
                        default=config.get("dropout", 0.1))
    parser.add_argument("--learn-var", action="store_true",
                        default=config.get("learn_var", False))
    parser.add_argument("--init-sd", type=float,
                        default=config.get("init_sd", 0.02))

    # Loss
    parser.add_argument("--loss", type=str, default=config.get("loss", "VAE"),
                        choices=["VAE", "beta"])
    parser.add_argument("--beta", type=float, default=config.get("beta", 1.0))
    parser.add_argument("--l1-strength", type=float,
                        default=config.get("l1_strength", 1e-3))
    parser.add_argument("--lap-strength", type=float,
                        default=config.get("lap_strength", 1e-4))

    # Evaluation
    parser.add_argument("--is-eval-only", action="store_true",
                        default=config.get("is_eval_only"))
    parser.add_argument("--no-test", action="store_true",
                        default=config.get("no_test"))
    parser.add_argument("--eval-batchsize", type=int,
                        default=config.get("eval_batchsize"))

    # Dataset
    parser.add_argument("--dataset", type=str,
                        default=config.get("dataset", "genenet"))
    parser.add_argument("--gene-expression-filename", type=str,
                        default=config.get("gene_expression_filename", None),
                        help="CSV with gene expression (genes x samples).")
    parser.add_argument("--gene-expression-dir", type=str,
                        default=config.get("gene_expression_dir", None),
                        help="Directory with train/test splits (X_train.csv, X_test.csv).")

    # PPI Priors
    parser.add_argument("--ppi-taxid", type=str, default=config.get("ppi_taxid", "9606"))
    parser.add_argument("--ppi-cache", type=str, default=config.get("ppi_cache", None))

    args = parser.parse_args(cli_args)

    # Validate dataset input
    if bool(args.gene_expression_filename) == bool(args.gene_expression_dir):
        parser.error("Specify exactly one of --gene-expression-filename or --gene-expression-dir.")

    parser.set_defaults(**vars(args))
    return parser.parse_args(cli_args)


def setup_logging(level: str = "info"):
    """Configure logging verbosity."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log_fmt = "%(asctime)s %(levelname)s: %(message)s"
    logging.basicConfig(level=numeric_level, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.setLevel(numeric_level)
    return logger


def main(args):
    logger = setup_logging(args.log_level)
    set_seed(args.seed)
    device = get_device(use_gpu=not args.no_cuda)
    exp_dir = join(args.outdir, args.name)
    logger.info(f"Experiment directory: {exp_dir}")

    # Training
    if not args.is_eval_only:
        create_safe_directory(exp_dir, logger=logger)

        # Data
        train_loader = get_dataloaders(
            dataset=args.dataset,
            batch_size=args.batch_size,
            logger=logger,
            train=True,
            drop_last=True,
            gene_expression_filename=args.gene_expression_filename,
            gene_expression_dir=args.gene_expression_dir,
        )
        n_genes = train_loader.dataset[0][0].shape[-1]
        logger.info(f"Training dataset size: {len(train_loader.dataset)}")

        # PPI Laplacian
        try:
            gene_list = getattr(train_loader.dataset, "genes", None)
            if gene_list is not None:
                L, G = load_ppi_laplacian(
                    gene_list,
                    taxid=args.ppi_taxid,
                    min_score=700,
                    cache_dir=args.ppi_cache or "~/.bsvae/ppi",
                )
                logger.info(f"PPI Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            else:
                L = None
        except Exception as e:
            logger.warning(f"Could not load PPI Laplacian: {e}")
            L = None

        # Model
        model = StructuredFactorVAE(
            n_genes=n_genes,
            n_latent=args.latent_dim,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
            init_sd=args.init_sd,
            learn_var=args.learn_var,
            L=L
        ).to(device)
        logger.info(f"Model initialized with {n_genes} genes, {args.latent_dim} latent dims")

        # Loss + optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        loss_f = BaseLoss(beta=args.beta,
                          l1_strength=args.l1_strength,
                          lap_strength=args.lap_strength)

        trainer = Trainer(
            model, optimizer, loss_f, device=device, logger=logger,
            save_dir=exp_dir, is_progress_bar=not args.no_cuda,
        )
        trainer(train_loader, epochs=int(args.epochs),
                checkpoint_every=int(args.checkpoint_every))
        # Persist the input dimension so evaluation can verify compatibility.
        args.n_genes = n_genes
        save_model(trainer.model, exp_dir, metadata=vars(args))

    # Evaluation
    if not args.no_test:
        model = load_model(exp_dir, is_gpu=not args.no_cuda)
        metadata = load_metadata(exp_dir)
        eval_batch_size = args.eval_batchsize or ( args.batch_size // 2 )
        test_loader = get_dataloaders(
            dataset="genenet",
            batch_size=eval_batch_size,
            shuffle=False,
            logger=logger,
            train=False,
            drop_last=False,
            gene_expression_filename=args.gene_expression_filename,
            gene_expression_dir=args.gene_expression_dir,
        )

        # Validate that evaluation data matches the trained model input size
        test_n_genes = test_loader.dataset[0][0].shape[-1]
        expected_genes = metadata.get("n_genes")
        if expected_genes is not None and test_n_genes != expected_genes:
            raise ValueError(
                "Gene dimension mismatch between evaluation data and trained model: "
                f"data has {test_n_genes} genes but model expects {expected_genes}. "
                "Please use evaluation data generated with the same gene set used for training "
                "or point --gene-expression-... to the matching files."
            )
        loss_f = BaseLoss(beta=args.beta,
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
