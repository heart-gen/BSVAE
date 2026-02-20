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
try:
    from bsvae.utils.datasets import get_dataloaders
except ImportError:  # pragma: no cover
    get_dataloaders = None
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
try:
    from bsvae.utils.ppi import load_ppi_laplacian
except ImportError:  # pragma: no cover
    load_ppi_laplacian = None
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
    """Legacy parser: CLI args with defaults pulled from deprecated hyperparam.ini."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", "-c", type=str,
                            default=join(dirname(__file__), "hyperparam.ini"),
                            help="Path to legacy hyperparam.ini (deprecated).")
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
                        default=config.get("latent_dim", 30))
    parser.add_argument("--hidden-dims", "-Z", type=ast_literal_eval,
                        default=config.get("hidden_dims", [256, 128]))
    parser.add_argument("--dropout", type=float,
                        default=config.get("dropout", 0.1))
    parser.add_argument("--learn-var", action="store_true",
                        dest="learn_var",
                        default=config.get("learn_var", True),
                        help="Enable per-gene heteroscedastic decoder variance (default: True).")
    parser.add_argument("--no-learn-var", action="store_false",
                        dest="learn_var",
                        help="Disable per-gene heteroscedastic decoder variance.")
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
    parser.add_argument("--coexpr-strength", type=float,
                        default=config.get("coexpr_strength", 1.0),
                        help="Weight for co-expression preservation loss (default: 1.0).")
    parser.add_argument("--coexpr-warmup-epochs", type=int,
                        default=config.get("coexpr_warmup_epochs", 50),
                        help="Warmup epochs to ramp coexpression loss from 0 to full weight (default: 50).")
    parser.add_argument("--coexpr-gamma", type=float,
                        default=config.get("coexpr_gamma", 6.0),
                        help="Soft-thresholding power for coexpression (default: 6.0).")
    parser.add_argument("--coexpr-block-size", type=int,
                        default=config.get("coexpr_block_size", 512),
                        help="Block size for correlation computation to limit memory (default: 512).")
    parser.add_argument("--coexpr-max-genes", type=int,
                        default=config.get("coexpr_max_genes", None),
                        help="Optional cap on number of genes used for coexpression (default: None).")
    parser.add_argument("--coexpr-auto-scale", action="store_true",
                        dest="coexpr_auto_scale",
                        default=config.get("coexpr_auto_scale", False),
                        help="Enable EMA-based auto scaling for coexpression loss (default: False).")
    parser.add_argument("--no-coexpr-auto-scale", action="store_false",
                        dest="coexpr_auto_scale",
                        help="Disable EMA-based auto scaling for coexpression loss.")
    parser.add_argument("--coexpr-ema-decay", type=float,
                        default=config.get("coexpr_ema_decay", 0.99),
                        help="EMA decay for coexpression auto scaling (default: 0.99).")
    parser.add_argument("--coexpr-scale-cap", type=float,
                        default=config.get("coexpr_scale_cap", 10.0),
                        help="Max multiplier for coexpression auto scaling (default: 10.0).")

    # KL annealing and anti-collapse
    parser.add_argument("--kl-warmup-epochs", type=int,
                        default=config.get("kl_warmup_epochs", 0),
                        help="Epochs to linearly ramp beta from 0 to target (default: 0).")
    parser.add_argument("--kl-anneal-mode", type=str,
                        default=config.get("kl_anneal_mode", "linear"),
                        choices=["linear", "cyclical"],
                        help="KL annealing schedule (default: linear).")
    parser.add_argument("--kl-cycle-length", type=int,
                        default=config.get("kl_cycle_length", 50),
                        help="Cycle length for cyclical annealing (default: 50).")
    parser.add_argument("--kl-n-cycles", type=int,
                        default=config.get("kl_n_cycles", 4),
                        help="Number of cycles for cyclical annealing (default: 4).")
    parser.add_argument("--free-bits", type=float,
                        default=config.get("free_bits", 0.0),
                        help="Minimum KL per latent dimension in nats (default: 0.0).")

    # Encoder options
    parser.add_argument("--use-batch-norm", action="store_true",
                        dest="use_batch_norm",
                        default=config.get("use_batch_norm", True),
                        help="Use BatchNorm1d in encoder (default: True).")
    parser.add_argument("--no-batch-norm", action="store_false",
                        dest="use_batch_norm",
                        help="Disable BatchNorm1d in encoder.")

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
    if get_dataloaders is None:
        raise ImportError("bsvae.utils.datasets dependencies are missing; install optional training dependencies.")

    logger = setup_logging(getattr(args, "log_level", "info"))
    set_seed(args.seed)
    device = get_device(use_gpu=not args.no_cuda)
    exp_dir = join(getattr(args, "outdir", "results"), args.name)
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
        if load_ppi_laplacian is None:
            logger.warning("PPI dependencies unavailable; skipping Laplacian prior loading")
            L = None
        else:
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
            L=L,
            use_batch_norm=getattr(args, "use_batch_norm", True)
        ).to(device)
        logger.info(f"Model initialized with {n_genes} genes, {args.latent_dim} latent dims")

        # Loss + optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        loss_f = BaseLoss(beta=args.beta,
                          l1_strength=args.l1_strength,
                          lap_strength=args.lap_strength,
                          kl_warmup_epochs=getattr(args, "kl_warmup_epochs", 0),
                          kl_anneal_mode=getattr(args, "kl_anneal_mode", "linear"),
                          kl_cycle_length=getattr(args, "kl_cycle_length", 50),
                          kl_n_cycles=getattr(args, "kl_n_cycles", 4),
                          free_bits=getattr(args, "free_bits", 0.0),
                          coexpr_strength=getattr(args, "coexpr_strength", 1.0),
                          coexpr_warmup_epochs=getattr(args, "coexpr_warmup_epochs", 50),
                          coexpr_gamma=getattr(args, "coexpr_gamma", 6.0),
                          coexpr_block_size=getattr(args, "coexpr_block_size", 512),
                          coexpr_max_genes=getattr(args, "coexpr_max_genes", None),
                          coexpr_auto_scale=getattr(args, "coexpr_auto_scale", False),
                          coexpr_ema_decay=getattr(args, "coexpr_ema_decay", 0.99),
                          coexpr_scale_cap=getattr(args, "coexpr_scale_cap", 10.0))

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
                          lap_strength=args.lap_strength,
                          kl_warmup_epochs=getattr(args, "kl_warmup_epochs", 0),
                          kl_anneal_mode=getattr(args, "kl_anneal_mode", "linear"),
                          kl_cycle_length=getattr(args, "kl_cycle_length", 50),
                          kl_n_cycles=getattr(args, "kl_n_cycles", 4),
                          free_bits=getattr(args, "free_bits", 0.0),
                          coexpr_strength=getattr(args, "coexpr_strength", 1.0),
                          coexpr_warmup_epochs=getattr(args, "coexpr_warmup_epochs", 50),
                          coexpr_gamma=getattr(args, "coexpr_gamma", 6.0),
                          coexpr_block_size=getattr(args, "coexpr_block_size", 512),
                          coexpr_max_genes=getattr(args, "coexpr_max_genes", None),
                          coexpr_auto_scale=getattr(args, "coexpr_auto_scale", False),
                          coexpr_ema_decay=getattr(args, "coexpr_ema_decay", 0.99),
                          coexpr_scale_cap=getattr(args, "coexpr_scale_cap", 10.0))
        evaluator = Evaluator(
            model, loss_f, device=device, logger=logger, save_dir=exp_dir,
            is_progress_bar=not args.no_cuda,
        )
        raw_eval_epoch = metadata.get("epochs", getattr(args, "epochs", 0))
        try:
            eval_epoch = max(int(raw_eval_epoch) - 1, 0)
        except (TypeError, ValueError):
            eval_epoch = 0
        evaluator(test_loader, epoch=eval_epoch)


def cli():
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    cli()
