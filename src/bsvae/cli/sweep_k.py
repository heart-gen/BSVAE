"""
bsvae-sweep-k — Two-pass sweep to select number of modules (K).

Usage
-----
bsvae-sweep-k NAME --dataset PATH [options]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import numpy as np

from torch import optim

from bsvae.models import GMMModuleVAE
from bsvae.models.losses import GMMVAELoss, WarmupLoss
from bsvae.utils.datasets import OmicsDataset, get_omics_dataloader
from bsvae.utils.helpers import (
    FormatterNoDuplicate,
    create_safe_directory,
    get_device,
    get_n_params,
    set_seed,
)
from bsvae.utils.modelIO import save_model
from bsvae.utils.training import Trainer, Evaluator


@dataclass
class SweepResult:
    k: int
    val_loss_mean: float
    val_loss_std: float
    stability_ari_mean: float | None
    stability_ari_std: float | None
    reps: int
    out_dir: str


def _parse_k_list(text: str) -> List[int]:
    """Parse K list from '8,12,16' or '8:24:4' (inclusive end)."""
    text = (text or "").strip()
    if not text:
        return []
    if ":" in text:
        parts = [p.strip() for p in text.split(":")]
        if len(parts) != 3:
            raise ValueError("Range format must be start:end:step (e.g., 8:24:4)")
        start, end, step = (int(p) for p in parts)
        if step <= 0:
            raise ValueError("step must be > 0")
        if end < start:
            raise ValueError("end must be >= start")
        return list(range(start, end + 1, step))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _split_feature_ids(
    feature_ids: Sequence[str],
    val_frac: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1 (exclusive)")
    import random
    rng = random.Random(seed)
    ids = list(feature_ids)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = ids[:n_val]
    train_ids = ids[n_val:]
    return train_ids, val_ids


def _setup_logging(level: str) -> logging.Logger:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="%(asctime)s %(levelname)s: %(message)s")
    return logging.getLogger(__name__)


def _build_model(args, n_features: int) -> GMMModuleVAE:
    return GMMModuleVAE(
        n_features=n_features,
        n_latent=args.latent_dim,
        n_modules=args.n_modules,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        use_batch_norm=args.use_batch_norm,
        sigma_min=args.sigma_min,
    )


def _build_losses(args) -> tuple[GMMVAELoss, WarmupLoss]:
    gmm_loss = GMMVAELoss(
        beta=args.beta,
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
        sep_strength=args.sep_strength,
        sep_alpha=args.sep_alpha,
        bal_strength=args.bal_strength,
        pi_entropy_strength=args.pi_entropy_strength,
        bal_ema_blend=args.bal_ema_blend,
        hier_strength=args.hier_strength,
    )
    warmup_loss = WarmupLoss(
        beta=args.beta,
        kl_warmup_epochs=args.warmup_epochs,
        kl_anneal_mode=args.kl_anneal_mode,
        kl_cycle_length=args.kl_cycle_length,
        free_bits=args.free_bits,
    )
    return gmm_loss, warmup_loss


def _train_and_eval(
    args,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    out_dir: str,
    epochs: int,
    logger: logging.Logger,
    seed: int,
    compute_assignments: bool,
) -> tuple[float, list[int] | None]:
    create_safe_directory(out_dir, logger=logger)
    set_seed(seed)
    device = get_device(use_gpu=not args.no_cuda)

    train_loader = get_omics_dataloader(
        path=args.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        feature_subset=list(train_ids),
        logger=logger,
    )
    n_features = train_loader.dataset.n_samples
    feature_ids = train_loader.dataset.feature_ids

    gene_groups = {}
    if args.hier_strength > 0 and args.tx2gene:
        from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene
        tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=feature_ids)
        gene_groups = group_isoforms_by_gene(tx2gene, feature_ids)
        logger.info("Hierarchical groups: %d multi-isoform genes", len(gene_groups))

    model = _build_model(args, n_features).to(device)
    logger.info(
        "Model: %d features, %d latent dims, %d modules | params=%d",
        n_features, args.latent_dim, args.n_modules, get_n_params(model),
    )
    gmm_loss, warmup_loss = _build_losses(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        gmm_loss_f=gmm_loss,
        warmup_loss_f=warmup_loss,
        device=device,
        logger=logger,
        save_dir=out_dir,
        is_progress_bar=not args.no_progress_bar,
        warmup_epochs=args.warmup_epochs,
        transition_epochs=args.transition_epochs,
        freeze_gmm_epochs=args.freeze_gmm_epochs,
        gene_groups=gene_groups,
        collapse_threshold=args.collapse_threshold,
        collapse_noise_scale=args.collapse_noise_scale,
    )
    trainer(train_loader, epochs=epochs, checkpoint_every=args.checkpoint_every)

    eval_bs = args.eval_batch_size or max(1, args.batch_size // 2)
    val_loader = get_omics_dataloader(
        path=args.dataset,
        batch_size=eval_bs,
        shuffle=False,
        drop_last=False,
        feature_subset=list(val_ids),
        logger=logger,
    )
    evaluator = Evaluator(
        model, gmm_loss, device=device, logger=logger, save_dir=out_dir,
        is_progress_bar=not args.no_progress_bar,
    )
    val_loss = evaluator(val_loader)

    metadata = vars(args).copy()
    metadata.update(
        {
            "n_features": n_features,
            "n_modules": args.n_modules,
            "k_sweep": True,
            "val_frac": args.val_frac,
            "val_seed": args.val_seed,
            "phase": args._phase,
            "epochs": epochs,
        }
    )
    save_model(trainer.model, out_dir, metadata=metadata)

    hard_assignments = None
    if compute_assignments:
        from bsvae.networks.module_extraction import extract_gmm_modules
        result = extract_gmm_modules(
            model=model,
            dataloader=val_loader,
            feature_ids=val_ids,
            output_dir=None,
        )
        hard_assignments = result["hard_assignments"].tolist()

    return val_loss, hard_assignments


def _write_results_csv(path: str, rows: Sequence[SweepResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k",
                "val_loss_mean",
                "val_loss_std",
                "stability_ari_mean",
                "stability_ari_std",
                "reps",
                "out_dir",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.k,
                    f"{r.val_loss_mean:.6f}",
                    f"{r.val_loss_std:.6f}",
                    "" if r.stability_ari_mean is None else f"{r.stability_ari_mean:.6f}",
                    "" if r.stability_ari_std is None else f"{r.stability_ari_std:.6f}",
                    r.reps,
                    r.out_dir,
                ]
            )


def parse_args(cli_args=None):
    p = argparse.ArgumentParser(
        prog="bsvae-sweep-k",
        description="Two-pass K sweep (coarse → fine) with held-out validation.",
        formatter_class=FormatterNoDuplicate,
    )

    p.add_argument("name", type=str, help="Sweep name (output subdirectory).")
    p.add_argument("--dataset", required=True, help="Expression matrix (features × samples).")

    # General
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", choices=["debug", "info", "warning", "error"], default="info")
    p.add_argument("--no-progress-bar", action="store_true", default=False)

    # Sweep controls
    p.add_argument("--k-grid", type=str, default="8,12,16,24,32",
                   help="Comma list or range start:end:step (default: 8,12,16,24,32).")
    p.add_argument("--sweep-epochs", type=int, default=60)
    p.add_argument("--stability-reps", type=int, default=1,
                   help="Number of training replicates per K for stability (default: 1).")
    p.add_argument("--stability-seed", type=int, default=13,
                   help="Base seed for stability replicates.")
    p.add_argument("--train-final", action="store_true", default=True,
                   help="Retrain best K on full dataset after sweep.")
    p.add_argument("--no-train-final", dest="train_final", action="store_false")
    p.add_argument("--final-epochs", type=int, default=None)

    # Validation split (avoid overfitting to K)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=13)

    # Training options (match bsvae-train)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--transition-epochs", type=int, default=10)
    p.add_argument("--freeze-gmm-epochs", type=int, default=0)
    p.add_argument("--collapse-threshold", type=float, default=0.5)
    p.add_argument("--collapse-noise-scale", type=float, default=0.5)

    # Model architecture
    p.add_argument("--n-modules", "-K", type=int, default=20)
    p.add_argument("--latent-dim", "-z", type=int, default=32)
    def _ast_eval(val):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    p.add_argument("--hidden-dims", type=_ast_eval, default=[512, 256, 128])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-batch-norm", action="store_false", dest="use_batch_norm")
    p.add_argument("--use-batch-norm", action="store_true", dest="use_batch_norm", default=True)
    p.add_argument("--sigma-min", type=float, default=0.3)

    # Loss hyperparameters
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--free-bits", type=float, default=0.5)
    p.add_argument("--kl-warmup-epochs", type=int, default=0)
    p.add_argument("--kl-anneal-mode", choices=["linear", "cyclical"], default="linear")
    p.add_argument("--kl-cycle-length", type=int, default=50)
    p.add_argument("--sep-strength", type=float, default=0.1)
    p.add_argument("--sep-alpha", type=float, default=2.0)
    p.add_argument("--bal-strength", type=float, default=0.01)
    p.add_argument("--bal-ema-blend", type=float, default=0.5)
    p.add_argument("--pi-entropy-strength", type=float, default=0.0)
    p.add_argument("--hier-strength", type=float, default=0.0)
    p.add_argument("--tx2gene", type=str, default=None)

    # Eval
    p.add_argument("--eval-batch-size", type=int, default=None)

    return p.parse_args(cli_args)


def main(args):
    logger = _setup_logging(args.log_level)
    set_seed(args.seed)

    sweep_root = os.path.join(args.outdir, args.name, "sweep_k")
    create_safe_directory(sweep_root, logger=logger)

    dataset = OmicsDataset(args.dataset, logger=logger)
    train_ids, val_ids = _split_feature_ids(dataset.feature_ids, args.val_frac, args.val_seed)
    logger.info(
        "K sweep split: %d train features, %d val features (val_frac=%.2f)",
        len(train_ids), len(val_ids), args.val_frac,
    )

    k_grid = _parse_k_list(args.k_grid)
    if not k_grid:
        raise ValueError("coarse grid is empty")

    results: list[SweepResult] = []

    # ----------------------------
    # Sweep pass
    # ----------------------------
    from bsvae.simulation.metrics import compute_ari
    for k in k_grid:
        args.n_modules = int(k)
        per_rep_losses: list[float] = []
        per_rep_assignments: list[list[int]] = []
        k_dir = os.path.join(sweep_root, f"k{k}")
        logger.info("Sweep: K=%d | out=%s | reps=%d", k, k_dir, args.stability_reps)

        for rep in range(max(1, args.stability_reps)):
            rep_seed = args.stability_seed + rep
            rep_dir = os.path.join(k_dir, f"rep_{rep:02d}")
            args._phase = f"sweep_rep_{rep:02d}"
            val_loss, hard = _train_and_eval(
                args=args,
                train_ids=train_ids,
                val_ids=val_ids,
                out_dir=rep_dir,
                epochs=args.sweep_epochs,
                logger=logger,
                seed=rep_seed,
                compute_assignments=(args.stability_reps > 1),
            )
            per_rep_losses.append(val_loss)
            if hard is not None:
                per_rep_assignments.append(hard)

        # Stability metric: mean pairwise ARI on held-out features
        ari_vals: list[float] = []
        if args.stability_reps > 1 and len(per_rep_assignments) >= 2:
            for i in range(len(per_rep_assignments)):
                for j in range(i + 1, len(per_rep_assignments)):
                    ari_vals.append(
                        compute_ari(
                            np.array(per_rep_assignments[i]),
                            np.array(per_rep_assignments[j]),
                        )
                    )
        val_mean = float(sum(per_rep_losses) / len(per_rep_losses))
        val_std = float(np.std(per_rep_losses)) if len(per_rep_losses) > 1 else 0.0
        ari_mean = float(np.mean(ari_vals)) if ari_vals else None
        ari_std = float(np.std(ari_vals)) if ari_vals else None

        results.append(
            SweepResult(
                k=k,
                val_loss_mean=val_mean,
                val_loss_std=val_std,
                stability_ari_mean=ari_mean,
                stability_ari_std=ari_std,
                reps=max(1, args.stability_reps),
                out_dir=k_dir,
            )
        )

    if args.stability_reps > 1:
        best = max(
            results,
            key=lambda r: (r.stability_ari_mean if r.stability_ari_mean is not None else -1.0),
        )
        logger.info("Best K by stability: K=%d (ARI=%.4f)", best.k, best.stability_ari_mean or -1.0)
    else:
        best = min(results, key=lambda r: r.val_loss_mean)
        logger.info("Best K by val loss: K=%d (val_loss=%.4f)", best.k, best.val_loss_mean)

    results_csv = os.path.join(sweep_root, "sweep_results.csv")
    _write_results_csv(results_csv, results)

    summary = {
        "selected_k": best.k,
        "selection_metric": "stability_ari" if args.stability_reps > 1 else "val_loss",
        "k_grid": k_grid,
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
        "stability_reps": args.stability_reps,
        "stability_seed": args.stability_seed,
        "results_csv": results_csv,
    }
    with open(os.path.join(sweep_root, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ----------------------------
    # Final training on full data
    # ----------------------------
    if args.train_final:
        final_k = best.k
        final_dir = os.path.join(args.outdir, args.name, f"final_k{final_k}")
        logger.info("Training final model on full dataset (K=%d) → %s", final_k, final_dir)
        args.n_modules = int(final_k)
        args._phase = "final"
        epochs = args.final_epochs if args.final_epochs is not None else args.sweep_epochs

        full_loader = get_omics_dataloader(
            path=args.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            feature_subset=None,
            logger=logger,
        )
        n_features = full_loader.dataset.n_samples
        feature_ids = full_loader.dataset.feature_ids
        device = get_device(use_gpu=not args.no_cuda)

        model = _build_model(args, n_features).to(device)
        gmm_loss, warmup_loss = _build_losses(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        gene_groups = {}
        if args.hier_strength > 0 and args.tx2gene:
            from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene
            tx2gene = load_tx2gene(path=args.tx2gene, feature_ids=feature_ids)
            gene_groups = group_isoforms_by_gene(tx2gene, feature_ids)
            logger.info("Hierarchical groups: %d multi-isoform genes", len(gene_groups))

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            gmm_loss_f=gmm_loss,
            warmup_loss_f=warmup_loss,
            device=device,
            logger=logger,
            save_dir=final_dir,
            is_progress_bar=not args.no_progress_bar,
            warmup_epochs=args.warmup_epochs,
            transition_epochs=args.transition_epochs,
            freeze_gmm_epochs=args.freeze_gmm_epochs,
            gene_groups=gene_groups,
            collapse_threshold=args.collapse_threshold,
            collapse_noise_scale=args.collapse_noise_scale,
        )
        trainer(full_loader, epochs=epochs, checkpoint_every=args.checkpoint_every)

        metadata = vars(args).copy()
        metadata.update(
            {
                "n_features": n_features,
                "n_modules": args.n_modules,
                "k_sweep": True,
                "phase": "final",
                "selected_k": final_k,
                "sweep_summary": summary,
                "epochs": epochs,
            }
        )
        save_model(trainer.model, final_dir, metadata=metadata)


def cli():
    args = parse_args(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    cli()
