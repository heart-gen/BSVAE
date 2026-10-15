"""
bsvae-sweep-k — Sweep the number of modules (K) with held-out validation.

Usage
-----
bsvae-sweep-k NAME --dataset PATH [options]

Selection
---------
* ``--stability-reps 1`` (default): lowest held-out ELBO (reconstruction +
  β·KL, without the regularisers, which are not comparable across K).
* ``--stability-reps > 1``: highest mean pairwise ARI of hard assignments on
  the held-out features across replicates; ties broken by held-out ELBO.

The held-out split is on features; when ``--tx2gene`` is given it is on
genes so sibling isoforms never straddle the split.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Subset

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
from bsvae.utils.datasets import OmicsDataset
from bsvae.utils.helpers import FormatterNoDuplicate, create_safe_directory, get_device, get_n_params, set_seed
from bsvae.utils.modelIO import save_model
from bsvae.utils.training import Evaluator


@dataclass
class SweepResult:
    k: int
    val_elbo_mean: float
    val_elbo_std: float
    val_loss_mean: float
    stability_ari_mean: Optional[float]
    stability_ari_std: Optional[float]
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


def _split_feature_indices(
    feature_ids: Sequence[str],
    val_frac: float,
    seed: int,
    gene_groups: Optional[Dict[str, List[int]]] = None,
) -> tuple[list[int], list[int]]:
    """Split dataset indices into train/val; whole genes move together when groups are given."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("--val-frac must be between 0 and 1 (exclusive)")
    rng = random.Random(seed)
    n = len(feature_ids)
    units: List[List[int]] = []
    if gene_groups:
        grouped = set()
        for idxs in gene_groups.values():
            units.append(sorted(idxs))
            grouped.update(idxs)
        units.extend([[i] for i in range(n) if i not in grouped])
    else:
        units = [[i] for i in range(n)]
    rng.shuffle(units)
    n_val = max(1, int(n * val_frac))
    val: List[int] = []
    train: List[int] = []
    for unit in units:
        if len(val) < n_val:
            val.extend(unit)
        else:
            train.extend(unit)
    return sorted(train), sorted(val)


def _subset_dataset(dataset: OmicsDataset, indices: Sequence[int], return_index: bool) -> OmicsDataset:
    """Materialise a subset as its own OmicsDataset (indices are re-based)."""
    idx = list(indices)
    return OmicsDataset(
        data=dataset.data[idx].numpy(),
        feature_ids=[dataset.feature_ids[i] for i in idx],
        sample_ids=dataset.sample_ids,
        logger=dataset.logger,
        return_index=return_index,
    )


def _train_and_eval(
    args,
    train_ds: OmicsDataset,
    val_ds: OmicsDataset,
    out_dir: str,
    epochs: int,
    logger: logging.Logger,
    seed: int,
    compute_assignments: bool,
) -> tuple[dict, list[int] | None]:
    create_safe_directory(out_dir, logger=logger)
    set_seed(seed)
    device = get_device(use_gpu=not args.no_cuda)

    n_features = train_ds.n_samples
    gene_groups = build_gene_groups(args, train_ds.feature_ids, logger)
    train_loader = build_train_loader(train_ds, args, gene_groups, logger)

    model = build_model(args, n_features).to(device)
    logger.info(
        "Model: %d features, %d latent dims, %d modules | params=%d",
        n_features, args.latent_dim, args.n_modules, get_n_params(model),
    )
    gmm_loss, warmup_loss = build_losses(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = build_trainer(
        args, model, optimizer, gmm_loss, warmup_loss, device, logger, out_dir, gene_groups,
    )
    trainer(train_loader, epochs=epochs, checkpoint_every=args.checkpoint_every)

    eval_bs = args.eval_batch_size or max(1, args.batch_size // 2)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, drop_last=False)
    evaluator = Evaluator(
        model, gmm_loss, device=device, logger=logger, save_dir=out_dir,
        is_progress_bar=not args.no_progress_bar,
    )
    val = evaluator(val_loader, return_components=True)

    metadata = training_metadata(args, n_features)
    metadata.update(
        {
            "n_modules": args.n_modules,
            "k_sweep": True,
            "val_frac": args.val_frac,
            "val_seed": args.val_seed,
            "phase": getattr(args, "_phase", "sweep"),
            "epochs": epochs,
            "val_metrics": val,
        }
    )
    save_model(trainer.model, out_dir, metadata=metadata)

    hard_assignments = None
    if compute_assignments:
        from bsvae.networks.module_extraction import extract_gmm_modules
        result = extract_gmm_modules(
            model=model,
            dataloader=val_loader,
            feature_ids=val_ds.feature_ids,
            output_dir=None,
            disable_progress=True,
        )
        hard_assignments = result["hard_assignments"].tolist()

    return val, hard_assignments


def _write_results_csv(path: str, rows: Sequence[SweepResult]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "k", "val_elbo_mean", "val_elbo_std", "val_loss_mean",
                "stability_ari_mean", "stability_ari_std", "reps", "out_dir",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.k,
                    f"{r.val_elbo_mean:.6f}",
                    f"{r.val_elbo_std:.6f}",
                    f"{r.val_loss_mean:.6f}",
                    "" if r.stability_ari_mean is None else f"{r.stability_ari_mean:.6f}",
                    "" if r.stability_ari_std is None else f"{r.stability_ari_std:.6f}",
                    r.reps,
                    r.out_dir,
                ]
            )


def parse_args(cli_args=None):
    p = argparse.ArgumentParser(
        prog="bsvae-sweep-k",
        description="K sweep with held-out validation (ELBO or stability ARI).",
        formatter_class=FormatterNoDuplicate,
    )
    p.add_argument("name", type=str, help="Sweep name (output subdirectory).")
    p.add_argument("--dataset", required=True, help="Expression matrix (features × samples).")

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
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=13)

    add_general_args(p)
    add_schedule_args(p, checkpoint_default=0)
    add_model_args(p)
    add_loss_args(p)
    return p.parse_args(cli_args)


def main(args):
    logger = setup_logging(args.log_level, name=__name__)
    set_seed(args.seed)

    sweep_root = os.path.join(args.outdir, args.name, "sweep_k")
    create_safe_directory(sweep_root, logger=logger)

    # Load once; every run works on in-memory subsets.
    dataset = OmicsDataset(args.dataset, logger=logger, return_index=True)
    split_groups = {}
    if args.tx2gene:
        from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene
        tx2gene = load_tx2gene(
            path=args.tx2gene, feature_ids=dataset.feature_ids,
            keep_version=args.keep_tx_version, logger=logger,
        )
        split_groups = group_isoforms_by_gene(tx2gene, dataset.feature_ids)
    train_idx, val_idx = _split_feature_indices(
        dataset.feature_ids, args.val_frac, args.val_seed, split_groups or None,
    )
    train_ds = _subset_dataset(dataset, train_idx, return_index=True)
    val_ds = _subset_dataset(dataset, val_idx, return_index=False)
    logger.info(
        "K sweep split: %d train features, %d val features (val_frac=%.2f%s)",
        len(train_idx), len(val_idx), args.val_frac,
        ", gene-aware" if split_groups else "",
    )

    k_grid = _parse_k_list(args.k_grid)
    if not k_grid:
        raise ValueError("coarse grid is empty")

    results: list[SweepResult] = []
    from bsvae.simulation.metrics import compute_ari
    for k in k_grid:
        args.n_modules = int(k)
        per_rep: list[dict] = []
        per_rep_assignments: list[list[int]] = []
        k_dir = os.path.join(sweep_root, f"k{k}")
        logger.info("Sweep: K=%d | out=%s | reps=%d", k, k_dir, args.stability_reps)

        for rep in range(max(1, args.stability_reps)):
            rep_seed = args.stability_seed + rep
            rep_dir = os.path.join(k_dir, f"rep_{rep:02d}")
            args._phase = f"sweep_rep_{rep:02d}"
            val, hard = _train_and_eval(
                args=args, train_ds=train_ds, val_ds=val_ds, out_dir=rep_dir,
                epochs=args.sweep_epochs, logger=logger, seed=rep_seed,
                compute_assignments=(args.stability_reps > 1),
            )
            per_rep.append(val)
            if hard is not None:
                per_rep_assignments.append(hard)

        ari_vals: list[float] = []
        if args.stability_reps > 1 and len(per_rep_assignments) >= 2:
            for i in range(len(per_rep_assignments)):
                for j in range(i + 1, len(per_rep_assignments)):
                    ari_vals.append(
                        compute_ari(np.array(per_rep_assignments[i]), np.array(per_rep_assignments[j]))
                    )
        elbos = [v["elbo"] for v in per_rep]
        losses = [v["loss"] for v in per_rep]
        results.append(
            SweepResult(
                k=k,
                val_elbo_mean=float(np.mean(elbos)),
                val_elbo_std=float(np.std(elbos)) if len(elbos) > 1 else 0.0,
                val_loss_mean=float(np.mean(losses)),
                stability_ari_mean=float(np.mean(ari_vals)) if ari_vals else None,
                stability_ari_std=float(np.std(ari_vals)) if ari_vals else None,
                reps=max(1, args.stability_reps),
                out_dir=k_dir,
            )
        )

    if args.stability_reps > 1:
        best = max(
            results,
            key=lambda r: (
                r.stability_ari_mean if r.stability_ari_mean is not None else -1.0,
                -r.val_elbo_mean,
            ),
        )
        logger.info("Best K by stability: K=%d (ARI=%.4f)", best.k, best.stability_ari_mean or -1.0)
        metric = "stability_ari"
    else:
        best = min(results, key=lambda r: r.val_elbo_mean)
        logger.info("Best K by held-out ELBO: K=%d (elbo=%.4f)", best.k, best.val_elbo_mean)
        metric = "val_elbo"

    results_csv = os.path.join(sweep_root, "sweep_results.csv")
    _write_results_csv(results_csv, results)
    summary = {
        "selected_k": best.k,
        "selection_metric": metric,
        "k_grid": k_grid,
        "val_frac": args.val_frac,
        "val_seed": args.val_seed,
        "gene_aware_split": bool(split_groups),
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
        create_safe_directory(final_dir, logger=logger)
        set_seed(args.seed)
        device = get_device(use_gpu=not args.no_cuda)

        n_features = dataset.n_samples
        gene_groups = build_gene_groups(args, dataset.feature_ids, logger)
        full_loader = build_train_loader(dataset, args, gene_groups, logger)
        model = build_model(args, n_features).to(device)
        gmm_loss, warmup_loss = build_losses(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        trainer = build_trainer(
            args, model, optimizer, gmm_loss, warmup_loss, device, logger, final_dir, gene_groups,
        )
        trainer(full_loader, epochs=epochs, checkpoint_every=args.checkpoint_every)

        metadata = training_metadata(args, n_features)
        metadata.update(
            {
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
