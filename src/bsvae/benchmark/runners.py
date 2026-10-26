"""
Method runners for the benchmark harness.

``run_bsvae`` trains a GMMModuleVAE through the same builders as
``bsvae-train`` and post-processes the assignments; ``run_wgcna`` calls the
packaged ``wgcna.R`` through ``Rscript``.  Both return a ``MethodResult``
with a ``labels`` Series (feature_id → int, -1 = grey) so that one metric
function scores every method.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WGCNA_SCRIPT = Path(__file__).with_name("wgcna.R")

# BSVAE defaults for benchmarking (shared with bsvae-train; overrides only).
# beta=2.0 and no latent-correlation loss were the most consistent settings
# across the long-tail, nonlinear and isoform reference scenarios once the
# KL is scaled per element; the latent-correlation loss used by the old
# benchmark compensated for the previous over-weighted KL and now hurts.
BSVAE_DEFAULTS: Dict = {
    "epochs": 150,
    "warmup_epochs": 30,
    "transition_epochs": 10,
    "latent_dim": 32,
    "hidden_dims": [256, 128],
    "batch_size": 128,
    "lr": 5e-4,
    "beta": 2.0,
    "normalize_input": True,
    "latent_corr_strength": 0.0,
    "bal_type": "entropy",
    "bal_strength": 0.1,
    "sep_strength": 0.1,
    "no_progress_bar": True,
    "checkpoint_every": 0,
}

POST_DEFAULTS: Dict = {
    "min_gamma": 0.5,
    "min_module_size": 10,
    "merge_cut_height": 0.4,
    "kme_reassign": False,
}

# The GMM is deliberately over-clustered: K-means initialisation allocates
# centroids to the largest modules, so small modules only get their own
# component when K exceeds the expected module count; the eigengene merge in
# post-processing then re-joins split pieces of large modules.
OVER_CLUSTER_FACTOR: float = 3.0
MIN_OVER_CLUSTER_K: int = 20

WGCNA_DEFAULTS: Dict = {
    "network_type": "signed",
    "soft_power": None,
    "min_module_size": 10,
    "deep_split": 2,
    "merge_cut_height": 0.25,
    "blockwise_threshold": 20000,
    "max_block_size": 5000,
    "n_threads": 1,
    "timeout_s": 7200,
}


@dataclass
class MethodResult:
    method: str
    status: str                                  # "ok" | "skipped" | "failed"
    labels: Optional[pd.Series] = None           # feature_id → int (-1 grey)
    gamma: Optional[np.ndarray] = None
    eigengenes: Optional[pd.DataFrame] = None
    runtime_s: float = 0.0
    params: Dict = field(default_factory=dict)
    message: str = ""
    edges: Optional[pd.DataFrame] = None         # node1, node2, edge_type, weight (edge-list methods)


# ---------------------------------------------------------------------------
# WGCNA
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def wgcna_available() -> bool:
    """True when Rscript and the WGCNA package are both usable."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    try:
        proc = subprocess.run(
            [rscript, "-e", 'quit(status = if (requireNamespace("WGCNA", quietly = TRUE)) 0 else 1)'],
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def run_wgcna(
    expr: pd.DataFrame,
    *,
    workdir: str | Path,
    network_type: str = "signed",
    soft_power: Optional[int] = None,
    min_module_size: int = 10,
    deep_split: int = 2,
    merge_cut_height: float = 0.25,
    blockwise_threshold: int = 20000,
    max_block_size: int = 5000,
    n_threads: int = 1,
    seed: int = 42,
    timeout_s: int = 7200,
    fixed_k: Optional[int] = None,
) -> MethodResult:
    """Run WGCNA on a features × samples matrix via ``Rscript``.

    ``fixed_k`` replaces the dynamic tree cut by a Ward clustering of the TOM dissimilarity cut
    into that many clusters (no merge step, no size floor): WGCNA's network at a forced
    resolution, single-block path only.
    """
    params = dict(
        network_type=network_type, soft_power=soft_power, min_module_size=min_module_size,
        deep_split=deep_split, merge_cut_height=merge_cut_height,
        blockwise_threshold=blockwise_threshold, max_block_size=max_block_size, seed=seed,
        fixed_k=fixed_k,
    )
    if not wgcna_available():
        logger.warning("Rscript/WGCNA not available; skipping WGCNA")
        return MethodResult("wgcna", "skipped", params=params, message="Rscript or WGCNA not found")

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    expr_path = workdir / "expr.csv.gz"
    expr.to_csv(expr_path, compression="gzip")

    cmd = [
        shutil.which("Rscript"), "--vanilla", str(WGCNA_SCRIPT),
        "--expr", str(expr_path), "--outdir", str(workdir),
        "--network-type", str(network_type),
        "--min-module-size", str(int(min_module_size)),
        "--deep-split", str(int(deep_split)),
        "--merge-cut-height", str(float(merge_cut_height)),
        "--blockwise-threshold", str(int(blockwise_threshold)),
        "--max-block-size", str(int(max_block_size)),
        "--threads", str(int(n_threads)),
        "--seed", str(int(seed)),
    ]
    if soft_power is not None:
        cmd += ["--soft-power", str(int(soft_power))]
    if fixed_k is not None:
        cmd += ["--fixed-k", str(int(fixed_k))]

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return MethodResult("wgcna", "failed", params=params, runtime_s=time.perf_counter() - t0,
                            message=f"timeout after {timeout_s}s")
    runtime = time.perf_counter() - t0
    (workdir / "wgcna.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        return MethodResult("wgcna", "failed", params=params, runtime_s=runtime,
                            message="Rscript failed: " + " | ".join(tail))

    mods = pd.read_csv(workdir / "modules.csv")
    labels = pd.Series(mods["module"].astype(int).values, index=mods["feature_id"].astype(str), name="module")
    eig = None
    eig_path = workdir / "eigengenes.csv"
    if eig_path.exists():
        eig = pd.read_csv(eig_path, index_col=0)
    try:
        params.update(json.loads((workdir / "params.json").read_text()))
    except Exception:  # pragma: no cover
        pass
    return MethodResult("wgcna", "ok", labels=labels, eigengenes=eig, runtime_s=runtime, params=params)


# ---------------------------------------------------------------------------
# BSVAE
# ---------------------------------------------------------------------------

def _bsvae_namespace(overrides: Dict) -> argparse.Namespace:
    """Build an args namespace with bsvae-train's defaults plus overrides."""
    from bsvae.cli._common import add_general_args, add_loss_args, add_model_args, add_schedule_args

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--epochs", type=int, default=100)
    add_general_args(p)
    add_schedule_args(p, checkpoint_default=0)
    add_model_args(p)
    add_loss_args(p)
    args = p.parse_args([])
    cfg = dict(BSVAE_DEFAULTS)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    unknown = [k for k in cfg if not hasattr(args, k)]
    if unknown:
        raise ValueError(f"Unknown BSVAE option(s): {unknown}")
    for k, v in cfg.items():
        setattr(args, k, v)
    return args


FEATURES_PER_COMPONENT: int = 25


def over_cluster_k(
    n_modules: int,
    factor: float = OVER_CLUSTER_FACTOR,
    minimum: int = MIN_OVER_CLUSTER_K,
    n_features: Optional[int] = None,
) -> int:
    """
    Number of GMM components to train for an expected ``n_modules`` modules:
    ``max(minimum, factor * n_modules)``, capped so that components average at
    least ``FEATURES_PER_COMPONENT`` features, and never below ``n_modules``.
    """
    k = max(int(minimum), int(round(factor * n_modules)))
    if n_features is not None:
        k = min(k, max(1, int(n_features) // FEATURES_PER_COMPONENT))
    return int(max(k, int(n_modules)))


def run_bsvae(
    expr: pd.DataFrame,
    *,
    n_modules: int,
    seed: int,
    workdir: str | Path,
    tx2gene: Optional[pd.DataFrame] = None,
    train_cfg: Optional[Dict] = None,
    post_cfg: Optional[Dict] = None,
    device=None,
    postprocess: bool = True,
    over_cluster_factor: Optional[float] = OVER_CLUSTER_FACTOR,
) -> MethodResult:
    """
    Train BSVAE on a features × samples matrix and return post-processed
    module labels (grey = -1), merged γ and eigengenes.

    ``n_modules`` is the expected number of modules; the GMM itself is trained
    with ``over_cluster_k(n_modules)`` components when ``over_cluster_factor``
    is set (``None`` or ``postprocess=False`` trains exactly ``n_modules``).
    """
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    from bsvae.cli._common import (
        build_gene_groups, build_losses, build_model, build_train_loader, build_trainer,
    )
    from bsvae.networks.extract_networks import extract_mu_gamma
    from bsvae.networks.postprocess import postprocess_assignments
    from bsvae.utils.datasets import OmicsDataset
    from bsvae.utils.helpers import set_seed

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg = dict(train_cfg or {})
    k_train = int(n_modules)
    if postprocess and over_cluster_factor:
        k_train = over_cluster_k(int(n_modules), over_cluster_factor, n_features=expr.shape[0])
    cfg["n_modules"] = k_train
    cfg["seed"] = int(seed)
    if tx2gene is not None and (cfg.get("hier_strength", 0.0) > 0 or cfg.get("isoform_stratified", False)):
        tx_path = workdir / "tx2gene.tsv"
        tx2gene.to_csv(tx_path, sep="\t", index=False)
        cfg["tx2gene"] = str(tx_path)
    post = dict(POST_DEFAULTS)
    post.update(post_cfg or {})
    params = {"train": {k: v for k, v in cfg.items()}, "post": post,
              "n_modules_expected": int(n_modules), "n_components_trained": k_train}

    quiet = logging.getLogger("bsvae.benchmark.bsvae")
    quiet.setLevel(logging.WARNING)
    t0 = time.perf_counter()
    try:
        args = _bsvae_namespace(cfg)
        set_seed(seed)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        dataset = OmicsDataset.from_frame(expr, logger=quiet, return_index=True)
        gene_groups = build_gene_groups(args, dataset.feature_ids, quiet)
        loader = build_train_loader(dataset, args, gene_groups, quiet)
        model = build_model(args, dataset.n_samples).to(device)
        gmm_loss, warmup_loss = build_losses(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        trainer = build_trainer(args, model, optimizer, gmm_loss, warmup_loss, device, quiet, str(workdir), gene_groups)
        trainer(loader, epochs=int(args.epochs), checkpoint_every=0)

        eval_ds = OmicsDataset.from_frame(expr, logger=quiet, return_index=False)
        eval_loader = DataLoader(eval_ds, batch_size=256, shuffle=False)
        mu, gamma, ids = extract_mu_gamma(model, eval_loader, device=device, disable_progress=True)
        feature_ids = [str(f) for f in ids]
        if postprocess:
            res = postprocess_assignments(
                gamma, feature_ids, expr,
                min_gamma=post["min_gamma"], min_module_size=post["min_module_size"],
                merge_cut_height=post["merge_cut_height"], kme_reassign_=post.get("kme_reassign", False),
            )
            labels = pd.Series(res.labels.astype(int), index=feature_ids, name="module")
            gamma_out, eig = res.gamma, res.eigengenes
            params["post"]["n_grey"] = int(res.n_grey)
            params["post"]["n_merges"] = len(res.merge_log)
        else:
            labels = pd.Series(gamma.argmax(axis=1).astype(int), index=feature_ids, name="module")
            gamma_out, eig = gamma, None
        np.savez_compressed(workdir / "gamma.npz", gamma=gamma_out, mu=mu, gamma_raw=gamma,
                            feature_ids=np.array(feature_ids, dtype=object))
        from bsvae.utils.modelIO import save_model
        save_model(model, str(workdir), metadata={**{k: v for k, v in vars(args).items() if not k.startswith("_")},
                                                  "n_features": dataset.n_samples})
        pd.DataFrame({"feature_id": feature_ids, "module": labels.values}).to_csv(workdir / "modules.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        logger.exception("BSVAE run failed")
        return MethodResult("bsvae", "failed", params=params, runtime_s=time.perf_counter() - t0, message=str(exc))
    return MethodResult("bsvae", "ok", labels=labels, gamma=gamma_out, eigengenes=eig,
                        runtime_s=time.perf_counter() - t0, params=params)


__all__ = ["MethodResult", "run_bsvae", "run_wgcna", "wgcna_available", "over_cluster_k",
           "BSVAE_DEFAULTS", "POST_DEFAULTS", "WGCNA_DEFAULTS", "OVER_CLUSTER_FACTOR"]
