"""
Data-driven K for the benchmark harness (``k_mode`` ``elbo`` or ``stability``).

The oracle mode hands BSVAE the planted module count. The two data-driven modes run one
``bsvae-sweep-k`` per dataset, cached under ``<run_dir>/methods/_ksweep`` and shared by every
method row of that dataset, which scores a K grid by held-out ELBO (gene-grouped validation
split when the scenario has transcript structure) and, with more than one stability replicate,
by the mean pairwise ARI of the held-out hard assignments across replicates. ``elbo`` and
``stability`` read their K from the same table, so the two rows differ only in the final fit.
The chosen K is then over-clustered like any other expected module count (``over_cluster_k``).
"""
from __future__ import annotations

import fcntl
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

K_MODES = ("oracle", "elbo", "stability")
K_MODE_METHODS = {"bsvae_elbo_k": "elbo", "bsvae_stability_k": "stability"}
K_GRID_DEFAULT: Sequence[int] = (4, 6, 8, 12, 16, 24, 32)
SWEEP_EPOCHS_DEFAULT = 60
STABILITY_REPS_DEFAULT = 3
SWEEP_NAME = "sweep"

logger = logging.getLogger(__name__)


def check_k_mode(k_mode: str) -> None:
    if k_mode not in K_MODES:
        raise ValueError(f"k_mode must be one of {K_MODES} (got {k_mode!r}); 'oracle' is the planted count")


def sweep_results_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "methods" / "_ksweep" / SWEEP_NAME / "sweep_k" / "sweep_results.csv"


def run_sweep(
    run_dir: str | Path,
    expr_path: str | Path,
    tx2gene_path: Optional[str | Path] = None,
    *,
    train_cfg: Optional[Dict] = None,
    k_grid: Sequence[int] = K_GRID_DEFAULT,
    sweep_epochs: int = SWEEP_EPOCHS_DEFAULT,
    stability_reps: int = STABILITY_REPS_DEFAULT,
    seed: int = 0,
    no_cuda: bool = False,
) -> Path:
    """Run the K sweep for one dataset unless its results table already exists; returns that table.

    A file lock serialises concurrent rows of the same dataset (the pack runner starts several rows
    at once), so the sweep is trained once and the second row reads the finished table.
    """
    out = sweep_results_path(run_dir)
    if out.exists():
        return out
    root = out.parents[2]                      # <run_dir>/methods/_ksweep
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            if not out.exists():
                _run(root, expr_path, tx2gene_path, train_cfg or {}, k_grid, sweep_epochs, stability_reps, seed, no_cuda)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    if not out.exists():
        raise RuntimeError(f"K sweep finished without writing {out}")
    return out


def _run(root: Path, expr_path, tx2gene_path, train_cfg: Dict, k_grid, sweep_epochs, stability_reps, seed, no_cuda) -> None:
    from bsvae.benchmark.runners import BSVAE_DEFAULTS
    from bsvae.cli.sweep_k import main as sweep_main, parse_args as sweep_parse_args

    argv = [SWEEP_NAME, "--dataset", str(expr_path), "--outdir", str(root),
            "--k-grid", ",".join(str(int(k)) for k in k_grid),
            "--sweep-epochs", str(int(sweep_epochs)), "--stability-reps", str(int(stability_reps)),
            "--no-train-final", "--seed", str(int(seed)), "--log-level", "warning"]
    if tx2gene_path is not None:
        argv += ["--tx2gene", str(tx2gene_path)]
    if no_cuda:
        argv += ["--no-cuda"]
    args = sweep_parse_args(argv)
    cfg = dict(BSVAE_DEFAULTS)
    cfg.update({k: v for k, v in train_cfg.items() if v is not None})
    for k in ("epochs", "n_modules", "seed", "no_cuda", "tx2gene", "outdir", "log_level"):
        cfg.pop(k, None)
    unknown = [k for k in cfg if not hasattr(args, k)]
    if unknown:
        raise ValueError(f"Unknown BSVAE option(s) for the K sweep: {unknown}")
    for k, v in cfg.items():
        setattr(args, k, v)
    logger.info("K sweep: grid %s, %d epochs, %d stability reps -> %s", list(k_grid), sweep_epochs, stability_reps, root)
    sweep_main(args)


def select_k(results_csv: str | Path, k_mode: str) -> Tuple[int, Dict]:
    """K for ``k_mode`` from a sweep table, with the selections of both modes in the returned info."""
    t = pd.read_csv(results_csv)
    if t.empty or "val_elbo_mean" not in t:
        raise ValueError(f"empty or malformed sweep table: {results_csv}")
    t = t.sort_values("k").reset_index(drop=True)
    elbo_row = t.loc[t["val_elbo_mean"].idxmin()]
    info: Dict = {"k_grid": [int(k) for k in t["k"]], "k_elbo": int(elbo_row["k"]),
                  "val_elbo_range": float(t["val_elbo_mean"].max() - t["val_elbo_mean"].min())}
    stab = t["stability_ari_mean"] if "stability_ari_mean" in t else pd.Series(dtype=float)
    if stab.notna().any():
        s = t.sort_values(["stability_ari_mean", "val_elbo_mean"], ascending=[False, True])
        info["k_stability"] = int(s["k"].iloc[0])
        info["stability_ari_at_k"] = float(s["stability_ari_mean"].iloc[0])
    if k_mode == "elbo":
        return info["k_elbo"], info
    if k_mode == "stability":
        if "k_stability" not in info:
            raise ValueError("the sweep has no stability column; rerun it with stability_reps > 1")
        return info["k_stability"], info
    check_k_mode(k_mode)
    raise ValueError("select_k is for the data-driven modes; oracle K comes from the scenario")


__all__ = ["K_MODES", "K_MODE_METHODS", "K_GRID_DEFAULT", "SWEEP_EPOCHS_DEFAULT", "STABILITY_REPS_DEFAULT",
           "check_k_mode", "run_sweep", "select_k", "sweep_results_path"]
