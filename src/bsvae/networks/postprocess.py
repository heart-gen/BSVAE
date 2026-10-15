"""
WGCNA-style post-processing of GMM module assignments.

The GMM gives every feature a soft assignment γ.  WGCNA, by contrast, leaves
low-connectivity features unassigned ("grey"), enforces a minimum module
size, merges modules whose eigengenes are highly correlated, and reports
module membership (kME).  This module brings those steps to BSVAE so the two
methods produce comparable outputs.

Label convention
----------------
Integer labels; ``-1`` (``GREY``) means unassigned.  On disk ``modules.csv``
carries ``feature_id, module, module_label`` where ``module_label`` is
``"grey"`` for -1 and ``"M<k>"`` otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GREY = -1


@dataclass
class PostprocessResult:
    labels: np.ndarray                      # (F,) int, -1 = grey
    gamma: np.ndarray                       # (F, M_final) merged soft assignments (rows sum to <= 1)
    eigengenes: Optional[pd.DataFrame]      # samples × modules (int columns)
    kme: Optional[pd.DataFrame]             # features × modules
    merge_log: List[dict]
    module_map: Dict[int, int]              # original component → final module (-1 = dissolved)
    n_grey: int
    params: dict = field(default_factory=dict)

    @property
    def n_modules(self) -> int:
        return int(len(set(self.labels[self.labels >= 0])))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def labels_to_names(labels: np.ndarray) -> List[str]:
    return ["grey" if int(k) < 0 else f"M{int(k)}" for k in labels]


def assign_with_confidence(gamma: np.ndarray, min_gamma: float = 0.5) -> np.ndarray:
    """argmax γ, but ``GREY`` wherever the top responsibility is below ``min_gamma``."""
    gamma = np.asarray(gamma, dtype=np.float64)
    labels = gamma.argmax(axis=1).astype(np.int64)
    conf = gamma.max(axis=1)
    labels[conf < float(min_gamma)] = GREY
    return labels


def enforce_min_module_size(
    labels: np.ndarray,
    gamma: np.ndarray,
    min_module_size: int = 10,
    min_gamma: float = 0.5,
    reassign: str = "next_best",
) -> np.ndarray:
    """
    Dissolve modules smaller than ``min_module_size`` (smallest first).

    Members of a dissolved module go to the live component with the highest
    γ (renormalised over live components) if that exceeds ``min_gamma``,
    otherwise to ``GREY``.  ``reassign="grey"`` sends them all to ``GREY``.
    """
    labels = np.asarray(labels).astype(np.int64).copy()
    gamma = np.asarray(gamma, dtype=np.float64)
    live = {int(k) for k in np.unique(labels) if k >= 0}
    while True:
        counts = {k: int((labels == k).sum()) for k in live}
        small = [k for k, c in counts.items() if c < int(min_module_size)]
        if not small:
            break
        k = min(small, key=lambda kk: (counts[kk], kk))
        members = np.where(labels == k)[0]
        live.discard(k)
        if reassign == "grey" or not live:
            labels[members] = GREY
            continue
        live_idx = np.array(sorted(live))
        sub = gamma[np.ix_(members, live_idx)]
        denom = sub.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        sub = sub / denom
        best = live_idx[sub.argmax(axis=1)]
        conf = sub.max(axis=1)
        labels[members] = np.where(conf >= float(min_gamma), best, GREY)
    return labels


def _relabel_compact(labels: np.ndarray) -> tuple[np.ndarray, Dict[int, int]]:
    """Map live labels to 0..M-1 (sorted by original id); returns (labels, old→new)."""
    old_ids = sorted(int(k) for k in np.unique(labels) if k >= 0)
    mapping = {k: i for i, k in enumerate(old_ids)}
    out = np.full_like(labels, GREY)
    for k, i in mapping.items():
        out[labels == k] = i
    return out, mapping


# ---------------------------------------------------------------------------
# Eigengenes and kME
# ---------------------------------------------------------------------------

def _zscore_rows(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    return (x - mu) / sd


def _first_pc(Z: np.ndarray, align: bool = True, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    First principal component scores (unit norm, like WGCNA's svd u[,1]) of a
    samples × features matrix of standardised profiles.  ``weights`` (per
    feature) scale the columns before the SVD.  The sign is chosen so the
    component correlates positively with the (weighted) mean profile.
    """
    Zc = Z - Z.mean(axis=0, keepdims=True)
    if weights is not None:
        Zc = Zc * np.sqrt(np.asarray(weights, dtype=np.float64))[np.newaxis, :]
    U, S, _ = np.linalg.svd(Zc, full_matrices=False)
    pc = U[:, 0]
    if align:
        ref = Zc.mean(axis=1)
        if np.dot(pc, ref) < 0:
            pc = -pc
    return pc


def module_eigengenes(
    expr: pd.DataFrame,
    labels: np.ndarray,
    feature_ids: Sequence[str],
    align: bool = True,
) -> pd.DataFrame:
    """
    Module eigengenes (first PC of the standardised member profiles).

    Parameters
    ----------
    expr : features × samples DataFrame (index = feature IDs)
    labels : (F,) int labels aligned to ``feature_ids``; -1 skipped
    align : flip each eigengene so it correlates positively with the module's
        mean standardised expression (WGCNA ``align="along average"``).

    Returns
    -------
    DataFrame samples × modules with integer module columns; modules with
    fewer than two features present in ``expr`` are NaN columns.
    """
    labels = np.asarray(labels).astype(np.int64)
    idx = expr.index.get_indexer(pd.Index([str(f) for f in feature_ids]))
    values = expr.values.astype(np.float64)
    out: Dict[int, np.ndarray] = {}
    for k in sorted(int(k) for k in np.unique(labels) if k >= 0):
        rows = idx[(labels == k) & (idx >= 0)]
        if len(rows) < 2:
            out[k] = np.full(values.shape[1], np.nan)
            continue
        Z = _zscore_rows(values[rows]).T          # samples × features
        out[k] = _first_pc(Z, align=align)
    df = pd.DataFrame(out, index=expr.columns)
    df.index.name = "sample_id"
    return df


def compute_kme(
    expr: pd.DataFrame,
    eigengenes: pd.DataFrame,
    feature_ids: Sequence[str],
) -> pd.DataFrame:
    """Module membership kME = Pearson cor(feature profile, module eigengene)."""
    X = expr.reindex([str(f) for f in feature_ids]).values.astype(np.float64)   # F × S
    Z = _zscore_rows(np.nan_to_num(X, nan=0.0))
    Z[np.isnan(X).any(axis=1)] = np.nan
    E = eigengenes.values.astype(np.float64)                                       # S × M
    E = (E - E.mean(axis=0, keepdims=True)) / np.where(
        E.std(axis=0, ddof=1, keepdims=True) == 0, 1.0, E.std(axis=0, ddof=1, keepdims=True)
    )
    S = X.shape[1]
    kme = Z @ E / max(S - 1, 1)
    return pd.DataFrame(kme, index=[str(f) for f in feature_ids], columns=eigengenes.columns)


def merge_close_modules(
    expr: pd.DataFrame,
    labels: np.ndarray,
    feature_ids: Sequence[str],
    cut_height: float = 0.4,
    max_iter: int = 50,
) -> tuple[np.ndarray, Dict[int, int], List[dict]]:
    """
    Merge modules whose eigengenes are closer than ``cut_height`` in
    1 − cor (average-linkage), iterating until no merge happens
    (analogue of WGCNA ``mergeCloseModules``).

    Returns
    -------
    labels : compact labels 0..M-1 (grey stays -1)
    module_map : input module id → output module id (-1 if none)
    merge_log : list of {"step", "merged", "into", "cor"}
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    labels = np.asarray(labels).astype(np.int64).copy()
    original_ids = sorted(int(k) for k in np.unique(labels) if k >= 0)
    current_of = {k: k for k in original_ids}          # original → current id
    log: List[dict] = []
    for step in range(int(max_iter)):
        me = module_eigengenes(expr, labels, feature_ids)
        me = me.dropna(axis=1, how="any")
        if me.shape[1] < 2:
            break
        ids = [int(c) for c in me.columns]
        C = np.corrcoef(me.values.T)
        D = np.clip(1.0 - C, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        Zl = linkage(squareform(D, checks=False), method="average")
        clusters = fcluster(Zl, t=float(cut_height), criterion="distance")
        merged_any = False
        for c in np.unique(clusters):
            members = [ids[i] for i in np.where(clusters == c)[0]]
            if len(members) < 2:
                continue
            sizes = {m: int((labels == m).sum()) for m in members}
            target = max(members, key=lambda m: (sizes[m], -m))
            for m in members:
                if m == target:
                    continue
                cor = float(C[ids.index(m), ids.index(target)])
                labels[labels == m] = target
                for orig, cur in current_of.items():
                    if cur == m:
                        current_of[orig] = target
                log.append({"step": step, "merged": m, "into": target, "cor": cor})
                merged_any = True
        if not merged_any:
            break

    labels, compact = _relabel_compact(labels)
    module_map = {orig: compact.get(cur, GREY) for orig, cur in current_of.items()}
    return labels, module_map, log


def kme_reassign(
    expr: pd.DataFrame,
    labels: np.ndarray,
    feature_ids: Sequence[str],
    min_kme: float = 0.3,
    kme_margin: float = 0.1,
    eigengenes: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    One kME pass: a feature whose best module membership is below
    ``min_kme`` becomes grey; a feature whose best other-module kME exceeds
    its own by more than ``kme_margin`` moves to that module.
    """
    labels = np.asarray(labels).astype(np.int64).copy()
    if eigengenes is None:
        eigengenes = module_eigengenes(expr, labels, feature_ids)
    eigengenes = eigengenes.dropna(axis=1, how="any")
    if eigengenes.shape[1] == 0:
        return labels
    kme = compute_kme(expr, eigengenes, feature_ids)
    cols = [int(c) for c in kme.columns]
    col_pos = {c: i for i, c in enumerate(cols)}
    K = kme.values
    for i in np.where(labels >= 0)[0]:
        own_mod = int(labels[i])
        if own_mod not in col_pos:
            continue
        row = K[i]
        if not np.isfinite(row).all():
            continue
        own = row[col_pos[own_mod]]
        j = int(row.argmax())
        best = row[j]
        if best < float(min_kme):
            labels[i] = GREY                     # fits nowhere
        elif cols[j] != own_mod and best > own + float(kme_margin):
            labels[i] = cols[j]                  # fits clearly better elsewhere
    return labels


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _merge_gamma(gamma: np.ndarray, module_map: Dict[int, int], n_final: int) -> np.ndarray:
    """
    Sum γ columns that map to the same final module.  Mass on dissolved
    components is dropped without renormalising, so rows sum to at most 1
    and the confidence of a feature is not inflated by components that no
    longer exist.
    """
    out = np.zeros((gamma.shape[0], n_final), dtype=np.float32)
    for orig, final in module_map.items():
        if final >= 0 and orig < gamma.shape[1]:
            out[:, final] += gamma[:, orig]
    return out


def postprocess_assignments(
    gamma: np.ndarray,
    feature_ids: Sequence[str],
    expr: Optional[pd.DataFrame] = None,
    *,
    min_gamma: float = 0.5,
    min_module_size: int = 10,
    merge_cut_height: float = 0.4,
    kme_reassign_: bool = False,
    min_kme: float = 0.3,
    kme_margin: float = 0.1,
) -> PostprocessResult:
    """
    confidence cutoff → minimum module size → (with ``expr``) eigengene
    merging → (optional) kME reassignment → final eigengenes and kME.

    ``gamma`` in the result sums the responsibilities of merged components
    (rows sum to at most 1; mass on dissolved components is dropped).

    ``expr`` is a features × samples DataFrame; without it merging is
    skipped and ``eigengenes``/``kme`` are ``None``.
    """
    gamma = np.asarray(gamma, dtype=np.float64)
    feature_ids = [str(f) for f in feature_ids]
    K = gamma.shape[1]
    params = dict(
        min_gamma=min_gamma, min_module_size=min_module_size,
        merge_cut_height=merge_cut_height, kme_reassign=kme_reassign_,
        min_kme=min_kme, kme_margin=kme_margin,
    )

    # Confidence cutoff on the raw responsibilities before merging: greyed
    # features are kept out of the eigengenes, which makes merging cleaner.
    # (Re-evaluating the cutoff on the merged γ was tested and rescued mostly
    # ambiguous features, lowering assigned-feature ARI on every seed.)
    labels = assign_with_confidence(gamma, min_gamma=min_gamma)
    labels = enforce_min_module_size(labels, gamma, min_module_size=min_module_size, min_gamma=min_gamma)
    merge_log: List[dict] = []

    if expr is not None:
        expr = expr.copy()
        expr.index = expr.index.astype(str)

    if expr is not None and merge_cut_height > 0:
        labels, module_map, merge_log = merge_close_modules(
            expr, labels, feature_ids, cut_height=merge_cut_height,
        )
        module_map = {k: module_map.get(k, GREY) for k in range(K)}
    else:
        labels, compact = _relabel_compact(labels)
        module_map = {k: compact.get(k, GREY) for k in range(K)}

    n_final = max(module_map.values(), default=-1) + 1
    gamma_merged = _merge_gamma(gamma, module_map, n_final) if n_final > 0 else np.zeros((len(labels), 0), np.float32)

    eigengenes = kme = None
    if expr is not None:
        if kme_reassign_ and n_final > 0:
            labels = kme_reassign(expr, labels, feature_ids, min_kme=min_kme, kme_margin=kme_margin)
            labels = enforce_min_module_size(
                labels, gamma_merged, min_module_size=min_module_size, min_gamma=min_gamma,
            )
        eigengenes = module_eigengenes(expr, labels, feature_ids)
        kme = compute_kme(expr, eigengenes, feature_ids) if eigengenes.shape[1] else None

    n_grey = int((labels == GREY).sum())
    logger.info(
        "Post-processing: %d modules, %d grey features (%.1f%%), %d merges",
        len(set(labels[labels >= 0])), n_grey, 100.0 * n_grey / max(len(labels), 1), len(merge_log),
    )
    return PostprocessResult(
        labels=labels, gamma=gamma_merged, eigengenes=eigengenes, kme=kme,
        merge_log=merge_log, module_map=module_map, n_grey=n_grey, params=params,
    )


def save_postprocess(result: PostprocessResult, output_dir: str | Path, feature_ids: Sequence[str]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature_id": [str(f) for f in feature_ids],
            "module": result.labels.astype(int),
            "module_label": labels_to_names(result.labels),
        }
    ).to_csv(out / "modules.csv", index=False)
    if result.eigengenes is not None:
        eg = result.eigengenes.copy()
        eg.columns = [f"ME{c}" for c in eg.columns]
        eg.to_csv(out / "eigengenes.csv")
    if result.kme is not None:
        kme = result.kme.copy()
        kme.columns = [f"kME{c}" for c in kme.columns]
        kme.index.name = "feature_id"
        kme.to_csv(out / "kME.csv")
    np.savez_compressed(
        out / "gamma_merged.npz", gamma=result.gamma,
        feature_ids=np.array([str(f) for f in feature_ids], dtype=object),
    )
    with open(out / "merge_log.json", "w") as f:
        json.dump(
            {
                "steps": result.merge_log,
                "module_map": {str(k): int(v) for k, v in result.module_map.items()},
                "n_grey": result.n_grey,
                "n_modules": result.n_modules,
                "params": result.params,
            },
            f, indent=2,
        )
    logger.info("Saved post-processed modules to %s", out)


__all__ = [
    "GREY",
    "PostprocessResult",
    "assign_with_confidence",
    "enforce_min_module_size",
    "module_eigengenes",
    "compute_kme",
    "merge_close_modules",
    "kme_reassign",
    "postprocess_assignments",
    "save_postprocess",
    "labels_to_names",
]
