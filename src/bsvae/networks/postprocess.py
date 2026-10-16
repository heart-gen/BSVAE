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
from typing import Tuple, Dict, List, Optional, Sequence

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


def enforce_gene_coherence(
    labels: np.ndarray,
    expr: pd.DataFrame,
    feature_ids: Sequence[str],
    tx2gene: "pd.Series | Dict[str, str]",
    *,
    min_kme: float = 0.3,
    alpha: float = 0.05,
    keep_version: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Keep an isoform in a module different from its gene's majority module only
    when its gene-programme-removed residual supports that membership.

    Isoforms of one gene share the gene's expression programme, so a noisy or
    low-abundance isoform often lands in a neighbouring module by abundance
    alone (abundance confounding).  For every assigned isoform whose label
    differs from its gene's majority label, the sibling-residual kME to its
    own module (eigengene residualised on the gene's module eigengene, as in
    :func:`bsvae.utils.isoform.isoform_specific_calls`) is tested; isoforms
    whose |kME| is below ``min_kme`` or not significant at BH FDR ``alpha``
    are moved to the gene's majority module.  Grey isoforms are untouched.

    Returns the corrected labels and a table of the candidates with
    ``gene_id, gene_module, module, kme_residual, p_value, q_value, kept``.
    """
    from collections import Counter

    from scipy import stats as sps

    from bsvae.utils.isoform import (
        _bh_adjust, _residualize_eigengenes, isoform_residuals, map_transcripts_to_genes,
    )

    fids = [str(f) for f in feature_ids]
    labels = np.asarray(labels).astype(np.int64).copy()
    expr = expr.copy(); expr.index = expr.index.astype(str)
    tx_gene = map_transcripts_to_genes(fids, tx2gene, keep_version)
    lab = pd.Series(labels, index=fids)
    gene_mod: Dict[str, int] = {}
    for g, idx in tx_gene.dropna().groupby(tx_gene.dropna()).groups.items():
        votes = [int(v) for v in lab.loc[list(idx)] if v >= 0]
        gene_mod[g] = Counter(votes).most_common(1)[0][0] if votes else -1
    gm = np.array([gene_mod.get(g, -1) if isinstance(g, str) else -1 for g in tx_gene.values])
    cand = np.where((labels >= 0) & (gm >= 0) & (labels != gm))[0]
    columns = ["gene_id", "gene_module", "module", "kme_residual", "p_value", "q_value", "kept"]
    if cand.size == 0:
        return labels, pd.DataFrame(columns=columns)
    eig = module_eigengenes(expr, labels, fids).dropna(axis=1, how="any")
    cols = [int(c) for c in eig.columns]
    resid, _ = isoform_residuals(expr, tx2gene, keep_version=keep_version)
    S = expr.shape[1]
    kme = np.full(cand.size, np.nan)
    for g in np.unique(gm[cand]):
        rows = cand[gm[cand] == g]
        E = _residualize_eigengenes(eig, cols.index(int(g))) if g in cols else eig
        ids = [fids[i] for i in rows]
        have = [t for t in ids if t in resid.index]
        if not have:
            continue
        K = compute_kme(resid, E, have)
        for i, t in zip(rows, ids):
            if t in K.index and labels[i] in cols:
                kme[np.where(cand == i)[0][0]] = float(K.loc[t, E.columns[cols.index(int(labels[i]))]])
    r2 = np.minimum(np.nan_to_num(kme) ** 2, 1 - 1e-12)
    tval = np.abs(np.nan_to_num(kme)) * np.sqrt(max(S - 2, 1) / (1 - r2))
    p = np.minimum(2.0 * sps.t.sf(tval, max(S - 2, 1)) * max(len(cols) - 1, 1), 1.0)
    p[np.isnan(kme)] = 1.0
    q = _bh_adjust(p)
    kept = (np.abs(np.nan_to_num(kme)) >= float(min_kme)) & (q < float(alpha))
    out = pd.DataFrame({"gene_id": tx_gene.values[cand], "gene_module": gm[cand], "module": labels[cand],
                        "kme_residual": kme, "p_value": p, "q_value": q, "kept": kept},
                       index=[fids[i] for i in cand])
    out.index.name = "feature_id"
    labels[cand[~kept]] = gm[cand[~kept]]
    logger.info("Gene coherence: %d divergent isoforms tested, %d kept, %d moved to their gene's module",
                cand.size, int(kept.sum()), int((~kept).sum()))
    return labels, out


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
    argmax → minimum module size → (with ``expr``) eigengene merging →
    confidence cutoff on the *merged* responsibilities → minimum module size
    → (optional) kME reassignment → final eigengenes and kME.

    ``gamma`` in the result sums the responsibilities of merged components
    (rows sum to at most 1; mass on dissolved components is dropped), and
    ``min_gamma`` is applied to those sums, so a feature spread across
    components of one over-clustered module is judged by their total.

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

    # argmax over components, then merge; the confidence cutoff is applied to
    # the merged responsibilities below.  With an over-clustered GMM a module
    # member's responsibility is spread over several components of the same
    # module, so judging confidence per component would grey out members
    # that are confident once those components are merged.
    labels = gamma.argmax(axis=1).astype(np.int64)
    labels = enforce_min_module_size(labels, gamma, min_module_size=min_module_size, min_gamma=0.0)
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
    if n_final > 0:
        gamma_merged = _merge_gamma(gamma, module_map, n_final)
        labels = assign_with_confidence(gamma_merged, min_gamma=min_gamma)
        labels = enforce_min_module_size(labels, gamma_merged, min_module_size=min_module_size, min_gamma=min_gamma)
        labels, compact = _relabel_compact(labels)
        module_map = {k: (compact.get(v, GREY) if v >= 0 else GREY) for k, v in module_map.items()}
        n_final = max(module_map.values(), default=-1) + 1
        gamma_merged = _merge_gamma(gamma, module_map, n_final) if n_final > 0 else np.zeros((len(labels), 0), np.float32)
    else:
        gamma_merged = np.zeros((len(labels), 0), np.float32)

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


__all__ = ["enforce_gene_coherence", 
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


# ---------------------------------------------------------------------------
# Consensus over runs (seeds)
# ---------------------------------------------------------------------------

def align_labels(reference: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Relabel ``labels`` so that its modules match ``reference`` modules by
    maximum overlap (Hungarian assignment on the contingency table).  Modules
    without a match keep fresh ids above the reference range; grey stays -1.
    """
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference).astype(np.int64)
    labels = np.asarray(labels).astype(np.int64)
    ref_ids = sorted(int(k) for k in np.unique(reference) if k >= 0)
    lab_ids = sorted(int(k) for k in np.unique(labels) if k >= 0)
    out = np.full_like(labels, GREY)
    if not ref_ids or not lab_ids:
        if lab_ids:
            for i, k in enumerate(lab_ids):
                out[labels == k] = i
        return out
    table = np.zeros((len(lab_ids), len(ref_ids)), dtype=np.int64)
    for i, a in enumerate(lab_ids):
        la = labels == a
        for j, b in enumerate(ref_ids):
            table[i, j] = int((la & (reference == b)).sum())
    rows, cols = linear_sum_assignment(-table)
    mapping: Dict[int, int] = {}
    for i, j in zip(rows, cols):
        if table[i, j] > 0:
            mapping[lab_ids[i]] = ref_ids[j]
    next_id = (max(ref_ids) + 1) if ref_ids else 0
    for a in lab_ids:
        if a not in mapping:
            mapping[a] = next_id
            next_id += 1
    for a, b in mapping.items():
        out[labels == a] = b
    return out


@dataclass
class ConsensusResult:
    labels: np.ndarray            # (F,) int, -1 = grey (no majority)
    agreement: np.ndarray         # (F,) fraction of runs agreeing with the consensus label
    aligned: np.ndarray           # (R, F) runs after alignment
    run_ari: List[float]          # ARI of each run against the consensus
    n_runs: int


def consensus_assignments(
    label_runs: Sequence[np.ndarray],
    min_agreement: float = 0.5,
    min_module_size: int = 10,
) -> ConsensusResult:
    """
    Consensus module assignment across several runs (e.g. seeds).

    Runs are aligned one at a time to the running consensus by maximum
    overlap, then each feature takes the label chosen by the most runs
    (grey votes count).  Features whose winning label is supported by fewer
    than ``min_agreement`` of the runs are grey, and modules that end up
    smaller than ``min_module_size`` are dissolved to grey.  The per-feature
    ``agreement`` doubles as a stability score.
    """
    from sklearn.metrics import adjusted_rand_score

    runs = [np.asarray(r).astype(np.int64) for r in label_runs]
    if not runs:
        raise ValueError("consensus_assignments needs at least one run")
    F = len(runs[0])
    if any(len(r) != F for r in runs):
        raise ValueError("all runs must have the same number of features")
    aligned = [_relabel_compact(runs[0])[0]]
    consensus = aligned[0].copy()
    for r in runs[1:]:
        a = align_labels(consensus, r)
        aligned.append(a)
        consensus = _vote(np.stack(aligned))[0]
    A = np.stack(aligned)
    labels, agreement = _vote(A)
    labels[agreement < float(min_agreement)] = GREY
    if min_module_size > 1:
        for k in [k for k in np.unique(labels) if k >= 0]:
            if (labels == k).sum() < int(min_module_size):
                labels[labels == k] = GREY
    labels, _ = _relabel_compact(labels)
    run_ari = [float(adjusted_rand_score(labels, a)) for a in A]
    return ConsensusResult(labels=labels, agreement=agreement, aligned=A, run_ari=run_ari, n_runs=len(runs))


def _vote(aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature majority label and the fraction of runs supporting it."""
    R, F = aligned.shape
    labels = np.full(F, GREY, dtype=np.int64)
    agreement = np.zeros(F, dtype=np.float64)
    for i in range(F):
        vals, counts = np.unique(aligned[:, i], return_counts=True)
        j = int(counts.argmax())
        labels[i] = int(vals[j])
        agreement[i] = counts[j] / R
    return labels, agreement


def save_consensus(result: ConsensusResult, output_dir: str | Path, feature_ids: Sequence[str]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feature_id": [str(f) for f in feature_ids],
            "module": result.labels.astype(int),
            "module_label": labels_to_names(result.labels),
            "agreement": np.round(result.agreement, 4),
        }
    ).to_csv(out / "consensus_modules.csv", index=False)
    with open(out / "consensus_summary.json", "w") as f:
        json.dump(
            {"n_runs": result.n_runs, "run_ari_to_consensus": result.run_ari,
             "n_modules": int(len(set(result.labels[result.labels >= 0]))),
             "n_grey": int((result.labels == GREY).sum()),
             "mean_agreement": float(result.agreement.mean())},
            f, indent=2,
        )


__all__ += ["align_labels", "consensus_assignments", "ConsensusResult", "save_consensus"]
