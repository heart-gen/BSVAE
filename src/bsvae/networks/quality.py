"""
Module quality: membership (kME) summaries, enrichment, and preservation.

``module_preservation`` is a lightweight analogue of WGCNA's
``modulePreservation`` Z-summary: for each reference module it asks whether
the module's members are still (a) more tightly correlated in a test dataset
than random feature sets of the same size (density, ``Z_density``) and (b)
ranked by membership the same way (connectivity, ``cor_kME``), reporting a
permutation Z-score for the density term.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from bsvae.networks.postprocess import GREY, _zscore_rows, compute_kme, module_eigengenes

logger = logging.getLogger(__name__)


def module_quality(expr: pd.DataFrame, labels: np.ndarray, feature_ids: Sequence[str]) -> pd.DataFrame:
    """
    Per-module summary: size, mean within-module correlation, eigengene
    variance explained, and the distribution of members' own-module kME.
    """
    labels = np.asarray(labels).astype(np.int64)
    expr = expr.copy(); expr.index = expr.index.astype(str)
    fids = [str(f) for f in feature_ids]
    eg = module_eigengenes(expr, labels, fids)
    kme = compute_kme(expr, eg, fids)
    X = expr.reindex(fids).values.astype(np.float64)
    Z = _zscore_rows(np.nan_to_num(X))
    rows = []
    for k in sorted(int(k) for k in np.unique(labels) if k >= 0):
        members = np.where(labels == k)[0]
        if len(members) < 2:
            continue
        Zm = Z[members]
        C = (Zm @ Zm.T) / max(Zm.shape[1] - 1, 1)
        mean_cor = float((C.sum() - np.trace(C)) / (len(members) * (len(members) - 1)))
        own = kme.iloc[members][k].values if k in kme.columns else np.full(len(members), np.nan)
        # Variance explained by the eigengene ≈ mean squared kME.
        rows.append({
            "module": k, "size": int(len(members)), "mean_within_cor": mean_cor,
            "var_explained": float(np.nanmean(own ** 2)),
            "kme_mean": float(np.nanmean(own)), "kme_median": float(np.nanmedian(own)),
            "kme_q10": float(np.nanquantile(own, 0.1)), "frac_kme_below_0.3": float(np.nanmean(own < 0.3)),
        })
    out = pd.DataFrame(rows)
    out["n_grey_total"] = int((labels == GREY).sum())
    return out


def module_preservation(
    expr_ref: pd.DataFrame,
    expr_test: pd.DataFrame,
    labels: np.ndarray,
    feature_ids: Sequence[str],
    n_perm: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Preservation of reference modules in a test dataset.

    For each module (defined on ``expr_ref`` members ``feature_ids``/``labels``):

    * ``density_test``: mean within-module correlation in the test data.
    * ``Z_density``: (density_test − mean of random same-size sets) / sd over
      ``n_perm`` random sets drawn from the shared features; > 2 is weak,
      > 10 strong preservation (WGCNA conventions for Zsummary).
    * ``cor_kME``: correlation between members' kME in the reference and in
      the test data (connectivity preservation).
    * ``n_shared``: members present in both datasets.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(np.int64)
    fids = np.array([str(f) for f in feature_ids])
    ref = expr_ref.copy(); ref.index = ref.index.astype(str)
    test = expr_test.copy(); test.index = test.index.astype(str)
    shared = [f for f in fids if f in ref.index and f in test.index]
    if len(shared) < 10:
        raise ValueError("fewer than 10 features shared between the two datasets")
    shared_set = set(shared)
    keep = np.array([f in shared_set for f in fids])
    fids_s, labels_s = fids[keep], labels[keep]
    Zt = _zscore_rows(test.reindex(fids_s).values.astype(np.float64))
    St = Zt.shape[1]
    eg_ref = module_eigengenes(ref, labels_s, fids_s)
    kme_ref = compute_kme(ref, eg_ref, fids_s)
    eg_test = module_eigengenes(test, labels_s, fids_s)
    kme_test = compute_kme(test, eg_test, fids_s)

    def density(idx: np.ndarray) -> float:
        Zm = Zt[idx]
        C = (Zm @ Zm.T) / max(St - 1, 1)
        n = len(idx)
        return float((C.sum() - np.trace(C)) / (n * (n - 1)))

    rows = []
    n_total = len(fids_s)
    for k in sorted(int(k) for k in np.unique(labels_s) if k >= 0):
        members = np.where(labels_s == k)[0]
        if len(members) < 3:
            continue
        d = density(members)
        null = np.array([density(rng.choice(n_total, size=len(members), replace=False)) for _ in range(int(n_perm))])
        z = (d - null.mean()) / (null.std(ddof=1) + 1e-12)
        if k in kme_ref.columns and k in kme_test.columns:
            a = kme_ref.iloc[members][k].values; b = kme_test.iloc[members][k].values
            ok = np.isfinite(a) & np.isfinite(b)
            cor_k = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() >= 3 else float("nan")
        else:
            cor_k = float("nan")
        rows.append({"module": k, "n_shared": int(len(members)), "density_test": d,
                     "density_null_mean": float(null.mean()), "Z_density": float(z), "cor_kME": cor_k})
    return pd.DataFrame(rows)


def module_enrichment(
    labels: np.ndarray,
    feature_ids: Sequence[str],
    annotation_sets: Dict[str, Sequence[str]],
    min_overlap: int = 3,
    fdr: bool = True,
) -> pd.DataFrame:
    """
    Fisher enrichment of annotation sets per module (grey excluded) as a tidy
    table with Benjamini–Hochberg adjusted p-values.
    """
    from bsvae.simulation.metrics import compute_module_enrichment_score

    labels = np.asarray(labels).astype(np.int64)
    keep = labels >= 0
    fids = [str(f) for f, m in zip(feature_ids, keep) if m]
    res = compute_module_enrichment_score(labels[keep], {k: [str(x) for x in v] for k, v in annotation_sets.items()},
                                          fids, min_overlap=min_overlap)
    rows = []
    for mod, anns in res.items():
        for name, st in anns.items():
            rows.append({"module": int(mod), "annotation": name, **st})
    df = pd.DataFrame(rows)
    if fdr and not df.empty:
        p = df["p_value"].values
        order = np.argsort(p)
        ranked = np.empty_like(p)
        n = len(p)
        q = p[order] * n / (np.arange(n) + 1)
        q = np.minimum.accumulate(q[::-1])[::-1]
        ranked[order] = np.minimum(q, 1.0)
        df["q_value"] = ranked
        df = df.sort_values(["q_value", "p_value"]).reset_index(drop=True)
    return df


__all__ = ["module_quality", "module_preservation", "module_enrichment"]
