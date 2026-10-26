"""
Isoform-level helpers: gene aggregates and isoform-specific residuals.

Transcripts of one gene share the gene's transcriptional programme, so their
profiles are dominated by a common component.  ``isoform_residuals`` removes
that component by regressing each isoform's (log-scale) profile on its gene's
aggregate profile; what remains is the isoform-specific co-expression
(splicing-, promoter- or stability-driven), which can be clustered with the
ordinary BSVAE pipeline to obtain *isoform-specific modules*.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bsvae.utils.hierarchy import strip_feature_version

logger = logging.getLogger(__name__)


def map_transcripts_to_genes(
    feature_ids: Sequence[str],
    tx2gene: "pd.Series | Dict[str, str]",
    keep_version: bool = False,
) -> pd.Series:
    """Series feature_id → gene_id (NaN when unmapped), with version handling."""
    mapping = {strip_feature_version(k, keep_version): str(v) for k, v in dict(tx2gene).items()}
    genes = [mapping.get(strip_feature_version(f, keep_version)) for f in feature_ids]
    return pd.Series(genes, index=[str(f) for f in feature_ids], name="gene_id")


def gene_aggregate(
    expr: pd.DataFrame,
    tx_gene: pd.Series,
    log_base: float = 2.0,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Gene-level profiles (genes × samples) from a log-scale transcript matrix:
    isoforms are summed on the linear scale and re-logged.
    """
    genes = tx_gene.reindex(expr.index.astype(str))
    keep = genes.notna().values
    lin = np.power(log_base, expr.values[keep]) - pseudocount
    lin = np.maximum(lin, 0.0)
    agg = pd.DataFrame(lin, index=genes.values[keep], columns=expr.columns).groupby(level=0).sum()
    return np.log(agg + pseudocount) / np.log(log_base)


def isoform_residuals(
    expr: pd.DataFrame,
    tx2gene: "pd.Series | Dict[str, str]",
    min_isoforms: int = 2,
    keep_version: bool = False,
    leave_one_out: bool = True,
    input_scale: str = "log2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regress each isoform's profile on its gene's aggregate profile and return
    the residual matrix (multi-isoform transcripts × samples) plus per-transcript
    statistics (gene_id, n_isoforms, slope, r2_gene, residual_sd).

    Residuals are what the gene-level programme does not explain, so
    clustering them finds isoform-specific co-expression; transcripts of
    single-isoform genes are excluded (their residual is identically zero).

    With ``leave_one_out`` (default) the gene profile each isoform is regressed
    on is the sum of its *sibling* isoforms, so an isoform that carries a large
    share of its gene does not regress out its own signal.  ``False`` uses the
    full gene aggregate (the previous behaviour).  ``input_scale`` is
    ``"log2"`` for log2(x + 1) matrices (isoforms are summed on the linear
    scale) or ``"linear"`` for matrices whose values add directly, such as
    Pearson residuals.
    """
    if input_scale not in ("log2", "linear"):
        raise ValueError("input_scale must be 'log2' or 'linear'")
    expr = expr.copy()
    expr.index = expr.index.astype(str)
    tx_gene = map_transcripts_to_genes(expr.index, tx2gene, keep_version)
    counts = tx_gene.value_counts()
    multi = tx_gene.map(counts).fillna(0) >= int(min_isoforms)
    sub = expr.loc[multi.values]
    genes = tx_gene.loc[sub.index]
    X = sub.values.astype(np.float64)
    if input_scale == "log2":
        agg = gene_aggregate(expr, tx_gene)
        if leave_one_out:
            lin_gene = np.power(2.0, agg.reindex(genes.values).values) - 1.0
            lin_self = np.maximum(np.power(2.0, X) - 1.0, 0.0)
            G = np.log2(np.maximum(lin_gene - lin_self, 0.0) + 1.0)   # siblings only
        else:
            G = agg.reindex(genes.values).values            # (T, S) gene profile per transcript
    else:
        keep = tx_gene.notna().values
        agg = pd.DataFrame(expr.values[keep], index=tx_gene.values[keep], columns=expr.columns).groupby(level=0).sum()
        G = agg.reindex(genes.values).values
        if leave_one_out:
            G = G - X

    Xc = X - X.mean(axis=1, keepdims=True)
    Gc = G - G.mean(axis=1, keepdims=True)
    var_g = (Gc ** 2).sum(axis=1)
    slope = np.where(var_g > 0, (Xc * Gc).sum(axis=1) / np.where(var_g > 0, var_g, 1.0), 0.0)
    resid = Xc - slope[:, None] * Gc
    var_x = (Xc ** 2).sum(axis=1)
    r2 = np.where(var_x > 0, 1.0 - (resid ** 2).sum(axis=1) / np.where(var_x > 0, var_x, 1.0), 0.0)

    residual_df = pd.DataFrame(resid, index=sub.index, columns=expr.columns)
    stats = pd.DataFrame(
        {
            "gene_id": genes.values,
            "n_isoforms": genes.map(counts).values.astype(int),
            "slope": slope,
            "r2_gene": r2,
            "residual_sd": resid.std(axis=1, ddof=1),
        },
        index=sub.index,
    )
    stats.index.name = "feature_id"
    logger.info(
        "Isoform residuals: %d transcripts in %d multi-isoform genes (%d single-isoform transcripts excluded)",
        len(sub), genes.nunique(), int((~multi).sum()),
    )
    return residual_df, stats


__all__ = ["map_transcripts_to_genes", "gene_aggregate", "isoform_residuals"]


def _zscore_rows_np(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, ddof=1, keepdims=True)
    return Xc / np.where(sd == 0, 1.0, sd)


def _standardize_cols_np(E: np.ndarray) -> np.ndarray:
    Ec = E - E.mean(axis=0, keepdims=True)
    sd = Ec.std(axis=0, ddof=1, keepdims=True)
    return Ec / np.where(sd == 0, 1.0, sd)


def _module_map(reference: np.ndarray, labels: np.ndarray) -> Dict[int, int]:
    """Module id in ``labels`` → id in ``reference``'s system, by maximum overlap."""
    from bsvae.networks.postprocess import align_labels

    aligned = align_labels(reference, labels)
    out: Dict[int, int] = {}
    for m in np.unique(labels):
        if m < 0:
            continue
        vals, counts = np.unique(aligned[labels == m], return_counts=True)
        out[int(m)] = int(vals[counts.argmax()])
    return out


def cross_fit_calls(
    expr: pd.DataFrame,
    tx2gene: "pd.Series | Dict[str, str]",
    labels_by_fold: Sequence[np.ndarray],
    fold_of_sample: Sequence[int],
    feature_ids: Sequence[str],
    min_kme: float = 0.3,
    alpha: Optional[float] = 0.05,
    require_module_agreement: bool = False,
    null: str = "analytic",
    **kwargs,
) -> pd.DataFrame:
    """
    Cross-fitted isoform-specific calls.

    ``labels_by_fold[f]`` are module labels learned on every sample *except*
    those with ``fold_of_sample == f``; each fold's samples are then scored
    with :func:`isoform_specific_calls` against labels that never saw them, so
    the selection of a transcript into a module by the clustering cannot
    inflate its own test (the residual winner's curse that the leave-one-
    gene-out eigengene does not remove).  Module ids of every fold are mapped
    onto fold 0's by maximum overlap.  For each transcript and each module the
    per-fold residual kME are turned into signed z-scores and combined with
    Stouffer's method (weights ``sqrt(n_f)``); the best module other than the
    gene's is then chosen on the *combined* statistic, Bonferroni-corrected
    over modules and BH-adjusted over transcripts.  Combining before
    selecting matters: selecting the best module in each fold and combining
    those maxima is anti-conservative.

    Returns one row per multi-isoform transcript with ``kme_residual`` (the
    sample-weighted mean of the chosen module's per-fold kME),
    ``z_combined``, ``p_value``, ``q_value``, ``best_other_module`` (fold-0
    ids), ``modules_agree`` (every fold's own best module is the combined
    one), ``label_discordant`` (the transcript's raw label differs from its
    gene's in any fold), per-fold ``kme_fold<f>`` columns and ``called``.
    ``require_module_agreement`` additionally demands ``modules_agree`` for a
    call (an annotation by default).  ``null="stratum"`` replaces the
    Bonferroni p of the combined statistic with the empirical null fitted per
    selection stratum (:func:`stratum_empirical_null` on |z|; scales in
    ``out.attrs["null_scale"]``); the per-fold scans always use the analytic
    form since only their kME enter the combination.
    """
    from scipy import stats as sps

    fold_of_sample = np.asarray(fold_of_sample)
    folds = sorted(int(f) for f in np.unique(fold_of_sample))
    if len(folds) != len(labels_by_fold):
        raise ValueError(f"{len(labels_by_fold)} label sets for {len(folds)} folds")
    kw = dict(kwargs); kw.pop("alpha", None); kw.pop("min_kme", None); kw.pop("return_kme", None)
    ref = np.asarray(labels_by_fold[0]).astype(np.int64)
    per_fold: List[pd.DataFrame] = []; kmes: List[pd.DataFrame] = []; n_per_fold: List[int] = []
    for f, lab in zip(folds, labels_by_fold):
        mask = fold_of_sample == f
        lab = np.asarray(lab).astype(np.int64)
        calls, k = isoform_specific_calls(expr.loc[:, mask], tx2gene, lab, feature_ids, min_kme=0.0, alpha=None,
                                          return_kme=True, **kw)
        mp = _module_map(ref, lab)           # injective (Hungarian; unmatched modules get fresh ids)
        k = k.copy(); k.columns = [mp.get(int(c), int(c)) for c in k.columns]
        calls = calls.copy(); calls["best_other_module"] = [mp.get(int(b), int(b)) for b in calls["best_other_module"].values]
        per_fold.append(calls); kmes.append(k); n_per_fold.append(int(mask.sum()))
    idx = per_fold[0].index
    for c in per_fold[1:]:
        idx = idx.intersection(c.index)
    cols = sorted(set().union(*[set(int(c) for c in k.columns) for k in kmes]))
    Z = np.zeros((len(idx), len(cols))); W = np.zeros((len(idx), len(cols)))
    KW = np.zeros((len(idx), len(cols))); NW = np.zeros((len(idx), len(cols)))
    fold_best: List[np.ndarray] = []; kme_cols: Dict[str, np.ndarray] = {}
    discordant = np.zeros(len(idx), bool)
    for f, (calls, k, n_f) in enumerate(zip(per_fold, kmes, n_per_fold)):
        k = k.reindex(index=idx, columns=cols)
        present = k.notna().values
        r = np.nan_to_num(k.values.astype(float))
        r2 = np.minimum(r ** 2, 1.0 - 1e-12)
        tval = np.abs(r) * np.sqrt(max(n_f - 2, 1) / (1.0 - r2))
        p_un = np.clip(2.0 * sps.t.sf(tval, max(n_f - 2, 1)), 1e-300, 1.0 - 1e-15)
        z = np.sign(r) * sps.norm.isf(p_un / 2.0)
        w = np.sqrt(n_f)
        Z += np.where(present, w * z, 0.0); W += np.where(present, w ** 2, 0.0)
        KW += np.where(present, n_f * r, 0.0); NW += np.where(present, n_f, 0.0)
        c = calls.reindex(idx)
        fold_best.append(np.array([cols.index(int(b)) if int(b) in cols else -1 for b in c["best_other_module"].values]))
        discordant |= c["label_discordant"].values.astype(bool)
    Zc = np.where(W > 0, Z / np.sqrt(np.maximum(W, 1e-12)), 0.0)
    best_j = np.abs(Zc).argmax(axis=1)
    rows = np.arange(len(idx))
    z_best = Zc[rows, best_j]
    n_mod = max(1, len(cols))
    null_scale: Dict[object, float] = {}
    if null == "analytic":
        p = np.minimum(2.0 * sps.norm.sf(np.abs(z_best)) * n_mod, 1.0)
    elif null == "stratum":
        p, null_scale = stratum_empirical_null(np.abs(z_best), discordant, n_mod)
    else:
        raise ValueError("null must be 'analytic' or 'stratum'")
    q = _bh_adjust(p)
    kme = KW[rows, best_j] / np.maximum(NW[rows, best_j], 1.0)
    agree = np.all(np.vstack(fold_best) == best_j[None, :], axis=0)
    called = np.abs(kme) >= float(min_kme)
    if alpha is not None:
        called &= q < float(alpha)
    if require_module_agreement:
        called &= agree
    c0 = per_fold[0].reindex(idx)
    for f, k in enumerate(kmes):
        kme_cols[f"kme_fold{f}"] = k.reindex(index=idx, columns=cols).values[rows, best_j]
    out = pd.DataFrame({
        "gene_id": c0["gene_id"].values,
        "gene_module": c0["gene_module"].values,
        "best_other_module": [cols[j] for j in best_j],
        "kme_residual": kme,
        "z_combined": z_best,
        "p_value": p,
        "q_value": q,
        "n_folds": len(folds),
        "modules_agree": agree,
        "label_discordant": discordant,
        "n_isoforms": c0["n_isoforms"].values,
        "mean_expr": expr.reindex(idx).mean(axis=1).values,
        **kme_cols,
        "called": called,
    }, index=idx)
    out.index.name = "feature_id"
    out.attrs["null_scale"] = null_scale
    logger.info("Cross-fitted isoform-specific calls over %d folds: %d of %d transcripts called",
                len(folds), int(called.sum()), len(out))
    return out


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.empty(n)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1]
    return np.minimum(q, 1.0)


def stratum_empirical_null(stat: np.ndarray, strata: np.ndarray, n_modules: "int | np.ndarray",
                           min_size: int = 50, levels: Tuple[float, ...] = (0.6, 0.8)) -> Tuple[np.ndarray, Dict[object, float]]:
    """
    Empirical null of a best-of-``n_modules`` statistic, fitted per selection stratum.

    ``stat`` is the absolute statistic of each transcript's best module (the
    |t| of the best residual kME, or the combined |z| of the cross-fit),
    ``strata`` the stratum each transcript was selected in (the callers pass
    ``label_discordant``: whether the transcript's own label is the module it
    is tested against).  Under the null ``stat / sigma_s`` is taken to be the
    maximum of ``n_modules`` independent |N(0, 1)|, with CDF
    ``(2 Phi(x) - 1) ** m``; ``sigma_s`` is fitted per stratum by least squares
    on the ``levels`` quantiles of ``stat`` (default the 60th and 80th
    percentiles).  The lower quantiles of a best-of-m statistic over
    correlated modules sit below the independent-module form, so anchoring the
    fit there (25th/50th) shrinks ``sigma_s`` and inflates the tail; anchors
    at the 90th percentile or above absorb real divergence once it exceeds
    about a tenth of a stratum.  The 60th/80th pair keeps the concordant
    stratum near the analytic calibration and tolerates up to ~15 % of
    real signal in the upper tail.  A stratum with fewer than
    ``min_size`` transcripts keeps ``sigma_s = 1``.  At ``sigma_s = 1`` the
    p-value is Sidak's ``1 - (1 - p_best) ** m`` rather than the Bonferroni
    ``min(1, m p_best)`` of the analytic null.

    Returns the p-values and the fitted scale per stratum.
    """
    from scipy import stats as sps

    a = np.abs(np.asarray(stat, dtype=np.float64))
    strata = np.asarray(strata)
    m = np.maximum(np.broadcast_to(np.asarray(n_modules, dtype=np.float64), a.shape), 1.0)
    levels = np.asarray(levels, dtype=np.float64)
    p = np.ones_like(a)
    scales: Dict[object, float] = {}
    for s in pd.unique(strata):
        idx = np.where(strata == s)[0]
        sigma = 1.0
        if idx.size >= int(min_size):
            ms = float(np.median(m[idx]))
            x = sps.norm.ppf((levels ** (1.0 / ms) + 1.0) / 2.0)      # null quantiles at unit scale
            q = np.quantile(a[idx], levels)
            sigma = float((q * x).sum() / (x * x).sum())
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
        scales[s.item() if hasattr(s, "item") else s] = sigma
        sf = np.minimum(sps.norm.sf(a[idx] / sigma), 0.5 - 1e-12)
        p[idx] = -np.expm1(m[idx] * np.log1p(-2.0 * sf))
    return np.clip(p, 0.0, 1.0), scales


__all__.append("stratum_empirical_null")


def _residualize_eigengenes(eigengenes: pd.DataFrame, column: int) -> pd.DataFrame:
    """Regress every eigengene on ``eigengenes[:, column]`` and return the residuals."""
    E = eigengenes.values.astype(np.float64)
    E = E - E.mean(axis=0, keepdims=True)
    g = E[:, column]
    denom = float((g ** 2).sum())
    if denom <= 0:
        return pd.DataFrame(E, index=eigengenes.index, columns=eigengenes.columns)
    beta = (E * g[:, None]).sum(axis=0) / denom
    R = E - g[:, None] * beta[None, :]
    R[:, column] = g
    return pd.DataFrame(R, index=eigengenes.index, columns=eigengenes.columns)


def isoform_specific_calls(
    expr: pd.DataFrame,
    tx2gene: "pd.Series | Dict[str, str]",
    labels: np.ndarray,
    feature_ids: Sequence[str],
    min_kme: float = 0.3,
    keep_version: bool = False,
    eigengenes: Optional[pd.DataFrame] = None,
    alpha: Optional[float] = 0.05,
    residualize_eigengenes: bool = True,
    leave_one_out: bool = True,
    input_scale: str = "log2",
    loo_eigengene: "bool | str" = "approx",
    labels_for_eigengenes: Optional[np.ndarray] = None,
    return_kme: bool = False,
    covariates: Optional[pd.DataFrame] = None,
    max_target_corr: Optional[float] = None,
    null: str = "analytic",
) -> "pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]":
    """
    Call isoforms whose *gene-programme-removed* profile belongs to a module
    other than their gene's ("isoform-specific membership").

    ``loo_eigengene`` guards against self-contribution: a transcript whose own
    raw label is the module it is tested against contributed to that module's
    eigengene, which inflates its kME.  ``"approx"`` (default) removes the
    gene's transcripts from the eigengene by a first-order downdate of the
    first principal component (the gene's loading-weighted profiles are
    subtracted from the eigengene scores; the loading vector itself is kept),
    which costs one matrix product per module.  ``"exact"`` (or ``True``)
    recomputes the module's PCA without the gene for every (gene, module)
    pair with a member, which is exact but slow transcriptome-wide.
    ``False`` disables the correction.  On abundance-only null simulations
    the type-I error at 0.05 for transcripts whose raw label is the tested
    module is 0.50 without the correction and 0.26 with the exact one; for
    the other transcripts it is 0.06 either way.  The remaining inflation is
    selection on the same data by the clustering and is removed only by
    held-out validation (see ``scripts/benchmark_calibration.py``).

    For every multi-isoform transcript the residual (see
    :func:`isoform_residuals`, leave-one-out by default) is correlated with
    each module eigengene from the transcript-level run (``labels`` on
    ``feature_ids``).  Because module eigengenes are themselves correlated,
    residual signal that the gene regression did not remove leaks into
    modules correlated with the gene's own module; with
    ``residualize_eigengenes`` every eigengene is first regressed on the
    gene's module eigengene, which halves false calls among unswitched minor
    isoforms on the isoform simulations without losing recall.

    A transcript is called when its residual kME to some module other than
    its gene's majority module is at least ``min_kme`` and, when ``alpha`` is
    set, the correlation is significant at Benjamini–Hochberg FDR ``alpha``
    (t-test on the best kME, Bonferroni-corrected over modules per transcript,
    BH across transcripts).  The effect-size floor matters at small sample
    sizes; the FDR test keeps precision when hundreds of samples make small
    correlations significant.  Low-abundance isoforms have attenuated
    correlations, so a fixed high ``min_kme`` systematically misses them;
    0.3 with the residualised eigengenes keeps precision above the previous
    0.4 default on the isoform simulations.

    ``null`` chooses the reference distribution of the best kME.
    ``"analytic"`` (default) is the t-test above, Bonferroni-corrected over
    modules.  ``"stratum"`` fits an empirical null per selection stratum
    (``label_discordant`` True / False) with :func:`stratum_empirical_null`:
    the |t| of the best module is scaled by a stratum-specific factor fitted
    on the lower half of the stratum's distribution, so transcripts whose
    own label is the tested module are compared with the distribution that
    selection actually gives them.  The fitted scales are returned in
    ``out.attrs["null_scale"]``.

    Every multi-isoform transcript is tested, whatever its own module label:
    the residual test is the primary evidence for isoform-specific
    membership, and ``label_discordant`` (the transcript's own label differs
    from its gene's majority module) is an annotation, not a filter.

    Returns a table indexed by transcript with gene_id, gene_module,
    isoform_module, label_discordant, best_other_module, kme_residual,
    p_value, q_value, r2_gene, n_isoforms, mean_expr and ``called``.
    """
    from collections import Counter

    from scipy import stats as sps

    from bsvae.networks.postprocess import compute_kme, module_eigengenes

    expr = expr.copy(); expr.index = expr.index.astype(str)
    fids = [str(f) for f in feature_ids]
    labels = np.asarray(labels).astype(np.int64)
    if eigengenes is None:
        eigengenes = module_eigengenes(expr, labels, fids)
    eigengenes = eigengenes.dropna(axis=1, how="any")
    resid, stats = isoform_residuals(expr, tx2gene, keep_version=keep_version, leave_one_out=leave_one_out,
                                     input_scale=input_scale)
    columns = ["gene_id", "gene_module", "isoform_module", "label_discordant", "best_other_module", "kme_residual",
               "p_value", "q_value", "r2_gene", "n_isoforms", "mean_expr", "called"]
    if eigengenes.shape[1] == 0 or len(resid) == 0:
        return pd.DataFrame(columns=columns)
    tx_gene = map_transcripts_to_genes(fids, tx2gene, keep_version)
    lab = pd.Series(labels, index=fids)
    gene_mod: Dict[str, int] = {}
    for g, idx in tx_gene.dropna().groupby(tx_gene.dropna()).groups.items():
        votes = [int(v) for v in lab.loc[list(idx)] if v >= 0]
        gene_mod[g] = Counter(votes).most_common(1)[0][0] if votes else -1
    cols = [int(c) for c in eigengenes.columns]
    gm_per_tx = np.array([gene_mod.get(g, -1) for g in stats["gene_id"].values])

    if covariates is not None and covariates.shape[1] > 0:
        # Sample-level covariates (library size, RIN, batch) that the sibling regression does not remove
        # are regressed out of both the residuals and the eigengenes before any correlation.
        Cv = covariates.reindex(expr.columns).values.astype(np.float64)
        if np.isnan(Cv).any():
            raise ValueError("covariates must be given for every sample of expr")
        Cv = np.column_stack([np.ones(len(Cv)), Cv])
        P = np.linalg.pinv(Cv)
        R = resid.values.astype(np.float64); resid = pd.DataFrame(R - (R @ P.T) @ Cv.T, index=resid.index, columns=resid.columns)
        Ev = eigengenes.values.astype(np.float64); eigengenes = pd.DataFrame(Ev - Cv @ (P @ Ev), index=eigengenes.index, columns=eigengenes.columns)

    kme = np.zeros((len(resid), len(cols)), dtype=np.float64)
    if residualize_eigengenes:
        for gm in np.unique(gm_per_tx):
            rows_idx = np.where(gm_per_tx == gm)[0]
            E = _residualize_eigengenes(eigengenes, cols.index(int(gm))) if gm in cols else eigengenes
            kme[rows_idx] = compute_kme(resid.iloc[rows_idx], E, list(resid.index[rows_idx])).values
    else:
        kme = compute_kme(resid, eigengenes, list(resid.index)).values
    mode = {True: "exact", False: "off"}.get(loo_eigengene, loo_eigengene)
    if mode not in ("exact", "approx", "off"):
        raise ValueError("loo_eigengene must be True/False, 'exact' or 'approx'")
    if mode == "approx":
        lab_e = labels if labels_for_eigengenes is None else np.asarray(labels_for_eigengenes).astype(np.int64)
        gene_of = stats["gene_id"].values
        Xz = _zscore_rows_np(expr.reindex(fids).values.astype(np.float64))
        S_ = expr.shape[1]
        for m in cols:
            members = np.where(lab_e == m)[0]
            if members.size < 3:
                continue
            Zm = Xz[members]                                   # members × samples, standardised rows
            # PC1 of the member profiles: Zm = U S V^T ; eigengene scores ∝ V[:, 0] = Zm^T u / s
            try:
                U, Sv, Vt = np.linalg.svd(Zm, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            u = U[:, 0]; e_full = Zm.T @ u                      # samples
            sign = 1.0 if np.corrcoef(e_full, Zm.mean(axis=0))[0, 1] >= 0 else -1.0
            member_gene = np.array([tx_gene.get(fids[i]) for i in members], dtype=object)
            for g in np.unique(gene_of):
                rows_idx = np.where(gene_of == g)[0]
                in_mod = np.where(member_gene == g)[0]
                if in_mod.size == 0:
                    continue
                e_loo = sign * (e_full - Zm[in_mod].T @ u[in_mod])
                e_loo = (e_loo - e_loo.mean()) / (e_loo.std(ddof=1) or 1.0)
                gm = int(gm_per_tx[rows_idx[0]])
                if residualize_eigengenes and gm in cols and gm != m:
                    eg = eigengenes.values[:, cols.index(gm)].astype(np.float64); eg = eg - eg.mean()
                    e_loo = e_loo - eg * float((e_loo * eg).sum() / max((eg ** 2).sum(), 1e-12))
                    e_loo = (e_loo - e_loo.mean()) / (e_loo.std(ddof=1) or 1.0)
                Zr = _zscore_rows_np(resid.iloc[rows_idx].values.astype(np.float64))
                kme[rows_idx, cols.index(m)] = Zr @ e_loo / (S_ - 1)
    if mode == "exact":
        lab_e = labels if labels_for_eigengenes is None else np.asarray(labels_for_eigengenes).astype(np.int64)
        lab_e_s = pd.Series(lab_e, index=fids)
        gene_of = stats["gene_id"].values
        for g in np.unique(gene_of):
            rows_idx = np.where(gene_of == g)[0]
            members_g = [t for t in tx_gene.index[tx_gene.values == g]]
            mods_g = {int(m) for m in lab_e_s.reindex(members_g).dropna().values if m >= 0 and int(m) in cols}
            gm = int(gm_per_tx[rows_idx[0]])
            for m in mods_g:
                sub_lab = lab_e.copy()
                sub_lab[np.isin(fids, members_g)] = -1
                e_m = module_eigengenes(expr, np.where(sub_lab == m, m, -1), fids).dropna(axis=1, how="any")
                if e_m.shape[1] == 0:
                    continue
                E_full = eigengenes.copy(); E_full.iloc[:, cols.index(m)] = e_m.iloc[:, 0].values
                if residualize_eigengenes and gm in cols:
                    E_full = _residualize_eigengenes(E_full, cols.index(gm))
                k = compute_kme(resid.iloc[rows_idx], E_full, list(resid.index[rows_idx])).values[:, cols.index(m)]
                kme[rows_idx, cols.index(m)] = k
    kme = np.nan_to_num(kme)
    for i, gm in enumerate(gm_per_tx):
        if gm in cols:
            kme[i, cols.index(int(gm))] = 0.0
    if max_target_corr is not None:
        # Modules whose eigengene correlates with the gene's module above the threshold are the same
        # programme split by the clustering, not a divergence target; exclude them per gene module.
        Ec = np.atleast_2d(np.nan_to_num(np.corrcoef(eigengenes.values.T.astype(np.float64))))
        for gm in np.unique(gm_per_tx):
            if gm not in cols:
                continue
            j = cols.index(int(gm))
            excl = np.where(np.abs(Ec[j]) > float(max_target_corr))[0]
            kme[np.ix_(np.where(gm_per_tx == gm)[0], excl)] = 0.0
    best_j = np.abs(kme).argmax(axis=1)
    best = kme[np.arange(len(resid)), best_j]
    n = expr.shape[1]
    r2 = np.minimum(best ** 2, 1.0 - 1e-12)
    tval = np.abs(best) * np.sqrt(max(n - 2, 1) / (1.0 - r2))
    own = lab.reindex(resid.index).fillna(-1).astype(int).values
    discordant = (own >= 0) & (gm_per_tx >= 0) & (own != gm_per_tx)
    null_scale: Dict[object, float] = {}
    if null == "analytic":
        p = np.minimum(2.0 * sps.t.sf(tval, max(n - 2, 1)) * len(cols), 1.0)
    elif null == "stratum":
        p, null_scale = stratum_empirical_null(tval, discordant, len(cols))
    else:
        raise ValueError("null must be 'analytic' or 'stratum'")
    q = _bh_adjust(p)
    called = np.abs(best) >= float(min_kme)
    if alpha is not None:
        called &= q < float(alpha)
    out = pd.DataFrame({
        "gene_id": stats["gene_id"].values,
        "gene_module": gm_per_tx,
        "isoform_module": own,
        "label_discordant": discordant,
        "best_other_module": [cols[j] for j in best_j],
        "kme_residual": best,
        "p_value": p,
        "q_value": q,
        "r2_gene": stats["r2_gene"].values,
        "n_isoforms": stats["n_isoforms"].values.astype(int),
        "mean_expr": expr.reindex(resid.index).mean(axis=1).values,
        "called": called,
    }, index=resid.index)
    out.index.name = "feature_id"
    out.attrs["null_scale"] = null_scale
    logger.info("Isoform-specific calls: %d of %d multi-isoform transcripts (kME ≥ %.2f%s)",
                int(out["called"].sum()), len(out), min_kme, f", FDR < {alpha}" if alpha is not None else "")
    if return_kme:
        return out, pd.DataFrame(kme, index=resid.index, columns=cols)
    return out


__all__.append("isoform_specific_calls")
