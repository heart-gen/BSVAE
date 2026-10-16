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
from typing import Dict, Optional, Sequence, Tuple

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
) -> pd.DataFrame:
    """
    Call isoforms whose *gene-programme-removed* profile belongs to a module
    other than their gene's ("isoform-specific membership").

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

    kme = np.zeros((len(resid), len(cols)), dtype=np.float64)
    if residualize_eigengenes:
        for gm in np.unique(gm_per_tx):
            rows_idx = np.where(gm_per_tx == gm)[0]
            E = _residualize_eigengenes(eigengenes, cols.index(int(gm))) if gm in cols else eigengenes
            kme[rows_idx] = compute_kme(resid.iloc[rows_idx], E, list(resid.index[rows_idx])).values
    else:
        kme = compute_kme(resid, eigengenes, list(resid.index)).values
    kme = np.nan_to_num(kme)
    for i, gm in enumerate(gm_per_tx):
        if gm in cols:
            kme[i, cols.index(int(gm))] = 0.0
    best_j = np.abs(kme).argmax(axis=1)
    best = kme[np.arange(len(resid)), best_j]
    n = expr.shape[1]
    r2 = np.minimum(best ** 2, 1.0 - 1e-12)
    tval = np.abs(best) * np.sqrt(max(n - 2, 1) / (1.0 - r2))
    p = np.minimum(2.0 * sps.t.sf(tval, max(n - 2, 1)) * len(cols), 1.0)
    q = _bh_adjust(p)
    called = np.abs(best) >= float(min_kme)
    if alpha is not None:
        called &= q < float(alpha)
    own = lab.reindex(resid.index).fillna(-1).astype(int).values
    out = pd.DataFrame({
        "gene_id": stats["gene_id"].values,
        "gene_module": gm_per_tx,
        "isoform_module": own,
        "label_discordant": (own >= 0) & (gm_per_tx >= 0) & (own != gm_per_tx),
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
    logger.info("Isoform-specific calls: %d of %d multi-isoform transcripts (kME ≥ %.2f%s)",
                int(out["called"].sum()), len(out), min_kme, f", FDR < {alpha}" if alpha is not None else "")
    return out


__all__.append("isoform_specific_calls")
