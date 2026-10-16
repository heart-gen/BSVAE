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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Regress each isoform's profile on its gene's aggregate profile and return
    the residual matrix (multi-isoform transcripts × samples) plus per-transcript
    statistics (gene_id, n_isoforms, slope, r2_gene, residual_sd).

    Residuals are what the gene-level programme does not explain, so
    clustering them finds isoform-specific co-expression; transcripts of
    single-isoform genes are excluded (their residual is identically zero).
    """
    expr = expr.copy()
    expr.index = expr.index.astype(str)
    tx_gene = map_transcripts_to_genes(expr.index, tx2gene, keep_version)
    counts = tx_gene.value_counts()
    multi = tx_gene.map(counts).fillna(0) >= int(min_isoforms)
    sub = expr.loc[multi.values]
    genes = tx_gene.loc[sub.index]
    agg = gene_aggregate(expr, tx_gene)
    G = agg.reindex(genes.values).values            # (T, S) gene profile per transcript
    X = sub.values.astype(np.float64)

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


def isoform_specific_calls(
    expr: pd.DataFrame,
    tx2gene: "pd.Series | Dict[str, str]",
    labels: np.ndarray,
    feature_ids: Sequence[str],
    min_kme: float = 0.4,
    keep_version: bool = False,
    eigengenes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Call isoforms whose *gene-programme-removed* profile belongs to a module
    other than their gene's module ("isoform-specific membership").

    For every multi-isoform transcript the residual (see
    :func:`isoform_residuals`) is correlated with each module eigengene from
    the transcript-level run (``labels`` on ``feature_ids``).  A transcript is
    called when its residual kME to some module other than its gene's
    majority module is at least ``min_kme``.  On the isoform simulations a
    threshold of 0.4–0.5 gives 6–9× enrichment of truly switched isoforms over
    the base rate; lower thresholds trade precision for recall.

    Returns a table indexed by transcript with gene_id, gene_module,
    best_other_module, kme_residual, r2_gene, n_isoforms and ``called``.
    """
    from collections import Counter

    from bsvae.networks.postprocess import compute_kme, module_eigengenes

    expr = expr.copy(); expr.index = expr.index.astype(str)
    fids = [str(f) for f in feature_ids]
    labels = np.asarray(labels).astype(np.int64)
    if eigengenes is None:
        eigengenes = module_eigengenes(expr, labels, fids)
    eigengenes = eigengenes.dropna(axis=1, how="any")
    resid, stats = isoform_residuals(expr, tx2gene, keep_version=keep_version)
    if eigengenes.shape[1] == 0 or len(resid) == 0:
        return pd.DataFrame(columns=["gene_id", "gene_module", "best_other_module", "kme_residual",
                                     "r2_gene", "n_isoforms", "called"])
    kme = compute_kme(resid, eigengenes, list(resid.index))
    tx_gene = map_transcripts_to_genes(fids, tx2gene, keep_version)
    lab = pd.Series(labels, index=fids)
    gene_mod: Dict[str, int] = {}
    for g, idx in tx_gene.dropna().groupby(tx_gene.dropna()).groups.items():
        votes = [int(v) for v in lab.loc[list(idx)] if v >= 0]
        gene_mod[g] = Counter(votes).most_common(1)[0][0] if votes else -1
    cols = [int(c) for c in kme.columns]
    rows = []
    for i, tx in enumerate(resid.index):
        g = stats.loc[tx, "gene_id"]
        gm = gene_mod.get(g, -1)
        row = kme.iloc[i].values.astype(float).copy()
        if gm in cols:
            row[cols.index(gm)] = 0.0
        row = np.nan_to_num(row)
        j = int(np.abs(row).argmax())
        rows.append({
            "gene_id": g, "gene_module": gm, "best_other_module": cols[j],
            "kme_residual": float(row[j]), "r2_gene": float(stats.loc[tx, "r2_gene"]),
            "n_isoforms": int(stats.loc[tx, "n_isoforms"]),
            "called": bool(abs(row[j]) >= float(min_kme)),
        })
    out = pd.DataFrame(rows, index=resid.index)
    out.index.name = "feature_id"
    logger.info("Isoform-specific calls: %d of %d multi-isoform transcripts (kME ≥ %.2f)",
                int(out["called"].sum()), len(out), min_kme)
    return out


__all__.append("isoform_specific_calls")
