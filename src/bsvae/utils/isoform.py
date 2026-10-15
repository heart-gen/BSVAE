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
