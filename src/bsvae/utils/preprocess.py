"""
Preprocessing helpers applied to a features × samples matrix before training.

``regress_covariates`` removes known sample covariates (batch, sex, age,
cell-type proportions, ...) from every feature profile by ordinary least
squares, the way ``limma::removeBatchEffect`` does, so that co-expression
driven by those covariates does not form modules.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def design_matrix(covariates: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Intercept-free design from a sample × covariate table: numeric columns are
    centred; categorical columns are one-hot encoded (first level dropped).
    """
    cov = covariates if columns is None else covariates[list(columns)]
    parts = []
    for name in cov.columns:
        col = cov[name]
        if pd.api.types.is_numeric_dtype(col) and col.nunique() > 2:
            parts.append((col.astype(float) - col.astype(float).mean()).rename(name))
        else:
            dummies = pd.get_dummies(col.astype(str), prefix=name, drop_first=True).astype(float)
            parts.append(dummies - dummies.mean())
    if not parts:
        raise ValueError("no covariates to regress")
    return pd.concat(parts, axis=1)


def regress_covariates(
    expr: pd.DataFrame,
    covariates: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    keep_mean: bool = True,
) -> pd.DataFrame:
    """
    Regress covariates out of every feature profile (features × samples).

    Parameters
    ----------
    expr : features × samples
    covariates : samples × covariates (index = sample IDs matching ``expr`` columns)
    columns : subset of covariate columns to use (default: all)
    keep_mean : add each feature's mean back so values stay on the input scale
    """
    samples = [str(c) for c in expr.columns]
    cov = covariates.copy()
    cov.index = cov.index.astype(str)
    missing = [s for s in samples if s not in cov.index]
    if missing:
        raise ValueError(f"{len(missing)} samples have no covariate row (e.g. {missing[:3]})")
    D = design_matrix(cov.loc[samples], columns).values.astype(np.float64)   # S × p
    X = expr.values.astype(np.float64)                                      # F × S
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    # Projection onto the covariate space, computed once for all features.
    beta, *_ = np.linalg.lstsq(D, Xc.T, rcond=None)                        # p × F
    resid = Xc - (D @ beta).T
    if keep_mean:
        resid = resid + mean
    logger.info("Regressed %d covariate columns out of %d features", D.shape[1], X.shape[0])
    return pd.DataFrame(resid, index=expr.index, columns=expr.columns)


__all__ = ["design_matrix", "regress_covariates"]
