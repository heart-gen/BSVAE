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


__all__ = ["size_factors", "pearson_residuals", "design_matrix", "regress_covariates"]


def size_factors(counts: "np.ndarray | pd.DataFrame", method: str = "median_ratio") -> np.ndarray:
    """Per-sample size factors for a features × samples count matrix (geometric mean 1)."""
    X = np.asarray(counts, dtype=np.float64)
    totals = np.maximum(X.sum(axis=0), 1.0)
    if method == "total":
        size = totals
    elif method == "median_ratio":
        expressed = (X > 0).all(axis=1)
        if expressed.sum() >= 10:
            logc = np.log(X[expressed])
            ref = logc.mean(axis=1, keepdims=True)
            size = np.exp(np.median(logc - ref, axis=0))
        else:
            size = totals
    else:
        raise ValueError(f"Unknown size-factor method: {method}")
    return size / np.exp(np.mean(np.log(size)))


def pearson_residuals(
    counts: pd.DataFrame,
    theta: "float | str" = 100.0,
    clip: Optional[float] = None,
    size_factor_method: str = "median_ratio",
) -> pd.DataFrame:
    """
    Analytic negative-binomial Pearson residuals of a features × samples count
    matrix (Lause, Berens & Kobak 2021), a variance-stabilised alternative to
    log-normalised counts for low-abundance features.

    The expected count of feature *i* in sample *j* is ``mu_ij = n_i * s_j /
    sum(s)`` with ``n_i`` the feature total and ``s_j`` the sample size factor;
    the residual is ``(x - mu) / sqrt(mu + mu^2 / theta)``, clipped to
    ``±clip`` (default ``sqrt(n_samples)``).  ``theta`` is a fixed
    overdispersion (100 by default) or ``"moments"`` for a per-feature
    method-of-moments estimate clipped to [1, 1000]; a fixed value keeps
    features with strong biological variation from being shrunk.
    """
    X = counts.values.astype(np.float64)
    s = size_factors(X, size_factor_method)
    mu = X.sum(axis=1, keepdims=True) * (s[None, :] / s.sum())
    if isinstance(theta, str):
        if theta != "moments":
            raise ValueError("theta must be a number or 'moments'")
        norm = X / s[None, :]
        m = norm.mean(axis=1); v = norm.var(axis=1, ddof=1)
        th = np.where(v > m, m ** 2 / np.maximum(v - m, 1e-12), 1000.0)
        th = np.clip(th, 1.0, 1000.0)[:, None]
    else:
        th = float(theta)
    resid = (X - mu) / np.sqrt(mu + mu ** 2 / th + 1e-12)
    c = float(np.sqrt(X.shape[1])) if clip is None else float(clip)
    resid = np.clip(resid, -c, c)
    return pd.DataFrame(resid, index=counts.index, columns=counts.columns)
