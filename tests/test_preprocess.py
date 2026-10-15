import numpy as np
import pandas as pd
import pytest

from bsvae.utils.preprocess import design_matrix, regress_covariates


def test_design_matrix_encodes_categorical_and_numeric():
    cov = pd.DataFrame({"batch": ["a", "b", "c", "a"], "age": [20.0, 30.0, 40.0, 50.0], "sex": ["F", "M", "F", "M"]},
                       index=list("wxyz"))
    D = design_matrix(cov)
    assert set(D.columns) == {"batch_b", "batch_c", "age", "sex_M"}
    assert abs(D["age"].mean()) < 1e-9


def test_regress_covariates_removes_batch_but_keeps_biology():
    rng = np.random.default_rng(0)
    S = 90
    batch = np.repeat([0, 1, 2], 30)
    z = rng.normal(size=S)
    beff = np.array([0.0, 2.0, -1.5])[batch]
    X = np.vstack([z + beff + 0.2 * rng.normal(size=S) for _ in range(20)]) + 5.0
    expr = pd.DataFrame(X, index=[f"f{i}" for i in range(20)], columns=[f"s{j}" for j in range(S)])
    cov = pd.DataFrame({"batch": batch.astype(str)}, index=expr.columns)
    out = regress_covariates(expr, cov)
    assert out.shape == expr.shape
    for i in range(20):
        r = out.iloc[i].values
        assert abs(np.mean(r[batch == 1]) - np.mean(r[batch == 0])) < 0.2     # batch gone
        assert abs(np.corrcoef(r, z)[0, 1]) > 0.95                             # biology kept
        assert abs(r.mean() - expr.iloc[i].mean()) < 1e-9                      # mean preserved
    with pytest.raises(ValueError, match="no covariate row"):
        regress_covariates(expr, cov.iloc[:10])
