"""The stratum-aware empirical null of the isoform-specific residual scan."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import cross_fit_calls, isoform_specific_calls, stratum_empirical_null


def _max_abs_normals(rng, n, m, sigma):
    return sigma * np.abs(rng.standard_normal((n, m))).max(axis=1)


def test_stratum_empirical_null_recovers_the_scale_and_is_uniform_per_stratum():
    rng = np.random.default_rng(0)
    a = np.concatenate([_max_abs_normals(rng, 4000, 8, 1.0), _max_abs_normals(rng, 4000, 8, 1.6)])
    strata = np.r_[np.zeros(4000, bool), np.ones(4000, bool)]
    p, scales = stratum_empirical_null(a, strata, 8)
    assert abs(scales[False] - 1.0) < 0.05 and abs(scales[True] - 1.6) < 0.08
    for s in (False, True):
        assert sps.kstest(p[strata == s], "uniform").pvalue > 0.01
        assert abs((p[strata == s] < 0.05).mean() - 0.05) < 0.015


def test_stratum_empirical_null_ignores_the_upper_tail_and_small_strata():
    rng = np.random.default_rng(1)
    a = _max_abs_normals(rng, 5000, 10, 1.0)
    a[:500] += 4.0                                   # 10 % real signal in the upper tail
    p, scales = stratum_empirical_null(a, np.zeros(5000, bool), 10)
    assert abs(scales[False] - 1.0) < 0.06
    assert (p[:500] < 0.01).mean() > 0.9
    p2, scales2 = stratum_empirical_null(a[:20], np.zeros(20, bool), 10)
    assert scales2[False] == 1.0 and np.all((p2 >= 0) & (p2 <= 1))
    # per-transcript module counts are accepted
    p3, _ = stratum_empirical_null(a, np.zeros(5000, bool), np.full(5000, 10))
    np.testing.assert_allclose(p3, p)


def _sim(seed=0):
    r = simulate_params(dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=150, n_samples=60, n_modules=5, usage_rate=0.0, switch_rate=0.0), seed)
    expr = r.expr_features_x_samples; expr.index = expr.index.astype(str); expr.columns = expr.columns.astype(str)
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
    truth = r.modules_hard.set_index("feature_id")["module"]
    truth.index = truth.index.astype(str)
    lab = truth.reindex(expr.index).fillna(-1).astype(int).values
    return expr, t2g, lab


def test_calls_keep_the_analytic_null_by_default_and_accept_the_stratum_null():
    expr, t2g, lab = _sim()
    fids = list(expr.index)
    base = isoform_specific_calls(expr, t2g, lab, fids)
    same = isoform_specific_calls(expr, t2g, lab, fids, null="analytic")
    pd.testing.assert_frame_equal(base, same)
    strat = isoform_specific_calls(expr, t2g, lab, fids, null="stratum")
    assert list(strat.columns) == list(base.columns)
    assert strat["p_value"].between(0, 1).all()
    assert set(strat.attrs["null_scale"]) <= {False, True}
    m = ~strat["label_discordant"].values
    rho = sps.spearmanr(np.abs(strat["kme_residual"].values[m]), strat["p_value"].values[m]).correlation
    assert rho < -0.99            # within a stratum the p-value is monotone in |kME|
    with pytest.raises(ValueError):
        isoform_specific_calls(expr, t2g, lab, fids, null="bogus")


def test_cross_fit_calls_accept_the_stratum_null():
    expr, t2g, lab = _sim()
    fids = list(expr.index)
    fold = np.arange(expr.shape[1]) % 2
    base = cross_fit_calls(expr, t2g, [lab, lab], fold, fids)
    strat = cross_fit_calls(expr, t2g, [lab, lab], fold, fids, null="stratum")
    assert list(strat.columns) == list(base.columns)
    assert strat["p_value"].between(0, 1).all() and "null_scale" in strat.attrs
    assert np.array_equal(base["best_other_module"].values, strat["best_other_module"].values)
