"""Cross-fitted isoform-specific calls and the coverage metric."""
import numpy as np
import pandas as pd
import pytest

from bsvae.simulation.metrics import coverage_metrics
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import cross_fit_calls, isoform_specific_calls


def test_coverage_metrics_abstains_on_low_scores():
    t = np.array([0] * 10 + [1] * 10)
    p = t.copy(); p[:3] = 1                      # three errors in module 0
    score = np.ones(20); score[:3] = 0.1         # the errors have the lowest confidence
    out = coverage_metrics(t, p, score, grey_fracs=(0.0, 0.15, 0.5))
    assert out["ari_at_grey_00"] < 1.0
    assert out["ari_at_grey_15"] == pytest.approx(1.0)
    assert out["ari_at_grey_50"] == pytest.approx(1.0)
    p2 = p.copy(); p2[:5] = -1                    # already 25 % grey: 15 % target is unreachable
    assert np.isnan(coverage_metrics(t, p2, score, grey_fracs=(0.15,))["ari_at_grey_15"])


def _sim(switch_rate, seed=0):
    r = simulate_params(dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=200, n_samples=120, n_modules=4,
                             switch_rate=switch_rate, usage_rate=0.0), seed)
    e = r.expr_features_x_samples; e.index = e.index.astype(str)
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
    truth = r.modules_hard.set_index("feature_id")["module"]; truth.index = truth.index.astype(str)
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id"); st.index = st.index.astype(str)
    return e, t2g, truth.reindex(e.index).fillna(-1).astype(int).values, st


def test_cross_fit_calls_matches_single_fold_structure_and_recovers_switches():
    e, t2g, truth, st = _sim(0.3)
    fold = np.arange(e.shape[1]) % 2
    # oracle labels for both folds (the function only needs labels per fold)
    calls = cross_fit_calls(e, t2g, [truth, truth], fold, list(e.index))
    single = isoform_specific_calls(e, t2g, truth, list(e.index))
    assert set(calls.index) == set(single.index)
    for c in ("kme_residual", "p_value", "q_value", "best_other_module", "modules_agree", "label_discordant", "called"):
        assert c in calls.columns
    sw = st.loc[calls.index, "is_switched"].values == 1
    # true switches point at the same target module in both folds; nulls pick a random one per fold
    assert calls["modules_agree"][sw].mean() > 0.7
    assert calls["modules_agree"][sw].mean() > calls["modules_agree"][~sw].mean()
    assert calls["called"][sw].mean() > 0.5
    assert calls["called"][~sw].mean() < 0.1


def test_cross_fit_calls_module_mapping_and_agreement():
    e, t2g, truth, st = _sim(0.3, seed=1)
    fold = np.arange(e.shape[1]) % 2
    perm = {m: (m + 1) % 4 for m in range(4)}
    relabelled = np.array([perm[m] if m >= 0 else -1 for m in truth])
    same = cross_fit_calls(e, t2g, [truth, truth], fold, list(e.index))
    calls = cross_fit_calls(e, t2g, [truth, relabelled], fold, list(e.index))
    # permuted ids are mapped back onto fold 0's: identical results to the un-permuted run
    assert (calls["best_other_module"].values == same["best_other_module"].values).all()
    assert (calls["modules_agree"].values == same["modules_agree"].values).all()
    assert np.allclose(calls["p_value"].values, same["p_value"].values)
    bad = np.where(truth >= 0, (truth + 1) % 4, -1)
    bad[: len(bad) // 2] = truth[: len(bad) // 2]  # half the features keep their labels, half are scrambled
    strict = cross_fit_calls(e, t2g, [truth, bad], fold, list(e.index))
    loose = cross_fit_calls(e, t2g, [truth, bad], fold, list(e.index), require_module_agreement=False)
    assert loose["called"].sum() >= strict["called"].sum()


def test_cross_fit_rejects_fold_mismatch():
    e, t2g, truth, _ = _sim(0.0)
    with pytest.raises(ValueError):
        cross_fit_calls(e, t2g, [truth], np.arange(e.shape[1]) % 2, list(e.index))


def test_covariates_remove_a_technical_axis_and_target_corr_excludes_modules():
    e, t2g, truth, st = _sim(0.0, seed=2)
    rng = np.random.default_rng(0)
    axis = rng.normal(size=e.shape[1])
    # inject a sample-level technical axis into every transcript with a transcript-specific weight
    w = rng.normal(scale=0.5, size=e.shape[0])
    e2 = e + np.outer(w, axis)
    cov = pd.DataFrame({"axis": axis}, index=e.columns)
    plain = isoform_specific_calls(e2, t2g, truth, list(e.index))
    adj = isoform_specific_calls(e2, t2g, truth, list(e.index), covariates=cov)
    assert (adj["p_value"] < 0.05).mean() < (plain["p_value"] < 0.05).mean()
    assert adj["called"].sum() <= plain["called"].sum()
    with pytest.raises(ValueError):
        isoform_specific_calls(e2, t2g, truth, list(e.index), covariates=cov.iloc[:10])
    # a correlation threshold of 0 excludes every other module, so nothing can be called
    none = isoform_specific_calls(e, t2g, truth, list(e.index), max_target_corr=-0.1)
    assert none.loc[none["gene_module"] >= 0, "called"].sum() == 0   # null genes have no module to compare with
