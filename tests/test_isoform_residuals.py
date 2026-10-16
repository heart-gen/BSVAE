import numpy as np
import pandas as pd
import pytest

from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import gene_aggregate, isoform_residuals, map_transcripts_to_genes


def test_residuals_remove_gene_programme_and_exclude_singletons():
    rng = np.random.default_rng(0)
    S = 60
    g = rng.normal(size=S)
    iso_a = 2 * g + 0.1 * rng.normal(size=S) + 5
    iso_b = 1.5 * g + 0.1 * rng.normal(size=S) + 3
    single = rng.normal(size=S)
    expr = pd.DataFrame(np.vstack([iso_a, iso_b, single]), index=["t1.2", "t2", "t3"],
                        columns=[f"s{j}" for j in range(S)])
    tx2gene = pd.Series({"t1": "gA", "t2": "gA", "t3": "gB"})
    resid, stats = isoform_residuals(expr, tx2gene, leave_one_out=False)
    assert list(resid.index) == ["t1.2", "t2"]
    agg = gene_aggregate(expr, map_transcripts_to_genes(expr.index, tx2gene))
    for t in resid.index:
        assert abs(np.corrcoef(resid.loc[t], agg.loc["gA"])[0, 1]) < 0.05
    assert stats.loc["t1.2", "r2_gene"] > 0.95 and stats.loc["t1.2", "n_isoforms"] == 2


def test_leave_one_out_regresses_on_siblings_only():
    """With two isoforms the leave-one-out gene profile of each is exactly the other isoform."""
    rng = np.random.default_rng(1)
    S = 50
    a = rng.normal(size=S) + 4
    b = rng.normal(size=S) + 3
    expr = pd.DataFrame(np.vstack([a, b]), index=["t1", "t2"], columns=[f"s{j}" for j in range(S)])
    tx2gene = {"t1": "g", "t2": "g"}
    resid, stats = isoform_residuals(expr, tx2gene, leave_one_out=True)
    bc = b - b.mean(); ac = a - a.mean()
    slope = (ac * bc).sum() / (bc ** 2).sum()
    np.testing.assert_allclose(resid.loc["t1"].values, ac - slope * bc, atol=1e-8)
    assert abs(stats.loc["t1", "slope"] - slope) < 1e-8
    # full-aggregate regression removes part of the isoform's own signal
    resid_full, _ = isoform_residuals(expr, tx2gene, leave_one_out=False)
    assert resid_full.loc["t1"].std() < resid.loc["t1"].std()


def test_residuals_isolate_switched_isoform_signal():
    """On the mixing simulation, the switched isoform's residual tracks its new module."""
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=80, n_modules=4, switch_rate=0.5,
             alpha=0.7, signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=2)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    resid, stats = isoform_residuals(expr, t2g)
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    z = r.module_latents.values
    gaps_sw, gaps_plain = [], []
    for tx in resid.index:
        row = st.loc[tx]
        x = resid.loc[tx].values
        if x.std() == 0:
            continue
        c_new = abs(np.corrcoef(x, z[:, int(row["tx_module"])])[0, 1])
        (gaps_sw if row["is_switched"] == 1 else gaps_plain).append(c_new)
    assert np.mean(gaps_sw) > 0.35
    assert np.mean(gaps_sw) > np.mean(gaps_plain) + 0.15


def test_isoform_specific_calls_enrich_switched_isoforms():
    from bsvae.utils.isoform import isoform_specific_calls
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=80, n_modules=4, switch_rate=0.4,
             alpha=1.0, signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=3)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values   # oracle labels
    calls = isoform_specific_calls(expr, t2g, truth, list(expr.index))
    assert {"p_value", "q_value", "mean_expr"} <= set(calls.columns)
    assert ((calls["q_value"] >= calls["p_value"] - 1e-12) | calls["p_value"].isna()).all()
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    is_sw = st.loc[calls.index, "is_switched"].values == 1
    called = calls["called"].values
    precision = (called & is_sw).sum() / called.sum()
    recall = (called & is_sw).sum() / is_sw.sum()
    assert recall > 0.5 and precision > 3 * is_sw.mean()
    hit = calls[called & is_sw]
    assert (hit["best_other_module"].values == st.loc[hit.index, "tx_module"].values).mean() > 0.8


def test_residualised_eigengenes_suppress_leakage_from_correlated_modules():
    """An unswitched minor isoform of a gene in module A must not be called for a module B
    correlated with A; residualising the eigengenes on A removes that leakage."""
    from bsvae.utils.isoform import isoform_specific_calls
    rng = np.random.default_rng(5)
    S, n_genes = 120, 40
    zA = rng.normal(size=S)
    zB = 0.6 * zA + 0.8 * rng.normal(size=S)             # B correlated with A
    rows, ids, t2g, labels = [], [], {}, []
    for g in range(n_genes):
        z = zA if g < n_genes // 2 else zB
        mod = 0 if g < n_genes // 2 else 1
        dom = 2.0 * z + 0.3 * rng.normal(size=S) + 6
        minor = 0.5 * z + 0.6 * rng.normal(size=S) + 2     # noisier, lower-expressed copy of the gene programme
        for k, x in enumerate([dom, minor]):
            ids.append(f"t{g}_{k}"); t2g[f"t{g}_{k}"] = f"g{g}"; rows.append(x); labels.append(mod)
    expr = pd.DataFrame(np.vstack(rows), index=ids, columns=[f"s{j}" for j in range(S)])
    labels = np.array(labels)
    raw = isoform_specific_calls(expr, t2g, labels, ids, min_kme=0.3, alpha=None, residualize_eigengenes=False)
    fixed = isoform_specific_calls(expr, t2g, labels, ids, min_kme=0.3, alpha=None, residualize_eigengenes=True)
    assert fixed["called"].mean() <= raw["called"].mean()
    assert fixed["called"].mean() < 0.1
    assert fixed["kme_residual"].abs().mean() < raw["kme_residual"].abs().mean()


def test_alpha_filters_small_correlations_in_large_samples():
    from bsvae.utils.isoform import isoform_specific_calls
    rng = np.random.default_rng(7)
    S = 400
    z0, z1 = rng.normal(size=S), rng.normal(size=S)
    rows, ids, t2g, labels = [], [], {}, []
    for g in range(30):
        z = z0 if g < 15 else z1
        for k in range(2):
            ids.append(f"t{g}_{k}"); t2g[f"t{g}_{k}"] = f"g{g}"; labels.append(0 if g < 15 else 1)
            rows.append(z * (2.0 if k == 0 else 0.4) + rng.normal(size=S) * (0.3 if k == 0 else 1.0) + 3)
    expr = pd.DataFrame(np.vstack(rows), index=ids, columns=[f"s{j}" for j in range(S)])
    loose = isoform_specific_calls(expr, t2g, np.array(labels), ids, min_kme=0.0, alpha=None)
    strict = isoform_specific_calls(expr, t2g, np.array(labels), ids, min_kme=0.0, alpha=0.05)
    assert loose["called"].all()
    assert strict["called"].mean() < 0.2


def test_linear_input_scale_sums_siblings_directly():
    rng = np.random.default_rng(3)
    S = 40
    a, b, c = rng.normal(size=S), rng.normal(size=S), rng.normal(size=S)
    expr = pd.DataFrame(np.vstack([a, b, c]), index=["t1", "t2", "t3"], columns=[f"s{j}" for j in range(S)])
    t2g = {"t1": "g", "t2": "g", "t3": "g"}
    resid, stats = isoform_residuals(expr, t2g, input_scale="linear")
    sib = b + c
    sc, ac = sib - sib.mean(), a - a.mean()
    slope = (ac * sc).sum() / (sc ** 2).sum()
    np.testing.assert_allclose(resid.loc["t1"].values, ac - slope * sc, atol=1e-8)
    with pytest.raises(ValueError):
        isoform_residuals(expr, t2g, input_scale="counts")


def test_calls_annotate_label_discordance_without_filtering():
    """Every multi-isoform transcript is tested; discordant raw labels are an annotation."""
    from bsvae.utils.isoform import isoform_specific_calls
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=200, n_samples=60, n_modules=4, switch_rate=0.3, alpha=1.0,
             signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=9)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    n_multi = int((st["gene_id"].map(st.groupby("gene_id").size()) >= 2).sum())
    calls = isoform_specific_calls(expr, t2g, truth, list(expr.index))
    assert len(calls) == n_multi
    assert {"isoform_module", "label_discordant"} <= set(calls.columns)
    disc = calls["label_discordant"].values
    sw = st.loc[calls.index, "is_switched"].values == 1
    assert (disc == sw).mean() > 0.95                       # oracle labels: discordance == switched
    # concordant transcripts are still tested and can be called
    assert calls.loc[~disc, "p_value"].notna().all()


def test_loo_eigengene_removes_self_contribution():
    """A transcript labelled in the module it is tested against inflates its own kME through the
    eigengene; the leave-one-gene-out eigengene removes that contribution."""
    from bsvae.utils.isoform import isoform_specific_calls
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=200, n_samples=60, n_modules=4, switch_rate=0.0,
             signal_scale=1.0, null_fraction=0.0)
    r = simulate_params(p, seed=12)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values.copy()
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    # mislabel 30 unswitched minor isoforms into another module (as a noisy clustering would)
    rng = np.random.default_rng(1)
    minor = np.where(st.loc[expr.index, "isoform_rank"].values > 0)[0]
    bad = rng.choice(minor, size=30, replace=False)
    lab = truth.copy()
    for i in bad:
        lab[i] = rng.choice([m for m in range(4) if m != truth[i]])
    plain = isoform_specific_calls(expr, t2g, lab, list(expr.index), alpha=None, min_kme=0.0)
    loo = isoform_specific_calls(expr, t2g, lab, list(expr.index), alpha=None, min_kme=0.0, loo_eigengene=True)
    ids = [expr.index[i] for i in bad]
    # for the mislabelled transcripts the plain kME to their (wrong) own module is inflated
    own_plain = np.array([plain.loc[t, "kme_residual"] if plain.loc[t, "best_other_module"] == lab[i] else np.nan for t, i in zip(ids, bad)])
    own_loo = np.array([loo.loc[t, "kme_residual"] if loo.loc[t, "best_other_module"] == lab[i] else np.nan for t, i in zip(ids, bad)])
    both = ~np.isnan(own_plain) & ~np.isnan(own_loo)
    assert both.sum() >= 5
    assert np.nanmean(np.abs(own_loo[both])) < np.nanmean(np.abs(own_plain[both]))
    # p-values for the mislabelled set are less extreme with the leave-one-gene-out eigengene
    assert loo.loc[ids, "p_value"].median() >= plain.loc[ids, "p_value"].median()
