"""Tests for the scenario simulator refactor and the isoform-aware generator."""

import json

import numpy as np
import pandas as pd
import pytest

from bsvae.simulation.scenario import (
    DEFAULT_CONFIG,
    ISOFORM_GENERATOR_DEFAULTS,
    expand_scenarios,
    generate_scenario,
    load_config,
    simulate_params,
    write_starter_config,
)


def _iso_params(**over):
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=200, n_samples=40, n_modules=4)
    p.update(over)
    return p


def _nb_params(**over):
    p = dict(DEFAULT_CONFIG["generator"], n_features=300, n_samples=40, n_modules=5)
    p.update(over)
    return p


def test_isoform_shapes_and_mapping():
    r = simulate_params(_iso_params(), seed=3)
    tx2gene = r.extra_truth["tx2gene.tsv"]
    n_tx = r.expr_features_x_samples.shape[0]
    assert len(tx2gene) == n_tx == len(r.modules_hard)
    assert r.expr_features_x_samples.shape == (n_tx, 40)
    assert tx2gene["transcript_id"].is_unique
    per_gene = tx2gene.groupby("gene_id").size()
    assert per_gene.min() >= 1 and per_gene.max() <= 6
    assert len(per_gene) == 200
    assert r.metadata["family"] == "nb_isoform_modules"


def test_isoform_null_genes_are_grey():
    r = simulate_params(_iso_params(null_fraction=0.2), seed=5)
    genes = r.extra_truth["modules_gene.csv"]
    frac = (genes["module"] == -1).mean()
    assert abs(frac - 0.2) < 0.02
    st = r.extra_truth["switch_table.tsv.gz"]
    null_tx = st[st["is_null"] == 1]
    assert (null_tx["tx_module"] == -1).all()
    assert (r.modules_hard.loc[r.modules_hard["is_null"] == 1, "module"] == -1).all()


def test_isoform_switching_structure():
    r0 = simulate_params(_iso_params(switch_rate=0.0), seed=7)
    st0 = r0.extra_truth["switch_table.tsv.gz"]
    assert (st0["tx_module"] == st0["gene_module"]).all()

    r = simulate_params(_iso_params(switch_rate=0.4, alpha=1.0), seed=7)
    st = r.extra_truth["switch_table.tsv.gz"]
    switched = st[st["is_switched"] == 1]
    assert len(switched) > 0
    # One switched isoform per switched gene, never the dominant isoform, never in null genes.
    assert switched.groupby("gene_id").size().max() == 1
    assert (switched["isoform_rank"] >= 1).all()
    assert (switched["gene_module"] >= 0).all()
    assert (switched["tx_module"] != switched["gene_module"]).all()
    single = st.groupby("gene_id")["transcript_id"].transform("size") == 1
    assert not (switched["transcript_id"].isin(st.loc[single, "transcript_id"])).any()


def test_isoform_alpha_mixing_monotone():
    """Higher alpha → switched isoform tracks its new module factor more, its gene's less."""
    def corr_gap(alpha):
        r = simulate_params(_iso_params(switch_rate=0.5, alpha=alpha, signal_scale=3.0), seed=11)
        st = r.extra_truth["switch_table.tsv.gz"]
        z = r.module_latents.values
        X = r.expr_features_x_samples
        gaps = []
        for _, row in st[st["is_switched"] == 1].iterrows():
            x = X.loc[row["transcript_id"]].values
            c_new = np.corrcoef(x, z[:, int(row["tx_module"])])[0, 1]
            c_gene = np.corrcoef(x, z[:, int(row["gene_module"])])[0, 1]
            gaps.append(c_new - c_gene)
        return float(np.mean(gaps))
    assert corr_gap(0.1) < corr_gap(0.5) < corr_gap(0.9)


def test_arms_are_paired_at_a_seed():
    a = simulate_params(_nb_params(confounding="none"), seed=2)
    b = simulate_params(_nb_params(confounding="moderate"), seed=2)
    pd.testing.assert_frame_equal(a.module_latents, b.module_latents)
    pd.testing.assert_frame_equal(a.modules_hard, b.modules_hard)
    assert np.allclose(a.gene_metadata["alpha"], b.gene_metadata["alpha"])
    assert b.metadata["n_batch_affected"] > 0 and a.metadata["n_batch_affected"] == 0


def test_batch_confounding_survives_cpm():
    """Gene-specific batch effects remain visible in log-CPM for affected features only."""
    b = simulate_params(_nb_params(confounding="strong", signal_scale=0.5, n_samples=120), seed=4)
    X = b.expr_features_x_samples.values
    batch = b.covariates["batch"].values
    aff = b.gene_metadata["batch_affected"].values == 1

    def batch_r2(x):
        groups = [x[batch == g] for g in np.unique(batch)]
        ssb = sum(len(v) * (v.mean() - x.mean()) ** 2 for v in groups)
        sst = ((x - x.mean()) ** 2).sum()
        return ssb / sst if sst > 0 else 0.0

    r2_aff = np.mean([batch_r2(X[i]) for i in np.where(aff)[0]])
    r2_un = np.mean([batch_r2(X[i]) for i in np.where(~aff)[0]])
    assert r2_aff > 0.15
    assert r2_aff > 2 * r2_un


def test_nonlinear_mode_is_feature_specific():
    r = simulate_params(_nb_params(nonlinear_mode="on"), seed=9)
    assert r.gene_metadata["nonlinear_func"].nunique() >= 3
    r_lin = simulate_params(_nb_params(nonlinear_mode="off"), seed=9)
    assert (r_lin.gene_metadata["nonlinear_func"] == 0).all()


def test_null_fraction_and_overlap_truth_for_nb_family():
    r = simulate_params(_nb_params(null_fraction=0.1, overlap_rate=0.2), seed=1)
    assert abs((r.modules_hard["module"] == -1).mean() - 0.1) < 0.02
    ov = r.modules_overlap
    assert (ov.loc[r.modules_hard["module"] == -1, "modules"] == "grey").all()
    assert (ov["n_memberships"] == 2).sum() > 0


def test_load_config_json_and_yaml(tmp_path):
    p = tmp_path / "cfg.json"
    write_starter_config(p)
    cfg = load_config(p)
    assert cfg["generator"]["family"] == "nb_latent_modules"
    yaml = pytest.importorskip("yaml")
    y = tmp_path / "cfg.yaml"
    y.write_text("base_seed: 3\ngenerator:\n  family: nb_isoform_modules\n  n_genes: 50\ngrid:\n  n_samples: [20, 30]\n")
    cfg = load_config(y)
    assert cfg["base_seed"] == 3 and cfg["generator"]["n_genes"] == 50
    scen = expand_scenarios(cfg)
    assert len(scen) == 2 and scen[0][1]["switch_rate"] == ISOFORM_GENERATOR_DEFAULTS["switch_rate"]


def test_generate_scenario_isoform_writes_truth_files(tmp_path):
    cfg = {
        "base_seed": 1,
        "generator": dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=60, n_samples=20, n_modules=3, id_version=True),
        "grid": {"n_samples": [20]},
        "outputs": {"emit_wgcna_matrix": False, "emit_gnvae_folds": True, "gnvae_n_folds": 2},
    }
    sid = expand_scenarios(cfg)[0][0]
    run_dir = generate_scenario(cfg, sid, rep=0, outdir=tmp_path)
    assert (run_dir / "truth" / "tx2gene.tsv").exists()
    assert (run_dir / "truth" / "modules_gene.csv").exists()
    assert (run_dir / "truth" / "switch_table.tsv.gz").exists()
    assert not (run_dir / "expr" / "samples_x_features.tsv.gz").exists()
    mi = json.load(open(run_dir / "method_inputs.json"))
    assert "bsvae_tx2gene" in mi and "wgcna_expr" not in mi
    tx2gene = pd.read_csv(run_dir / "truth" / "tx2gene.tsv", sep="\t")
    assert tx2gene["transcript_id"].str.endswith(".1").all()
    # GNVAE folds split samples (columns), not features.
    tr = pd.read_csv(run_dir / "gnvae" / "fold_0" / "X_train.tsv.gz", sep="\t", index_col=0)
    te = pd.read_csv(run_dir / "gnvae" / "fold_0" / "X_test.tsv.gz", sep="\t", index_col=0)
    assert tr.shape[0] == te.shape[0] and tr.shape[1] + te.shape[1] == 20


def test_median_ratio_normalisation_removes_compositional_correlation():
    from bsvae.simulation.scenario import normalize_counts
    base = _nb_params(n_features=600, n_samples=80, null_fraction=0.15, signal_scale=2.5, overlap_rate=0.0)
    def null_corr(method):
        r = simulate_params(dict(base, normalization=method), seed=5)
        X = r.expr_features_x_samples.values
        null = r.modules_hard["module"].values == -1
        Xz = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        C = np.corrcoef(Xz[null])
        return float(np.abs(C[np.triu_indices(null.sum(), 1)]).mean())
    assert null_corr("median_ratio") < 0.5 * null_corr("cpm")
    counts = np.random.default_rng(0).poisson(5, size=(10, 50)).astype(float)
    counts[0] *= 4                      # one sample with a 4x larger library
    out = normalize_counts(counts, "median_ratio")
    assert out.shape == counts.shape and np.isfinite(out).all()
    assert abs(out[0].mean() - out[1:].mean()) < 0.3    # library size removed
    with pytest.raises(ValueError):
        normalize_counts(counts, "tmm")


def test_usage_regulation_keeps_gene_total_flat():
    """Usage-regulated isoforms track their module factor while the gene total does not."""
    from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=200, n_modules=5, switch_rate=0.0,
             usage_rate=0.5, usage_strength=1.5, null_fraction=0.1)
    r = simulate_params(p, seed=3)
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    z = r.module_latents.values
    expr = r.expr_features_x_samples
    assert {"mechanism", "is_usage", "usage_strength"} <= set(st.columns)
    usage = st.index[st["is_usage"] == 1]
    assert len(usage) > 20 and (st.loc[usage, "is_switched"] == 1).all() and (st.loc[usage, "mechanism"] == "usage").all()
    assert (st.loc[usage, "isoform_rank"] > 0).all() and (st.loc[usage, "tx_module"] != st.loc[usage, "gene_module"]).all()
    cu, cd, ct = [], [], []
    for t in usage:
        g = st.loc[t, "gene_id"]; m = int(st.loc[t, "tx_module"])
        sib = st.index[st["gene_id"] == g]
        dom = st.index[(st["gene_id"] == g) & (st["isoform_rank"] == 0)][0]
        tot = np.log2(np.power(2.0, expr.loc[sib]).sub(1).clip(lower=0).sum(axis=0) + 1)
        cu.append(np.corrcoef(expr.loc[t], z[:, m])[0, 1])
        cd.append(np.corrcoef(expr.loc[dom], z[:, m])[0, 1])
        ct.append(np.corrcoef(tot, z[:, m])[0, 1])
    assert np.mean(cu) > 0.3 and np.mean(cd) < -0.1 and abs(np.mean(ct)) < 0.15
    assert r.metadata["n_usage_transcripts"] == len(usage)


def test_usage_from_null_and_switch_exclusive():
    from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=60, n_modules=4, switch_rate=0.3,
             usage_rate=0.8, usage_from="null", null_fraction=0.4)
    r = simulate_params(p, seed=5)
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    usage = st[st["is_usage"] == 1]
    assert len(usage) > 10 and (usage["gene_module"] == -1).all() and (usage["tx_module"] >= 0).all()
    # a gene is never both switched and usage-regulated
    sw_genes = set(st.loc[st["mechanism"] == "switch", "gene_id"])
    assert not (set(usage["gene_id"]) & sw_genes)
    with pytest.raises(ValueError):
        simulate_params(dict(p, usage_from="bogus"), seed=1)
