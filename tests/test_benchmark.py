"""Tests for the BSVAE-vs-WGCNA benchmark harness and grey-aware metrics."""

import json

import numpy as np
import pandas as pd
import pytest

from bsvae.benchmark.harness import evaluate_run, run_grid, summarize
from bsvae.benchmark.runners import MethodResult, run_bsvae, run_wgcna, wgcna_available
from bsvae.simulation.generate import simulate_omics_data
from bsvae.simulation.metrics import (
    ari_nmi_all,
    ari_nmi_assigned,
    best_match_metrics,
    evaluate_labels,
    gene_level_metrics,
    grey_metrics,
    normalize_labels,
)
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_normalize_labels_handles_strings_and_grey():
    assert normalize_labels(["grey", "M1", "M0", "M1"]).tolist() == [-1, 1, 0, 1]
    assert normalize_labels(np.array([2, -1, 0])).tolist() == [2, -1, 0]
    assert normalize_labels(np.array([1.0, np.nan])).tolist() == [1, -1]


def test_assigned_ari_ignores_grey_but_all_ari_does_not():
    t = np.repeat([0, 1, 2], 10)
    p = t.copy()
    p[::5] = -1                      # 20% grey, otherwise perfect
    assert ari_nmi_assigned(t, p)["ari_assigned"] == pytest.approx(1.0)
    assert ari_nmi_assigned(t, p)["n_dropped"] == 6
    assert ari_nmi_all(t, p)["ari_all"] < 1.0


def test_grey_precision_recall():
    t = np.array([-1, -1, 0, 0, 1, 1])
    p = np.array([-1, 0, -1, 0, 1, 1])
    g = grey_metrics(t, p)
    assert g["grey_precision"] == 0.5 and g["grey_recall"] == 0.5
    assert g["frac_true_grey"] == pytest.approx(1 / 3)


def test_best_match_metrics_perfect_and_split():
    t = np.repeat([0, 1], 20)
    assert best_match_metrics(t, t)["mean_best_match_jaccard"] == 1.0
    p = t.copy(); p[:10] = 2            # module 0 split in half
    m = best_match_metrics(t, p)
    assert m["mean_best_match_jaccard"] == pytest.approx(0.75)
    assert m["module_recovery_50"] == 1.0


def test_gene_level_metrics_split_detection():
    tx_ids = ["g1a", "g1b", "g2a", "g2b", "g3a", "g3b", "g3c"]
    tx2gene = {"g1a": "g1", "g1b": "g1", "g2a": "g2", "g2b": "g2", "g3a": "g3", "g3b": "g3", "g3c": "g3"}
    truth_gene = {"g1": 0, "g2": 1, "g3": 1}
    truth_tx = {"g1a": 0, "g1b": 0, "g2a": 1, "g2b": 1, "g3a": 1, "g3b": 1, "g3c": 0}   # g3c switched
    pred = [0, 0, 1, 1, 1, 1, 0]
    m = gene_level_metrics(pred, tx_ids, tx2gene, truth_gene, truth_tx)
    assert m["gene_ari"] == pytest.approx(1.0)
    assert m["n_switched_genes"] == 1 and m["split_detection_rate"] == 1.0
    assert m["isoform_gene_agreement"] == pytest.approx(2 / 3)
    pred_missed = [0, 0, 1, 1, 1, 1, 1]
    assert gene_level_metrics(pred_missed, tx_ids, tx2gene, truth_gene, truth_tx)["split_detection_rate"] == 0.0


def test_evaluate_labels_keys():
    t = np.repeat([0, 1], 5); p = t.copy()
    m = evaluate_labels(t, p)
    for k in ("ari_all", "ari_assigned", "grey_recall", "mean_best_match_jaccard", "K_ratio"):
        assert k in m


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _block_expr():
    X, gt, _ = simulate_omics_data(n_features=240, n_samples=60, n_modules=4,
                                   within_module_correlation=0.9, seed=0)
    expr = pd.DataFrame(X, index=[f"f{i}" for i in range(240)], columns=[f"s{j}" for j in range(60)])
    return expr, gt


def test_run_wgcna_skips_without_rscript(tmp_path, monkeypatch):
    import bsvae.benchmark.runners as r
    monkeypatch.setattr(r, "wgcna_available", lambda: False)
    expr, _ = _block_expr()
    res = run_wgcna(expr, workdir=tmp_path)
    assert res.status == "skipped" and res.labels is None


@pytest.mark.skipif(not wgcna_available(), reason="Rscript/WGCNA not installed")
@pytest.mark.slow
def test_run_wgcna_recovers_blocks(tmp_path):
    expr, gt = _block_expr()
    res = run_wgcna(expr, workdir=tmp_path, network_type="signed")
    assert res.status == "ok", res.message
    assert set(res.labels.index) == set(expr.index)
    assert res.labels.max() >= 1
    m = evaluate_labels(gt, res.labels.reindex(expr.index).values)
    assert m["ari_assigned"] > 0.8
    assert (tmp_path / "eigengenes.csv").exists() and "soft_power" in res.params


def test_run_bsvae_recovers_blocks(tmp_path):
    expr, gt = _block_expr()
    res = run_bsvae(
        expr, n_modules=4, seed=0, workdir=tmp_path,
        train_cfg={"epochs": 25, "warmup_epochs": 8, "transition_epochs": 4, "hidden_dims": [32, 16],
                   "no_cuda": True, "batch_size": 32},
    )
    assert res.status == "ok", res.message
    assert set(res.labels.index) == set(expr.index)
    m = evaluate_labels(gt, res.labels.reindex(expr.index).values)
    assert m["ari_assigned"] > 0.8
    assert res.gamma.shape[0] == 240 and (tmp_path / "modules.csv").exists()


def test_run_bsvae_rejects_unknown_option(tmp_path):
    expr, _ = _block_expr()
    res = run_bsvae(expr, n_modules=4, seed=0, workdir=tmp_path, train_cfg={"not_an_option": 1})
    assert res.status == "failed" and "not_an_option" in res.message


def test_evaluate_run_requires_feature_overlap():
    truth = {"labels": pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])}
    res = MethodResult("x", "ok", labels=pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"]))
    assert evaluate_run(truth, res)["ari_all"] == 1.0
    bad = MethodResult("x", "ok", labels=pd.Series([0, 1], index=["a", "zz"]))
    with pytest.raises(ValueError, match="truth features"):
        evaluate_run(truth, bad)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def test_run_grid_isoform_resume_and_summary(tmp_path):
    cfg = {
        "base_seed": 1, "reps": 1,
        "generator": dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=120, n_samples=30, n_modules=3),
        "grid": {"alpha": [1.0, 0.5]},
        "outputs": {"emit_gnvae_folds": False},
    }
    bs = {"epochs": 6, "warmup_epochs": 2, "transition_epochs": 1, "hidden_dims": [16], "no_cuda": True}
    df = run_grid(cfg, tmp_path, methods=("bsvae",), seeds=[0, 1], bsvae_cfg=bs)
    assert len(df) == 4 and set(df["method"]) == {"bsvae"}
    for col in ("scenario_id", "rep", "seed", "status", "ari_all", "gene_ari", "split_detection_rate", "p_alpha"):
        assert col in df.columns
    assert (df["status"] == "ok").all()
    assert (tmp_path / "summary.tsv").exists()
    s = summarize(df)
    assert len(s) == 2 and (s["n_runs"] == 2).all()
    # Per-run artefacts
    run_dir = tmp_path / "scenarios" / df["scenario_id"].iloc[0] / "rep_000" / "methods" / "bsvae" / "seed_0"
    assert (run_dir / "metrics.json").exists() and (run_dir / "modules.csv").exists()
    # Resume adds no rows; adding a seed adds exactly two.
    df2 = run_grid(cfg, tmp_path, methods=("bsvae",), seeds=[0, 1], bsvae_cfg=bs)
    assert len(df2) == 4
    df3 = run_grid(cfg, tmp_path, methods=("bsvae",), seeds=[0, 1, 2], bsvae_cfg=bs)
    assert len(df3) == 6


@pytest.mark.skipif(not wgcna_available(), reason="Rscript/WGCNA not installed")
def test_wgcna_script_parses():
    import subprocess
    from bsvae.benchmark.runners import WGCNA_SCRIPT
    proc = subprocess.run(["Rscript", "--vanilla", "-e", f'invisible(parse(file="{WGCNA_SCRIPT}"))'],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr


def test_over_cluster_k():
    from bsvae.benchmark.runners import over_cluster_k
    assert over_cluster_k(10) == 30
    assert over_cluster_k(4) == 20          # floor
    assert over_cluster_k(10, factor=1.0, minimum=1) == 10
    assert over_cluster_k(4, n_features=240) == 9     # capped at 25 features per component
    assert over_cluster_k(4, n_features=50) == 4      # never below n_modules


def test_run_bsvae_over_clusters_then_merges(tmp_path):
    expr, gt = _block_expr()
    res = run_bsvae(
        expr, n_modules=4, seed=0, workdir=tmp_path,
        train_cfg={"epochs": 25, "warmup_epochs": 8, "transition_epochs": 4, "hidden_dims": [32, 16],
                   "no_cuda": True, "batch_size": 32},
        over_cluster_factor=3.0,
    )
    assert res.status == "ok", res.message
    assert res.params["n_components_trained"] == 9 and res.params["n_modules_expected"] == 4
    m = evaluate_labels(gt, res.labels.reindex(expr.index).values)
    assert m["ari_assigned"] > 0.8
    assert m["n_modules_pred"] <= 6


def test_switched_isoform_metrics_toy():
    from bsvae.simulation.metrics import switched_isoform_metrics
    st = pd.DataFrame({
        "transcript_id": ["a", "b", "c", "d", "e"],
        "is_switched": [1, 1, 0, 0, 1],
        "tx_module": [2, 3, 0, 0, 2],
    })
    pred = {"a": 0, "b": 1, "c": -1, "d": 0, "e": 0}     # d is a false call; b wrongly separated? no: b→1, a,e→0
    m = switched_isoform_metrics(pred, list(pred), st)
    assert m["n_switched_true"] == 3 and m["n_called"] == 4
    assert m["switched_recall"] == 1.0 and m["switched_precision"] == 0.75
    assert m["switched_ari"] == pytest.approx(1.0)


def test_run_grid_residual_and_consensus_methods(tmp_path):
    cfg = {
        "base_seed": 1, "reps": 1,
        "generator": dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=150, n_samples=30, n_modules=3, switch_rate=0.3),
        "grid": {"alpha": [1.0]},
        "outputs": {"emit_gnvae_folds": False},
    }
    bs = {"epochs": 6, "warmup_epochs": 2, "transition_epochs": 1, "hidden_dims": [16], "no_cuda": True}
    df = run_grid(cfg, tmp_path, methods=("bsvae_residual", "bsvae_consensus"), seeds=[0], bsvae_cfg=bs, consensus_runs=2)
    assert set(df["method"]) == {"bsvae_residual", "bsvae_consensus"}
    assert (df["status"] == "ok").all()
    res = df[df["method"] == "bsvae_residual"].iloc[0]
    assert np.isfinite(res["switched_recall"]) and "switched_precision" in df.columns
    cons = df[df["method"] == "bsvae_consensus"].iloc[0]
    assert cons["m_n_runs"] == 2 and np.isfinite(cons["ari_all"])
