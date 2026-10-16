"""Tests for WGCNA-style post-processing of GMM assignments."""

import json

import numpy as np
import pandas as pd
import pytest

from bsvae.networks.postprocess import (
    GREY,
    assign_with_confidence,
    compute_kme,
    enforce_min_module_size,
    kme_reassign,
    merge_close_modules,
    module_eigengenes,
    postprocess_assignments,
    save_postprocess,
)
from bsvae.networks.module_extraction import compute_module_eigengenes_from_soft
from bsvae.simulation.metrics import compute_ari


def _block_expr(sizes=(30, 30, 30), n_samples=40, noise=0.3, seed=0):
    """Features × samples with one latent factor per block."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for k, sz in enumerate(sizes):
        z = rng.normal(size=n_samples)
        for _ in range(sz):
            rows.append(2.0 * z + noise * rng.normal(size=n_samples) + rng.normal())
            labels.append(k)
    fids = [f"f{i}" for i in range(len(rows))]
    expr = pd.DataFrame(np.array(rows), index=fids, columns=[f"s{j}" for j in range(n_samples)])
    return expr, np.array(labels), fids


def test_assign_with_confidence():
    gamma = np.array([[0.9, 0.1], [0.45, 0.55], [0.5, 0.5]])
    assert assign_with_confidence(gamma, min_gamma=0.6).tolist() == [0, GREY, GREY]
    assert assign_with_confidence(gamma, min_gamma=0.5).tolist() == [0, 1, 0]
    assert assign_with_confidence(gamma, min_gamma=0.0).tolist() == [0, 1, 0]


def test_enforce_min_module_size_next_best_and_grey():
    gamma = np.zeros((19, 3))
    gamma[:10, 0] = 0.9; gamma[:10, 1] = 0.1
    gamma[10:16, 1] = 0.9; gamma[10:16, 0] = 0.1
    gamma[16:, 2] = 0.6; gamma[16:, 1] = 0.3; gamma[16:, 0] = 0.1
    labels = gamma.argmax(1)
    out = enforce_min_module_size(labels, gamma, min_module_size=5, min_gamma=0.5)
    # Module 2 (3 members) dissolves; renormalised over live {0,1}: 1 gets 0.75 → next best.
    assert (out[:10] == 0).all() and (out[10:16] == 1).all()
    assert (out[16:] == 1).all()
    out_grey = enforce_min_module_size(labels, gamma, min_module_size=5, reassign="grey")
    assert (out_grey[16:] == GREY).all()
    # Below min_gamma after renormalisation → grey.
    gamma2 = gamma.copy(); gamma2[16:, 1] = 0.2; gamma2[16:, 0] = 0.2
    out2 = enforce_min_module_size(labels, gamma2, min_module_size=5, min_gamma=0.6)
    assert (out2[16:] == GREY).all()
    # Components that never had members are not reassignment targets.
    gamma3 = np.zeros((13, 3))
    gamma3[:10, 0] = 0.9; gamma3[:10, 1] = 0.1
    gamma3[10:, 2] = 0.6; gamma3[10:, 1] = 0.3; gamma3[10:, 0] = 0.1
    out3 = enforce_min_module_size(gamma3.argmax(1), gamma3, min_module_size=5, min_gamma=0.5)
    assert (out3 == 0).all()


def test_module_eigengenes_sign_and_nan():
    expr, labels, fids = _block_expr(sizes=(20, 1))
    eg = module_eigengenes(expr, labels, fids)
    assert list(eg.columns) == [0, 1]
    mean_profile = ((expr.values[:20] - expr.values[:20].mean(1, keepdims=True))
                    / expr.values[:20].std(1, ddof=1, keepdims=True)).mean(0)
    assert np.corrcoef(eg[0].values, mean_profile)[0, 1] > 0.9
    assert eg[1].isna().all()
    assert abs(np.linalg.norm(eg[0].values) - 1.0) < 1e-6


def test_compute_kme_within_vs_across():
    expr, labels, fids = _block_expr(sizes=(25, 25))
    eg = module_eigengenes(expr, labels, fids)
    kme = compute_kme(expr, eg, fids)
    own = np.array([kme.iloc[i, labels[i]] for i in range(len(fids))])
    other = np.array([kme.iloc[i, 1 - labels[i]] for i in range(len(fids))])
    assert own.min() > 0.8
    assert np.abs(other).max() < 0.5


def test_merge_close_modules_merges_split_block_only():
    expr, labels, fids = _block_expr(sizes=(40, 30))
    split = labels.copy()
    split[:20] = 2                       # true block 0 split into labels 0 and 2
    merged, module_map, log = merge_close_modules(expr, split, fids, cut_height=0.25)
    assert compute_ari(labels, merged) == pytest.approx(1.0)
    assert len(log) == 1 and log[0]["cor"] > 0.75
    assert module_map[0] == module_map[2] and module_map[1] != module_map[0]
    assert sorted(set(merged)) == [0, 1]


def test_kme_reassign_moves_misassigned_feature():
    expr, labels, fids = _block_expr(sizes=(30, 30))
    wrong = labels.copy()
    wrong[0] = 1                         # feature 0 belongs to block 0
    fixed = kme_reassign(expr, wrong, fids, min_kme=0.3, kme_margin=0.1)
    assert fixed[0] == 0
    noise = expr.copy()
    noise.loc["f5"] = np.random.default_rng(1).normal(size=expr.shape[1])
    out = kme_reassign(noise, labels, fids, min_kme=0.3)
    assert out[5] == GREY


def test_postprocess_end_to_end_improves_on_raw_argmax(tmp_path):
    expr, labels, fids = _block_expr(sizes=(40, 35, 30), n_samples=50)
    rng = np.random.default_rng(3)
    K = 5
    gamma = np.full((len(fids), K), 0.02)
    for i, k in enumerate(labels):
        gamma[i, k] = 0.9
    # Split block 0 across components 0 and 3, and make 4 noisy features low-confidence.
    gamma[:15, 3] = gamma[:15, 0]; gamma[:15, 0] = 0.02
    gamma[-4:, :] = 1.0 / K + rng.normal(0, 0.01, size=(4, K))
    gamma = gamma / gamma.sum(1, keepdims=True)
    raw = gamma.argmax(1)
    res = postprocess_assignments(gamma, fids, expr, min_gamma=0.5, min_module_size=5,
                                  merge_cut_height=0.25)
    keep = res.labels >= 0
    assert res.n_grey == 4
    assert compute_ari(labels[keep], res.labels[keep]) > compute_ari(labels, raw)
    assert compute_ari(labels[keep], res.labels[keep]) == pytest.approx(1.0)
    assert res.gamma.shape == (len(fids), res.n_modules)
    assert (res.gamma.sum(1) <= 1.0 + 1e-6).all() and res.gamma.sum(1).min() > 0.5
    assert res.eigengenes.shape[1] == res.n_modules and res.kme.shape == (len(fids), res.n_modules)
    save_postprocess(res, tmp_path, fids)
    mods = pd.read_csv(tmp_path / "modules.csv")
    assert set(mods.columns) == {"feature_id", "module", "module_label"}
    assert (mods.loc[mods["module"] == -1, "module_label"] == "grey").all()
    log = json.load(open(tmp_path / "merge_log.json"))
    assert log["n_grey"] == 4 and len(log["steps"]) == 1
    assert (tmp_path / "kME.csv").exists() and (tmp_path / "eigengenes.csv").exists()


def test_postprocess_without_expression_skips_merge():
    gamma = np.array([[0.9, 0.1]] * 12 + [[0.1, 0.9]] * 12)
    res = postprocess_assignments(gamma, [f"f{i}" for i in range(24)], None, min_module_size=5)
    assert res.eigengenes is None and res.kme is None and res.merge_log == []
    assert sorted(set(res.labels)) == [0, 1]


def test_soft_eigengenes_standardised_and_aligned():
    expr, labels, fids = _block_expr(sizes=(20, 20))
    expr.loc["f0"] = expr.loc["f0"] * 1000      # a hugely scaled feature must not dominate
    gamma = np.zeros((40, 2)); gamma[np.arange(40), labels] = 1.0
    eg = compute_module_eigengenes_from_soft(expr, gamma, fids)
    assert eg.shape == (expr.shape[1], 2)
    z = ((expr.values[:20] - expr.values[:20].mean(1, keepdims=True))
         / expr.values[:20].std(1, ddof=1, keepdims=True)).mean(0)
    assert np.corrcoef(eg["module_0"], z)[0, 1] > 0.9
    gamma_empty = np.zeros((40, 3)); gamma_empty[np.arange(40), labels] = 1.0
    eg3 = compute_module_eigengenes_from_soft(expr, gamma_empty, fids)
    assert eg3["module_2"].isna().all()


def test_align_labels_matches_by_overlap():
    from bsvae.networks.postprocess import align_labels
    ref = np.array([0, 0, 0, 1, 1, 1, 2, 2, -1])
    lab = np.array([5, 5, 5, 7, 7, 7, 9, 9, -1])     # permuted ids
    assert align_labels(ref, lab).tolist() == ref.tolist()
    lab2 = np.array([5, 5, 5, 7, 7, 7, 3, 3, 3])      # module 3 also covers ref's grey feature
    out = align_labels(ref, lab2)
    assert out[:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert out[6] == out[7] == out[8] == 2             # best overlap with ref module 2


def test_consensus_recovers_truth_and_flags_unstable_features():
    from bsvae.networks.postprocess import consensus_assignments
    rng = np.random.default_rng(0)
    truth = np.repeat([0, 1, 2], 40)
    runs = []
    for r in range(5):
        perm = rng.permutation(3)
        lab = perm[truth]
        noise = rng.choice(120, size=6, replace=False)      # 6 random flips per run
        lab[noise] = rng.integers(0, 3, size=6)
        runs.append(lab)
    runs[0][:5] = -1                                       # grey in one run only
    res = consensus_assignments(runs, min_agreement=0.5, min_module_size=5)
    from bsvae.simulation.metrics import compute_ari
    assert compute_ari(truth, res.labels) > 0.95
    assert res.agreement.min() >= 0.4 and res.agreement.mean() > 0.9
    assert all(a > 0.8 for a in res.run_ari) and res.n_runs == 5
    # A feature the runs split evenly on falls below a 0.6 agreement floor → grey.
    runs2 = [np.array([0, 0, 1, 1, i % 2]) for i in range(4)]
    res2 = consensus_assignments(runs2, min_agreement=0.6, min_module_size=1)
    assert res2.labels[4] == -1 and res2.labels[:4].tolist() == [0, 0, 1, 1]
    assert res2.agreement[4] == pytest.approx(0.5)


def test_consensus_cli_roundtrip(tmp_path):
    from bsvae.cli.networks import build_parser, handle_consensus_modules
    import logging
    fids = [f"f{i}" for i in range(30)]
    truth = np.repeat([0, 1, 2], 10)
    paths = []
    for r in range(3):
        lab = ((truth + r) % 3) if r else truth
        p = tmp_path / f"run{r}.csv"
        pd.DataFrame({"feature_id": fids, "module": lab}).to_csv(p, index=False)
        paths.append(str(p))
    args = build_parser().parse_args(["consensus-modules", "--modules", *paths, "--output-dir", str(tmp_path / "out"),
                                      "--min-module-size", "5"])
    handle_consensus_modules(args, logging.getLogger("t"))
    out = pd.read_csv(tmp_path / "out" / "consensus_modules.csv")
    assert set(out.columns) == {"feature_id", "module", "module_label", "agreement"}
    from bsvae.simulation.metrics import compute_ari
    assert compute_ari(truth, out["module"].values) == pytest.approx(1.0)
    assert json.load(open(tmp_path / "out" / "consensus_summary.json"))["n_runs"] == 3



def test_gene_coherence_moves_unsupported_divergence_and_keeps_switches():
    """Unswitched isoforms placed apart from their gene by abundance noise return to the gene's
    module; a truly switched isoform (residual tracks the other module) stays."""
    from bsvae.networks.postprocess import enforce_gene_coherence
    from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=120, n_modules=4, switch_rate=0.3,
             alpha=1.0, signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=8)
    expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values.copy()
    # corrupt: move 40 unswitched minor isoforms to a random other module
    rng = np.random.default_rng(0)
    minor = np.where((st.loc[expr.index, "isoform_rank"].values > 0) & (st.loc[expr.index, "is_switched"].values == 0))[0]
    bad = rng.choice(minor, size=40, replace=False)
    corrupted = truth.copy()
    for i in bad:
        corrupted[i] = rng.choice([m for m in range(4) if m != truth[i]])
    fixed, table = enforce_gene_coherence(corrupted, expr, list(expr.index), t2g)
    assert set(table.columns) == {"gene_id", "gene_module", "module", "kme_residual", "p_value", "q_value", "kept"}
    assert (fixed[bad] == truth[bad]).mean() > 0.8          # noise-driven divergence undone
    sw = np.where(st.loc[expr.index, "is_switched"].values == 1)[0]
    assert (fixed[sw] == truth[sw]).mean() > 0.7            # real switches survive
    # labels never become grey and non-candidates are untouched
    untouched = np.setdiff1d(np.arange(len(truth)), np.concatenate([bad, sw]))
    assert (fixed[untouched] == corrupted[untouched]).all()
    # no candidates → unchanged
    same, tab = enforce_gene_coherence(truth[truth >= 0][:0], expr.iloc[:0], [], t2g)
    assert len(tab) == 0
