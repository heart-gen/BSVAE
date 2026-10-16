import numpy as np
import pandas as pd
import pytest

from bsvae.benchmark.edges import edge_metrics, edges_from_labels
from bsvae.benchmark.methods import (
    acorde_available, isoform_usage, run_acorde, run_ica, run_nmf, run_twn, run_usage_network, twn_available, twn_inputs,
)
from bsvae.simulation.metrics import evaluate_labels, normalize_labels
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params


@pytest.fixture(scope="module")
def iso():
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=250, n_samples=100, n_modules=5, switch_rate=0.2,
             alpha=1.0, signal_scale=2.5, null_fraction=0.1)
    r = simulate_params(p, seed=11)
    expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values
    truth_gene = dict(zip(r.extra_truth["modules_gene.csv"]["gene_id"].astype(str), r.extra_truth["modules_gene.csv"]["module"].astype(int)))
    return expr, t2g, truth, truth_gene, r.extra_truth["switch_table.tsv.gz"], r.edge_list


def _ari(truth, res, expr):
    lab = res.labels.reindex(expr.index).fillna(-1).astype(int).values
    return evaluate_labels(truth, normalize_labels(lab))["ari_assigned"], lab


def test_isoform_usage_shares_sum_to_one_and_exclude_singletons(iso):
    expr, t2g, *_ = iso
    u = isoform_usage(expr, t2g)
    genes = pd.Series([t2g[t] for t in u.index], index=u.index)
    assert (genes.value_counts() >= 2).all()
    share = 1 / (1 + np.exp(-u.values))
    tot = pd.DataFrame(share, index=genes.values, columns=u.columns).groupby(level=0).sum()
    # shares sum to one wherever the gene is expressed in a sample (zero totals clip every isoform to eps)
    lin = np.maximum(np.power(2.0, expr.reindex(u.index).values) - 1.0, 0.0)
    gene_tot = pd.DataFrame(lin, index=genes.values, columns=u.columns).groupby(level=0).sum()
    expressed = gene_tot.values > 0
    assert np.allclose(tot.values[expressed], 1.0, atol=0.05)
    assert np.allclose(tot.values[~expressed], 0.0, atol=0.05)


def test_ica_and_nmf_recover_modules(iso, tmp_path):
    expr, t2g, truth, *_ = iso
    for fn in (run_ica, run_nmf):
        res = fn(expr, n_components=5, workdir=tmp_path / fn.__name__, seed=0)
        assert res.status == "ok"
        ari, lab = _ari(truth, res, expr)
        assert ari > 0.6
        assert (tmp_path / fn.__name__ / "modules.csv").exists()
        assert res.params["n_modules"] >= 2
    small = run_ica(expr, n_components=5, workdir=tmp_path / "ica_small", seed=0, min_module_size=10_000)
    assert (small.labels.values < 0).all()


@pytest.mark.skipif(not acorde_available(), reason="Rscript/dynamicTreeCut not available")
def test_acorde_runner(iso, tmp_path):
    expr, t2g, truth, *_ = iso
    res = run_acorde(expr, workdir=tmp_path / "acorde", seed=0)
    assert res.status == "ok", res.message
    ari, lab = _ari(truth, res, expr)
    assert ari > 0.5 and (lab >= 0).mean() > 0.5
    assert res.params["lowcor_threshold"] is None
    strict = run_acorde(expr, workdir=tmp_path / "acorde_strict", seed=0, lowcor_threshold=3, expand=False)
    assert strict.status == "ok" and (strict.labels.values < 0).mean() > (lab < 0).mean()


@pytest.mark.skipif(not acorde_available(), reason="Rscript/WGCNA not available")
def test_usage_network_runner(iso, tmp_path):
    from bsvae.benchmark.runners import wgcna_available
    if not wgcna_available():
        pytest.skip("WGCNA not available")
    expr, t2g, truth, *_ = iso
    res = run_usage_network(expr, t2g, workdir=tmp_path / "usage", seed=0)
    assert res.status == "ok", res.message
    lab = res.labels.reindex(expr.index).fillna(-1).astype(int)
    genes = pd.Series([t2g[t] for t in lab.index], index=lab.index)
    single = genes.map(genes.value_counts()) == 1
    assert (lab[single] == -1).all()
    assert res.params["n_usage_transcripts"] == int((~single).sum())


def test_twn_inputs_shapes(iso):
    expr, t2g, *_ = iso
    te, ir = twn_inputs(expr, t2g)
    assert te.shape[0] == expr.shape[1] and ir.shape[0] == expr.shape[1]
    assert te.shape[1] == len(set(t2g.values()))
    assert ((ir.values > 0) & (ir.values < 1)).all()


@pytest.mark.skipif(not twn_available(), reason="Rscript/glasso not available")
@pytest.mark.slow
def test_twn_runner_and_edge_metrics(iso, tmp_path):
    expr, t2g, truth, truth_gene, st, edge_list = iso
    res = run_twn(expr, t2g, workdir=tmp_path / "twn", lambda_tt=0.6, lambda_ti=0.5, lambda_ii=0.5)
    assert res.status == "ok", res.message
    e = res.edges
    assert set(e.columns) == {"node1", "node2", "edge_type", "weight"} and set(e["edge_type"]) <= {1, 2, 3}
    # within-gene edges were removed
    gene_of = lambda n: t2g.get(n, n)
    assert all(gene_of(a) != gene_of(b) for a, b in zip(e["node1"], e["node2"]))
    m = edge_metrics(e, truth_gene, st, edge_list, t2g)
    assert m["tt_precision"] > 0.8 and 0 <= m["tt_recall"] <= 1
    assert m["ti_switched_recall"] > m["ti_unswitched_false_rate"]


def test_edges_from_labels_and_metrics_on_truth(iso):
    expr, t2g, truth, truth_gene, st, edge_list = iso
    labels = pd.Series(truth, index=expr.index)
    e = edges_from_labels(expr, labels, t2g, top_k=5)
    assert (e["edge_type"] == 1).all() and len(e) > 0
    m = edge_metrics(e, truth_gene, None, edge_list, t2g)
    assert m["tt_precision"] == pytest.approx(1.0)
    assert 0.05 < m["tt_recall"] <= 1.0          # top-5 edges per gene against the simulator's top-25 truth
    # a shuffled labelling keeps some precision (top correlated pairs are still co-expressed) but
    # loses recall and precision relative to the truth labelling
    rng = np.random.default_rng(0)
    e2 = edges_from_labels(expr, pd.Series(rng.permutation(truth), index=expr.index), t2g, top_k=5)
    m2 = edge_metrics(e2, truth_gene, None, edge_list, t2g)
    assert m2["tt_precision"] < m["tt_precision"] and m2["tt_recall"] < m["tt_recall"]
