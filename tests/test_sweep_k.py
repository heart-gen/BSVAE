"""End-to-end tests for bsvae-sweep-k on a toy dataset."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from bsvae.cli import sweep_k


@pytest.fixture
def toy_isoform_dataset(tmp_path):
    rng = np.random.default_rng(0)
    n_genes, n_samples = 12, 10
    tx_ids, gene_ids, rows = [], [], []
    for g in range(n_genes):
        base = rng.normal(size=n_samples)
        for i in range(2):
            tx_ids.append(f"ENST{g:05d}{i}.1")
            gene_ids.append(f"ENSG{g:05d}")
            rows.append(base + 0.1 * rng.normal(size=n_samples))
    df = pd.DataFrame(np.array(rows, dtype=np.float32), index=tx_ids,
                      columns=[f"s{j}" for j in range(n_samples)])
    expr = tmp_path / "expr.csv"
    df.to_csv(expr)
    tx2gene = tmp_path / "tx2gene.tsv"
    pd.DataFrame({"transcript_id": tx_ids, "gene_id": gene_ids}).to_csv(tx2gene, sep="\t", index=False)
    return str(expr), str(tx2gene), tx_ids, gene_ids


def test_split_feature_indices_gene_aware():
    ids = [f"t{i}" for i in range(10)]
    groups = {"gA": [0, 1, 2], "gB": [3, 4]}
    train, val = sweep_k._split_feature_indices(ids, 0.3, seed=1, gene_groups=groups)
    assert sorted(train + val) == list(range(10))
    for idxs in groups.values():
        in_val = [i in val for i in idxs]
        assert all(in_val) or not any(in_val)   # whole gene on one side


def test_sweep_k_end_to_end(toy_isoform_dataset, tmp_path):
    expr, tx2gene, tx_ids, gene_ids = toy_isoform_dataset
    args = sweep_k.parse_args([
        "sweep", "--dataset", expr, "--outdir", str(tmp_path), "--no-cuda", "--no-progress-bar",
        "--k-grid", "2,3", "--sweep-epochs", "4", "--warmup-epochs", "2", "--transition-epochs", "1",
        "--latent-dim", "4", "--hidden-dims", "[8]", "--batch-size", "4", "--val-frac", "0.25",
        "--tx2gene", tx2gene, "--final-epochs", "3",
    ])
    sweep_k.main(args)
    root = tmp_path / "sweep" / "sweep_k"
    summary = json.load(open(root / "sweep_summary.json"))
    assert summary["selected_k"] in (2, 3)
    assert summary["selection_metric"] == "val_elbo"
    assert summary["gene_aware_split"] is True
    res = pd.read_csv(root / "sweep_results.csv")
    assert list(res["k"]) == [2, 3]
    assert np.isfinite(res["val_elbo_mean"]).all()
    final = tmp_path / "sweep" / f"final_k{summary['selected_k']}"
    assert (final / "model.pt").exists()


def test_sweep_k_stability_selection(toy_isoform_dataset, tmp_path):
    expr, tx2gene, _, _ = toy_isoform_dataset
    args = sweep_k.parse_args([
        "sweep", "--dataset", expr, "--outdir", str(tmp_path), "--no-cuda", "--no-progress-bar",
        "--k-grid", "2", "--sweep-epochs", "3", "--warmup-epochs", "1", "--transition-epochs", "1",
        "--latent-dim", "4", "--hidden-dims", "[8]", "--batch-size", "4", "--val-frac", "0.25",
        "--stability-reps", "2", "--no-train-final",
    ])
    sweep_k.main(args)
    summary = json.load(open(tmp_path / "sweep" / "sweep_k" / "sweep_summary.json"))
    assert summary["selection_metric"] == "stability_ari"
    res = pd.read_csv(tmp_path / "sweep" / "sweep_k" / "sweep_results.csv")
    assert np.isfinite(res["stability_ari_mean"]).all()
