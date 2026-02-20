"""Tests for utility modules: datasets, helpers, hierarchy, initialization, modelIO."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from bsvae.utils.helpers import (
    set_seed,
    get_device,
    get_n_params,
    create_safe_directory,
    check_bounds,
)
from bsvae.utils.initialization import weights_init
from bsvae.utils.datasets import OmicsDataset, get_omics_dataloader
from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene, IsoformStratifiedSampler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def test_set_seed_reproducibility():
    set_seed(42)
    x = torch.randn(5)
    set_seed(42)
    y = torch.randn(5)
    assert torch.allclose(x, y)


def test_get_device_returns_device():
    device = get_device(use_gpu=False)
    assert device == torch.device("cpu")


def test_get_n_params():
    import torch.nn as nn
    model = nn.Linear(10, 5)
    assert get_n_params(model) == 10 * 5 + 5


def test_create_safe_directory(tmp_path):
    d = str(tmp_path / "testdir")
    create_safe_directory(d)
    assert os.path.isdir(d)


def test_check_bounds_valid():
    val = check_bounds("3.5", float, lb=0.0, ub=10.0, name="x")
    assert abs(val - 3.5) < 1e-9


def test_check_bounds_out_of_range():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        check_bounds("-1.0", float, lb=0.0, ub=10.0, name="x")


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------

def test_weights_init_relu():
    import torch.nn as nn
    layer = nn.Linear(20, 10)
    weights_init(layer, "relu")
    assert layer.bias.abs().sum().item() == 0.0  # zeros


def test_weights_init_linear():
    import torch.nn as nn
    layer = nn.Linear(20, 10)
    weights_init(layer, "linear")
    assert layer.bias.abs().sum().item() == 0.0


# ---------------------------------------------------------------------------
# OmicsDataset — CSV format
# ---------------------------------------------------------------------------

def _make_csv_expr(tmp_path, n_features=20, n_samples=15):
    feat_ids = [f"gene_{i}" for i in range(n_features)]
    sample_ids = [f"sample_{j}" for j in range(n_samples)]
    df = pd.DataFrame(
        np.random.randn(n_features, n_samples),
        index=feat_ids,
        columns=sample_ids,
    )
    path = str(tmp_path / "expr.csv")
    df.to_csv(path)
    return path, feat_ids, sample_ids


def test_omics_dataset_csv_load(tmp_path):
    path, feat_ids, sample_ids = _make_csv_expr(tmp_path)
    ds = OmicsDataset(path)
    assert len(ds) == len(feat_ids)
    assert ds.n_samples == len(sample_ids)


def test_omics_dataset_getitem(tmp_path):
    path, feat_ids, sample_ids = _make_csv_expr(tmp_path)
    ds = OmicsDataset(path)
    profile, fid = ds[0]
    assert profile.shape == (len(sample_ids),)
    assert isinstance(fid, str)


def test_omics_dataset_tsv_load(tmp_path):
    feat_ids = [f"gene_{i}" for i in range(10)]
    sample_ids = [f"s_{j}" for j in range(8)]
    df = pd.DataFrame(np.random.randn(10, 8), index=feat_ids, columns=sample_ids)
    path = str(tmp_path / "expr.tsv")
    df.to_csv(path, sep="\t")
    ds = OmicsDataset(path)
    assert len(ds) == 10
    assert ds.n_samples == 8


def test_omics_dataset_feature_subset(tmp_path):
    path, feat_ids, _ = _make_csv_expr(tmp_path, n_features=20)
    subset = feat_ids[:5]
    ds = OmicsDataset(path, feature_subset=subset)
    assert len(ds) == 5
    assert ds.feature_ids == subset


def test_get_omics_dataloader(tmp_path):
    path, _, _ = _make_csv_expr(tmp_path)
    loader = get_omics_dataloader(path, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    profiles, fids = batch
    assert profiles.shape[1] == 15  # n_samples


# ---------------------------------------------------------------------------
# hierarchy
# ---------------------------------------------------------------------------

def test_load_tx2gene_from_path(tmp_path):
    df = pd.DataFrame({
        "transcript_id": ["ENST001", "ENST002", "ENST003"],
        "gene_id": ["ENSG001", "ENSG001", "ENSG002"],
    })
    path = str(tmp_path / "tx2gene.tsv")
    df.to_csv(path, sep="\t", index=False)
    tx2gene = load_tx2gene(path=path)
    assert tx2gene["ENST001"] == "ENSG001"
    assert tx2gene["ENST002"] == "ENSG001"


def test_load_tx2gene_infer_from_ensembl():
    feat_ids = ["ENST00000001", "ENST00000002", "ENST00000003"]
    tx2gene = load_tx2gene(feature_ids=feat_ids)
    assert tx2gene["ENST00000001"] == "ENSG00000001"


def test_group_isoforms_by_gene():
    tx2gene = pd.Series({
        "tx1": "geneA", "tx2": "geneA", "tx3": "geneB", "tx4": "geneC",
    })
    feature_ids = ["tx1", "tx2", "tx3", "tx4"]
    groups = group_isoforms_by_gene(tx2gene, feature_ids)
    assert "geneA" in groups
    assert len(groups["geneA"]) == 2
    # geneB and geneC have only 1 isoform — excluded
    assert "geneB" not in groups
    assert "geneC" not in groups


def test_isoform_stratified_sampler():
    multi_idx = list(range(10))  # features 0-9 are multi-isoform
    sampler = IsoformStratifiedSampler(
        n_features=50,
        multi_isoform_indices=multi_idx,
        batch_size=8,
        p_multi=1.0,     # always draw from multi-isoform pool
        num_samples=24,
    )
    indices = list(iter(sampler))
    assert len(indices) == 24
    assert all(idx < 10 for idx in indices)  # only from multi pool


def test_isoform_stratified_sampler_uniform():
    sampler = IsoformStratifiedSampler(
        n_features=50,
        multi_isoform_indices=list(range(10)),
        batch_size=8,
        p_multi=0.0,     # never draw from multi pool
        num_samples=16,
    )
    indices = list(iter(sampler))
    assert len(indices) == 16
    # All should be in range 0..49 (drawn from full pool)
    assert all(0 <= idx < 50 for idx in indices)
