"""Extra utility tests (hierarchy, modelIO helpers)."""

import tempfile
import os
import numpy as np
import pandas as pd
import pytest
import torch

from bsvae.utils.hierarchy import load_tx2gene, group_isoforms_by_gene
from bsvae.utils.modelIO import save_metadata, load_metadata, numpy_serialize


# ---------------------------------------------------------------------------
# modelIO helpers
# ---------------------------------------------------------------------------

def test_save_load_metadata(tmp_path):
    meta = {"n_features": 100, "n_modules": 10, "latent_dim": 32}
    save_metadata(meta, str(tmp_path))
    loaded = load_metadata(str(tmp_path))
    assert loaded == meta


def test_numpy_serialize_array():
    arr = np.array([1.0, 2.0, 3.0])
    result = numpy_serialize(arr)
    assert result == [1.0, 2.0, 3.0]


def test_numpy_serialize_scalar():
    val = np.float32(3.14)
    result = numpy_serialize(val)
    assert abs(result - 3.14) < 1e-3


def test_numpy_serialize_unknown_type():
    with pytest.raises(TypeError):
        numpy_serialize("not_numpy")


# ---------------------------------------------------------------------------
# hierarchy — edge cases
# ---------------------------------------------------------------------------

def test_load_tx2gene_missing_path_and_ids_raises():
    with pytest.raises(ValueError):
        load_tx2gene(path=None, feature_ids=None)


def test_load_tx2gene_non_ensembl_maps_to_itself():
    feat_ids = ["custom_gene_1", "custom_gene_2"]
    tx2gene = load_tx2gene(feature_ids=feat_ids)
    assert tx2gene["custom_gene_1"] == "custom_gene_1"


def test_group_isoforms_by_gene_empty():
    tx2gene = pd.Series({"tx1": "gA", "tx2": "gB"})  # each gene has 1 isoform
    feature_ids = ["tx1", "tx2"]
    groups = group_isoforms_by_gene(tx2gene, feature_ids)
    assert len(groups) == 0  # no multi-isoform genes


def test_group_isoforms_feature_not_in_tx2gene():
    tx2gene = pd.Series({"tx1": "gA", "tx2": "gA"})
    # feature_ids has an extra feature not in tx2gene
    feature_ids = ["tx1", "tx2", "tx3"]
    groups = group_isoforms_by_gene(tx2gene, feature_ids)
    assert "gA" in groups
    assert len(groups["gA"]) == 2  # only tx1 and tx2 found


# ---------------------------------------------------------------------------
# OmicsDataset HDF5 (requires h5py)
# ---------------------------------------------------------------------------

def test_omics_dataset_hdf5(tmp_path):
    h5py = pytest.importorskip("h5py")
    from bsvae.utils.datasets import OmicsDataset

    n_feat, n_samp = 15, 10
    feat_ids = [f"gene_{i}".encode() for i in range(n_feat)]
    sample_ids = [f"sample_{j}".encode() for j in range(n_samp)]
    data = np.random.randn(n_feat, n_samp).astype(np.float32)

    path = str(tmp_path / "expr.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
        f.create_dataset("features", data=feat_ids)
        f.create_dataset("samples", data=sample_ids)

    ds = OmicsDataset(path)
    assert len(ds) == n_feat
    assert ds.n_samples == n_samp
    profile, fid = ds[0]
    assert profile.shape == (n_samp,)
