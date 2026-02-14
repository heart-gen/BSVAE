import os
import tempfile
import shutil
import torch
import pandas as pd
import numpy as np
import types
import argparse

import pytest

from bsvae.utils import datasets, helpers, initialization, mapping, modelIO, ppi, training, evaluate
from bsvae.models.structured import StructuredFactorVAE
from bsvae.models.losses import BaseLoss


# --------------------------
# datasets.py
# --------------------------
def test_geneexpression_split_and_presplit(tmp_path):
    # Create fake CSV (genes x samples)
    n_genes = 20
    n_samples = 12
    df = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=[f"gene{i}" for i in range(n_genes)],
        columns=[f"s{i}" for i in range(n_samples)],
    )
    csv_path = tmp_path / "expr.csv"
    df.to_csv(csv_path)

    ds_train = datasets.GeneExpression(gene_expression_filename=str(csv_path),
                                       fold_id=0, train=True)
    ds_test = datasets.GeneExpression(gene_expression_filename=str(csv_path),
                                      fold_id=0, train=False)
    assert len(ds_train) + len(ds_test) == n_samples
    assert ds_train[0][0].shape[0] == n_genes

    # Pre-split mode
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    train_samples = df.columns[:8]
    test_samples = df.columns[8:]
    df[train_samples].to_csv(split_dir / "X_train.csv")
    df[test_samples].to_csv(split_dir / "X_test.csv")

    ds_pre = datasets.GeneExpression(gene_expression_dir=str(split_dir), train=True)
    assert len(ds_pre) == len(train_samples)


def test_geneexpression_supports_compressed(tmp_path):
    n_genes = 20
    n_samples = 12
    df = pd.DataFrame(
        np.random.randn(n_genes, n_samples),
        index=[f"gene{i}" for i in range(n_genes)],
        columns=[f"s{i}" for i in range(n_samples)],
    )

    # Compressed TSV
    tsv_gz_path = tmp_path / "expr.tsv.gz"
    df.to_csv(tsv_gz_path, sep="\t", compression="gzip")

    ds_train = datasets.GeneExpression(
        gene_expression_filename=str(tsv_gz_path), fold_id=0, train=True
    )
    ds_test = datasets.GeneExpression(
        gene_expression_filename=str(tsv_gz_path), fold_id=0, train=False
    )
    assert len(ds_train) + len(ds_test) == n_samples
    assert ds_train[0][0].shape[0] == n_genes

    # Compressed pre-split CSV files
    split_dir = tmp_path / "split_compressed"
    split_dir.mkdir()
    train_samples = df.columns[:8]
    test_samples = df.columns[8:]
    df[train_samples].to_csv(split_dir / "X_train.csv.gz", compression="gzip")
    df[test_samples].to_csv(split_dir / "X_test.csv.gz", compression="gzip")

    ds_pre = datasets.GeneExpression(gene_expression_dir=str(split_dir), train=True)
    assert len(ds_pre) == len(train_samples)


def test_get_dataloaders(tmp_path):
    df = pd.DataFrame(np.random.randn(36, 3),
                      index=[f"g{i}" for i in range(36)],
                      columns=[f"s{i}" for i in range(3)])
    csv_path = tmp_path / "expr.csv"
    df.to_csv(csv_path)
    loader = datasets.get_dataloaders("genenet",
                                      gene_expression_filename=str(csv_path),
                                      batch_size=2)
    batch = next(iter(loader))
    x, g = batch
    assert x.shape[0] == 2


# --------------------------
# helpers.py
# --------------------------
def test_create_safe_directory_and_seed(tmp_path):
    d = tmp_path / "dir"
    os.makedirs(d)
    helpers.create_safe_directory(str(d))  # should archive existing
    assert os.path.exists(d)

    helpers.set_seed(123)
    a = np.random.rand()
    helpers.set_seed(123)
    b = np.random.rand()
    assert np.isclose(a, b)


def test_get_device_and_params():
    model = torch.nn.Linear(4, 2)
    dev = helpers.get_device(use_gpu=False)
    assert isinstance(dev, torch.device)
    assert helpers.get_model_device(model) == next(model.parameters()).device
    assert helpers.get_n_params(model) > 0


def test_update_namespace_and_config(tmp_path):
    ns = types.SimpleNamespace()
    helpers.update_namespace_(ns, {"x": 5})
    assert ns.x == 5

    ini_path = tmp_path / "config.ini"
    with open(ini_path, "w") as f:
        f.write("[sect]\nval = 42\n")
    d = helpers.get_config_section(str(ini_path), "sect")
    assert d["val"] == 42


def test_check_bounds():
    assert helpers.check_bounds("5", int, lb=0, ub=10) == 5
    with pytest.raises(argparse.ArgumentTypeError):
        helpers.check_bounds("20", int, lb=0, ub=10)


# --------------------------
# initialization.py
# --------------------------
@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh", "sigmoid", "linear"])
def test_weights_init_linear(act):
    lin = torch.nn.Linear(10, 5)
    initialization.weights_init(lin, activation=act)


# --------------------------
# mapping.py
# --------------------------
def test_map_genes_to_string_and_subset():
    ann = pd.DataFrame({
        "ENSG": ["e1", "e2"],
        "ENSP": ["p1", "p2"]
    })
    mapping_df = mapping.map_genes_to_string(["e1", "e2"], ann, id_type="ENSG")
    assert "gene_id" in mapping_df

    edges = pd.DataFrame({"protein1": ["p1"], "protein2": ["p2"], "score": [900]})
    edges_sub, gene_list = mapping.subset_genes_and_laplacian(edges, mapping_df)
    assert not edges_sub.empty
    assert gene_list


# --------------------------
# modelIO.py
# --------------------------
def test_save_and_load_model(tmp_path):
    model = StructuredFactorVAE(n_genes=10, n_latent=3, use_batch_norm=False)
    dirpath = tmp_path / "model"
    modelIO.save_model(model, str(dirpath))

    metadata = modelIO.load_metadata(str(dirpath))
    assert metadata["use_batch_norm"] is False

    loaded = modelIO.load_model(str(dirpath), is_gpu=False)
    assert isinstance(loaded, StructuredFactorVAE)
    assert loaded.encoder.use_batch_norm is False

    arrs = {"a": np.array([1, 2, 3])}
    modelIO.save_np_arrays(arrs, str(dirpath), "arrs.json")
    loaded = modelIO.load_np_arrays(str(dirpath), "arrs.json")
    assert np.allclose(loaded["a"], arrs["a"])


def test_load_model_infers_batch_norm_from_state_dict_when_metadata_missing(tmp_path):
    dirpath = tmp_path / "legacy_model"
    model = StructuredFactorVAE(n_genes=10, n_latent=3, use_batch_norm=False)
    modelIO.save_model(model, str(dirpath))

    metadata = modelIO.load_metadata(str(dirpath))
    metadata.pop("use_batch_norm", None)
    modelIO.save_metadata(metadata, str(dirpath))

    loaded = modelIO.load_model(str(dirpath), is_gpu=False)
    assert loaded.encoder.use_batch_norm is False


# --------------------------
# ppi.py
# --------------------------
def test_graph_to_laplacian_and_load(monkeypatch):
    edges = pd.DataFrame({"protein1": ["a"], "protein2": ["b"], "score": [900]})
    G = ppi.nx.Graph()
    G.add_edge("a", "b", weight=1.0)
    L = ppi.graph_to_laplacian(G, ["a", "b"], sparse=False)
    assert torch.is_tensor(L)


# --------------------------
# training.py + evaluate.py
# --------------------------
def test_trainer_and_evaluator(tmp_path):
    x = torch.randn(8, 10)
    dataset = torch.utils.data.TensorDataset(x)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    model = StructuredFactorVAE(n_genes=10, n_latent=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_f = BaseLoss()

    trainer = training.Trainer(model, opt, loss_f, save_dir=str(tmp_path))
    trainer(loader, epochs=1, checkpoint_every=1)

    evaluator = evaluate.Evaluator(model, loss_f, save_dir=str(tmp_path))
    losses = evaluator(loader)
    assert isinstance(losses, dict)
