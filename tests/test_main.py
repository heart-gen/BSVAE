"""Tests for bsvae-train CLI (cli/train.py)."""

import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from bsvae.cli.train import parse_args, main


@pytest.fixture
def fake_isoform_dataset(tmp_path):
    """Toy expression matrix with paired ENST feature IDs (10 genes × 2 isoforms each)."""
    feat_ids = [f"ENST{i:011d}" for i in range(20)]
    df = pd.DataFrame(
        np.random.randn(20, 8),
        index=feat_ids,
        columns=[f"sample_{j}" for j in range(8)],
    )
    path = tmp_path / "isoform_expr.csv"
    df.to_csv(path)
    return str(path)


@pytest.fixture
def fake_tx2gene(tmp_path):
    """tx2gene TSV: consecutive pairs of ENST IDs share a gene."""
    rows = []
    for i in range(0, 20, 2):
        tx1, tx2 = f"ENST{i:011d}", f"ENST{i+1:011d}"
        gene = f"ENSG{i//2:011d}"
        rows += [(tx1, gene), (tx2, gene)]
    df = pd.DataFrame(rows, columns=["transcript_id", "gene_id"])
    path = tmp_path / "tx2gene.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


@pytest.fixture
def fake_dataset_csv(tmp_path):
    """Create a toy expression matrix CSV (features × samples)."""
    df = pd.DataFrame(
        np.random.randn(10, 6),
        index=[f"gene_{i}" for i in range(10)],
        columns=[f"sample_{j}" for j in range(6)],
    )
    path = tmp_path / "expr.csv"
    df.to_csv(path)
    return str(path)


def test_parse_args_defaults(fake_dataset_csv, tmp_path):
    args = parse_args([
        "my_experiment",
        "--dataset", fake_dataset_csv,
        "--outdir", str(tmp_path),
        "--no-cuda",
    ])
    assert args.name == "my_experiment"
    assert args.n_modules == 20
    assert args.latent_dim == 32
    assert args.epochs == 100
    assert args.warmup_epochs == 20


def test_parse_args_custom_modules(fake_dataset_csv, tmp_path):
    args = parse_args([
        "exp",
        "--dataset", fake_dataset_csv,
        "--outdir", str(tmp_path),
        "--n-modules", "8",
        "--latent-dim", "16",
        "--no-cuda",
    ])
    assert args.n_modules == 8
    assert args.latent_dim == 16


def test_main_trains_and_saves(fake_dataset_csv, tmp_path):
    """End-to-end: train for minimal epochs and verify model is saved."""
    args = parse_args([
        "test_run",
        "--dataset", fake_dataset_csv,
        "--outdir", str(tmp_path),
        "--epochs", "3",
        "--warmup-epochs", "2",
        "--transition-epochs", "1",
        "--n-modules", "3",
        "--latent-dim", "4",
        "--hidden-dims", "[8]",
        "--batch-size", "4",
        "--no-cuda",
        "--no-eval",
        "--no-progress-bar",
    ])
    main(args)

    import os
    exp_dir = str(tmp_path / "test_run")
    assert os.path.exists(os.path.join(exp_dir, "model.pt"))
    assert os.path.exists(os.path.join(exp_dir, "specs.json"))

    from bsvae.utils.modelIO import load_metadata
    meta = load_metadata(exp_dir)
    assert meta["n_modules"] == 3


def _isoform_base_args(dataset, outdir, name="run"):
    return [
        name, "--dataset", dataset, "--outdir", outdir,
        "--epochs", "4", "--warmup-epochs", "2", "--transition-epochs", "1",
        "--n-modules", "3", "--latent-dim", "4", "--hidden-dims", "[8]",
        "--batch-size", "8", "--no-cuda", "--no-eval", "--no-progress-bar",
    ]


def test_main_hier_loss(fake_isoform_dataset, fake_tx2gene, tmp_path):
    """main() with --hier-strength + --tx2gene trains without error."""
    import os
    args = parse_args(_isoform_base_args(fake_isoform_dataset, str(tmp_path), "hier_run") + [
        "--hier-strength", "0.5", "--tx2gene", fake_tx2gene,
    ])
    main(args)
    assert os.path.exists(os.path.join(str(tmp_path), "hier_run", "model.pt"))


def test_main_isoform_stratified(fake_isoform_dataset, fake_tx2gene, tmp_path):
    """main() with --isoform-stratified replaces DataLoader and completes training."""
    import os
    args = parse_args(_isoform_base_args(fake_isoform_dataset, str(tmp_path), "strat_run") + [
        "--hier-strength", "0.5", "--tx2gene", fake_tx2gene,
        "--isoform-stratified", "--p-multi", "0.8",
    ])
    main(args)
    assert os.path.exists(os.path.join(str(tmp_path), "strat_run", "model.pt"))


def test_main_bad_gene_groups_index_raises(fake_isoform_dataset, fake_tx2gene, tmp_path, monkeypatch):
    """gene_groups with an out-of-bounds index raises ValueError before training starts."""
    import bsvae.utils.hierarchy as hier_mod

    original = hier_mod.group_isoforms_by_gene

    def corrupt_groups(tx2gene, feature_ids):
        result = original(tx2gene, feature_ids)
        if result:
            first = next(iter(result))
            result[first] = result[first] + [9999]
        return result

    monkeypatch.setattr(hier_mod, "group_isoforms_by_gene", corrupt_groups)

    args = parse_args(_isoform_base_args(fake_isoform_dataset, str(tmp_path), "bad_run") + [
        "--hier-strength", "0.5", "--tx2gene", fake_tx2gene,
    ])
    with pytest.raises(ValueError, match="out-of-bounds"):
        main(args)
