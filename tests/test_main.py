"""Tests for bsvae-train CLI (cli/train.py)."""

import tempfile
import numpy as np
import pandas as pd
import pytest
import torch

from bsvae.cli.train import parse_args, main


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
