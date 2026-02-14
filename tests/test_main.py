import sys
import types
import tempfile
import pandas as pd
import numpy as np
import torch
import pytest

import bsvae.main as main


# -------------------------------
# Fixtures
# -------------------------------
@pytest.fixture
def fake_dataset_csv(tmp_path):
    """Create a toy expression matrix CSV (genes x samples)."""
    df = pd.DataFrame(
        np.random.randn(6, 3),
        index=[f"gene{i}" for i in range(6)],
        columns=[f"s{i}" for i in range(3)],
    )
    csv_path = tmp_path / "expr.csv"
    df.to_csv(csv_path)
    return str(csv_path)


# -------------------------------
# Unit tests
# -------------------------------
def test_ast_literal_eval():
    assert main.ast_literal_eval("[1,2]") == [1, 2]
    assert main.ast_literal_eval("foo") == "foo"


def test_load_config_and_parse_arguments(tmp_path, fake_dataset_csv):
    ini = tmp_path / "hyper.ini"
    with open(ini, "w") as f:
        f.write("[Custom]\nseed=123\nno_cuda=True\n")

    args = main.parse_arguments([
        "exp1",
        "--config", str(ini),
        "--gene-expression-filename", fake_dataset_csv,
    ])
    assert args.name == "exp1"
    assert args.seed == 123
    assert args.no_cuda is True


def test_parse_arguments_invalid_dataset(tmp_path):
    ini = tmp_path / "h.ini"
    with open(ini, "w") as f:
        f.write("[Custom]\nseed=1\nno_cuda=False\n")

    with pytest.raises(SystemExit):
        main.parse_arguments([
            "exp1",
            "--config", str(ini),
            "--gene-expression-filename", "a.csv",
            "--gene-expression-dir", "b/",
        ])


def test_main_training_and_eval(monkeypatch, tmp_path, fake_dataset_csv):
    """Run main() with mocks for Trainer/Evaluator to avoid heavy compute."""

    # Patch dataloaders to return a trivial dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.data = torch.randn(4, 3)
            self.genes = [f"g{i}" for i in range(3)]
        def __getitem__(self, idx):
            return self.data[idx], f"g{idx}"
        def __len__(self):
            return len(self.data)

    def fake_get_dataloaders(*args, **kwargs):
        return torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

    monkeypatch.setattr(main, "get_dataloaders", fake_get_dataloaders)

    # Patch Trainer/Evaluator to just log calls
    class DummyTrainer:
        def __init__(self, *a, **k): self.model = torch.nn.Linear(3, 2)
        def __call__(self, *a, **k): return None

    class DummyEvaluator:
        def __init__(self, *a, **k): pass
        def __call__(self, *a, **k): return {"loss": 0.0}

    monkeypatch.setattr(main, "Trainer", DummyTrainer)
    monkeypatch.setattr(main, "Evaluator", DummyEvaluator)

    # Patch save/load to avoid filesystem I/O
    monkeypatch.setattr(main, "save_model", lambda *a, **k: None)
    monkeypatch.setattr(main, "load_model", lambda *a, **k: torch.nn.Linear(3, 2))
    monkeypatch.setattr(main, "load_metadata", lambda *a, **k: {"dummy": True})
    monkeypatch.setattr(main, "load_ppi_laplacian", lambda *a, **k: (None, None))

    # Prepare args
    args = types.SimpleNamespace(
        name="exp1",
        seed=123,
        no_cuda=True,
        is_eval_only=False,
        dataset="genenet",
        batch_size=2,
        gene_expression_filename=fake_dataset_csv,
        gene_expression_dir=None,
        latent_dim=2,
        hidden_dims=[4],
        dropout=0.1,
        init_sd=0.02,
        learn_var=False,
        lr=1e-3,
        beta=1.0,
        l1_strength=1e-3,
        lap_strength=1e-4,
        coexpr_strength=0.1,
        epochs=1,
        checkpoint_every=1,
        ppi_taxid="9606",
        ppi_cache=str(tmp_path),
        no_test=False,
        eval_batchsize=2,
    )

    main.main(args)  # should run without error


def test_cli_entrypoint(monkeypatch, tmp_path, fake_dataset_csv):
    """Test CLI-level call with sys.argv patched."""

    # Reuse mocks from above
    monkeypatch.setattr(main, "get_dataloaders",
                        lambda *a, **k: torch.utils.data.DataLoader(
                            [(torch.randn(3), "g1")], batch_size=1))
    monkeypatch.setattr(main, "Trainer",
                        lambda *a, **k: type("T", (), {"model": torch.nn.Linear(3, 2), "__call__": lambda s,*a,**k: None})())
    monkeypatch.setattr(main, "Evaluator",
                        lambda *a, **k: type("E", (), {"__call__": lambda s,*a,**k: {"loss": 0.0}})())
    monkeypatch.setattr(main, "save_model", lambda *a, **k: None)
    monkeypatch.setattr(main, "load_model", lambda *a, **k: torch.nn.Linear(3, 2))
    monkeypatch.setattr(main, "load_metadata", lambda *a, **k: {"dummy": True})
    monkeypatch.setattr(main, "load_ppi_laplacian", lambda *a, **k: (None, None))

    # Write a minimal config file
    ini = tmp_path / "hyper.ini"
    with open(ini, "w") as f:
        f.write("[Custom]\nseed=42\nno_cuda=True\n")

    sys_argv_backup = sys.argv[:]
    sys.argv = [
        "main.py",
        "exp_cli",
        "--config", str(ini),
        "--gene-expression-filename", fake_dataset_csv,
    ]
    try:
        main.cli()  # Should run end-to-end
    finally:
        sys.argv = sys_argv_backup
