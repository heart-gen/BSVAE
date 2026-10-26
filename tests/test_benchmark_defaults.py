"""Smoke test for scripts/benchmark_defaults.py: one tiny cell, every setting, labels reused on rerun."""
import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_defaults.py"


def _run(outdir: Path, extra):
    argv = ["benchmark_defaults.py", "--outdir", str(outdir), "--n-genes", "120", "--n-samples", "48", "--n-modules", "4",
            "--epochs", "6", "--seeds", "0", "--switch-rates", "0.15", "--loo", "approx", "--target-corrs", "none,0.4",
            "--no-cuda", "--log-level", "warning", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        return runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old


@pytest.mark.filterwarnings("ignore")
def test_defaults_sweep_writes_every_setting(tmp_path):
    _run(tmp_path, [])
    df = pd.read_csv(tmp_path / "results.tsv", sep="\t")
    assert set(df["check"]) == {"type1", "power"}
    assert set(df["design"]) == {"full", "crossfit"}
    # 2 covariates x 2 target corrs x (1 loo for full + 2 agreement for crossfit) = 12 settings
    settings = df[["design", "covariates", "max_target_corr", "loo", "module_agreement"]].drop_duplicates()
    assert len(settings) == 12
    null = df[df["check"] == "type1"]
    assert {"all", "label_concordant"} <= set(null["stratum"])
    assert null["type1_0.05"].between(0, 1).all()
    power = df[df["check"] == "power"]
    assert len(power) == 12 and (power["switch_rate"] == 0.15).all()
    assert power["n_switched"].gt(0).all()
    assert (tmp_path / "seed_0" / "labels_null_full.npy").exists()
    assert (tmp_path / "seed_0" / "labels_switch0.15_xf1.npy").exists()
    # cell parameters and the null travel with every row
    assert (df["n_genes"] == 120).all() and (df["factor_corr"] == 0.3).all()
    assert (df["null"] == "analytic").all()
    # reuse saved labels: identical results without the model directories
    (tmp_path / "results.tsv").unlink()
    _run(tmp_path, ["--reuse-labels"])
    df2 = pd.read_csv(tmp_path / "results.tsv", sep="\t")
    pd.testing.assert_frame_equal(df.sort_index(axis=1), df2.sort_index(axis=1))
    # the stratum-aware null runs on the same saved labels and labels its rows
    _run(tmp_path, ["--reuse-labels", "--null", "stratum"])
    df3 = pd.read_csv(tmp_path / "results.tsv", sep="\t")
    assert (df3["null"] == "stratum").all() and len(df3) == len(df)
    assert df3.loc[df3["check"] == "type1", "type1_0.05"].between(0, 1).all()
