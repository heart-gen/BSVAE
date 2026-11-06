"""Smoke test for scripts/benchmark_rare_isoforms.py: production call rule by default, per-isoform tables, settings per row."""
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_rare_isoforms.py"


def _run(outdir: Path, extra):
    argv = ["benchmark_rare_isoforms.py", "--outdir", str(outdir), "--n-genes", "120", "--n-samples", "48",
            "--n-modules", "4", "--epochs", "6", "--seeds", "0", "--scenarios", "hard_rare",
            "--variants", "base,filter,pearson", "--no-cuda", "--log-level", "warning", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        return runpy.run_path(str(SCRIPT), run_name="__main__")
    finally:
        sys.argv = old


@pytest.mark.filterwarnings("ignore")
def test_rare_isoforms_default_call_rule_and_outputs(tmp_path):
    _run(tmp_path, [])
    df = pd.read_csv(tmp_path / "results.tsv", sep="\t")
    assert set(df["stratum"]) == {"all", "low", "mid", "high", "unswitched_minor"}
    assert set(df["variant"]) == {"base", "filter", "pearson"}
    # settings travel with every row; the default call has no effect-size floor and adjusts for library size
    assert (df["min_kme"] == 0.0).all() and (df["alpha"] == 0.05).all()
    assert (df["covariates"] == "libsize").all() and df["max_target_corr"].isna().all()
    assert (df["n_samples"] == 48).all() and (df["n_genes"] == 120).all()
    assert df["called"].between(0, 1).all()
    assert (df["called_high_kme"] <= df["called"] + 1e-12).all()
    ceiling = pd.read_csv(tmp_path / "oracle_ceiling.tsv", sep="\t")
    assert set(ceiling["stratum"]) == {"low", "mid", "high"} and (ceiling["n_samples"] == 48).all()
    for variant in ["base", "filter", "pearson"]:
        tab = pd.read_csv(tmp_path / "hard_rare" / "seed_0" / variant / "per_isoform.tsv.gz", sep="\t", index_col=0)
        assert {"is_switched", "minor", "tercile", "kme_residual", "p_value", "q_value", "called"} <= set(tab.columns)
        assert set(tab.loc[tab["is_switched"], "tercile"]) == {"low", "mid", "high"}
        assert tab.loc[~tab["is_switched"], "tercile"].isna().all()
        tested = tab["q_value"].notna()
        # the call is FDR only: called == q < alpha on every tested transcript, nothing untested is called
        assert (tab.loc[tested, "called"] == (tab.loc[tested, "q_value"] < 0.05)).all()
        assert not tab.loc[~tested, "called"].any()


@pytest.mark.filterwarnings("ignore")
def test_rare_isoforms_floor_is_a_subset(tmp_path):
    _run(tmp_path / "nofloor", ["--variants", "base"])
    _run(tmp_path / "floor", ["--variants", "base", "--min-kme", "0.3", "--covariates", "none"])
    a = pd.read_csv(tmp_path / "nofloor" / "hard_rare" / "seed_0" / "base" / "per_isoform.tsv.gz", sep="\t", index_col=0)
    b = pd.read_csv(tmp_path / "floor" / "hard_rare" / "seed_0" / "base" / "per_isoform.tsv.gz", sep="\t", index_col=0)
    df = pd.read_csv(tmp_path / "floor" / "results.tsv", sep="\t")
    assert (df["min_kme"] == 0.3).all() and (df["covariates"] == "none").all()
    floored = b["called"] & b["kme_residual"].abs().ge(0.3)
    assert (b["called"] == floored).all()
    assert np.isfinite(a["p_value"].dropna()).all()
