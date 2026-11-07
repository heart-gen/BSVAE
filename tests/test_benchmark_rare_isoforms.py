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
    # the confidence tier is a share of calls, undefined (NaN) where a stratum has no calls
    assert (df["n_called"] == (df["called"] * df["n"]).round()).all()
    assert (df["n_called_high_kme"] <= df["n_called"]).all()
    has_calls = df["n_called"] > 0
    assert df.loc[has_calls, "frac_calls_high_kme"].between(0, 1).all()
    assert df.loc[~has_calls, "frac_calls_high_kme"].isna().all()
    assert "called_high_kme" not in df
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
        # the per-stratum summary agrees with the per-isoform table it was computed from
        row = df[(df["variant"] == variant) & (df["stratum"] == "all")].iloc[0]
        calls = tab[tab["is_switched"] & tab["called"]]
        assert row["n_called"] == len(calls)
        assert row["n_called_high_kme"] == int((calls["kme_residual"].abs() >= 0.3).sum())


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


def test_call_confidence_is_a_share_of_calls():
    call_confidence = runpy.run_path(str(SCRIPT))["call_confidence"]   # not run as __main__: no benchmark
    # 100 isoforms, 10 called, every call above the tier: the share of calls is 1.0, not 10 / 100
    sub = pd.DataFrame({"called": [True] * 10 + [False] * 90, "kme_residual": [0.5] * 10 + [0.9] * 90})
    out = call_confidence(sub)
    assert out == {"n_called": 10, "n_called_high_kme": 10, "frac_calls_high_kme": 1.0}
    # uncalled isoforms never count, whatever their kME; negative kME counts by magnitude
    sub = pd.DataFrame({"called": [True, True, True, True, False], "kme_residual": [0.31, -0.45, 0.1, -0.2, 0.99]})
    assert call_confidence(sub) == {"n_called": 4, "n_called_high_kme": 2, "frac_calls_high_kme": 0.5}
    # no calls: the share is undefined (NaN), not 0
    out = call_confidence(pd.DataFrame({"called": [False] * 5, "kme_residual": [0.8] * 5}))
    assert out["n_called"] == 0 and out["n_called_high_kme"] == 0 and np.isnan(out["frac_calls_high_kme"])
