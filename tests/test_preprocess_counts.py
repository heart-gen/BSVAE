import numpy as np
import pandas as pd
import pytest

from bsvae.utils.preprocess import pearson_residuals, size_factors


def _counts(seed=0, F=200, S=30):
    rng = np.random.default_rng(seed)
    lib = np.exp(rng.normal(0, 0.3, size=S))
    base = np.exp(rng.normal(3, 1.5, size=F))
    mu = base[:, None] * lib[None, :]
    x = rng.negative_binomial(20, 20 / (20 + mu))
    return pd.DataFrame(x.astype(float), index=[f"f{i}" for i in range(F)], columns=[f"s{j}" for j in range(S)]), lib


def test_size_factors_track_library_size():
    counts, lib = _counts()
    sf = size_factors(counts)
    assert abs(np.exp(np.mean(np.log(sf))) - 1.0) < 1e-9
    assert np.corrcoef(np.log(sf), np.log(lib))[0, 1] > 0.95


def test_pearson_residuals_are_variance_stabilised():
    counts, _ = _counts()
    r = pearson_residuals(counts, theta=20.0)
    assert r.shape == counts.shape and list(r.index) == list(counts.index)
    sd = r.std(axis=1)
    lo, hi = sd[counts.mean(axis=1) < 10], sd[counts.mean(axis=1) > 200]
    # under the generating NB model the residual sd is ~1 regardless of abundance
    assert 0.6 < lo.mean() < 1.4 and 0.6 < hi.mean() < 1.4
    assert np.abs(r.values).max() <= np.sqrt(counts.shape[1]) + 1e-9


def test_pearson_residuals_moments_and_errors():
    counts, _ = _counts(1)
    r = pearson_residuals(counts, theta="moments", clip=3.0)
    assert np.abs(r.values).max() <= 3.0
    with pytest.raises(ValueError):
        pearson_residuals(counts, theta="bogus")
    with pytest.raises(ValueError):
        size_factors(counts, method="nope")


def test_generate_scenario_writes_counts(tmp_path):
    from bsvae.simulation.scenario import DEFAULT_CONFIG, generate_scenario, expand_scenarios
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["generator"].update(n_features=60, n_samples=12, n_modules=2)
    cfg["grid"] = {}; cfg["stress"] = []
    sid = expand_scenarios(cfg)[0][0]
    generate_scenario(cfg, sid, rep=0, outdir=tmp_path)
    p = tmp_path / "scenarios" / sid / "rep_000" / "expr" / "counts_features_x_samples.tsv.gz"
    assert p.exists()
    c = pd.read_csv(p, sep="\t", index_col=0)
    assert c.shape == (60, 12) and (c.values >= 0).all() and np.allclose(c.values, np.round(c.values))
