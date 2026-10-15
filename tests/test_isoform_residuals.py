import numpy as np
import pandas as pd
import pytest

from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import gene_aggregate, isoform_residuals, map_transcripts_to_genes


def test_residuals_remove_gene_programme_and_exclude_singletons():
    rng = np.random.default_rng(0)
    S = 60
    g = rng.normal(size=S)
    iso_a = 2 * g + 0.1 * rng.normal(size=S) + 5
    iso_b = 1.5 * g + 0.1 * rng.normal(size=S) + 3
    single = rng.normal(size=S)
    expr = pd.DataFrame(np.vstack([iso_a, iso_b, single]), index=["t1.2", "t2", "t3"],
                        columns=[f"s{j}" for j in range(S)])
    tx2gene = pd.Series({"t1": "gA", "t2": "gA", "t3": "gB"})
    resid, stats = isoform_residuals(expr, tx2gene)
    assert list(resid.index) == ["t1.2", "t2"]
    agg = gene_aggregate(expr, map_transcripts_to_genes(expr.index, tx2gene))
    for t in resid.index:
        assert abs(np.corrcoef(resid.loc[t], agg.loc["gA"])[0, 1]) < 0.05
    assert stats.loc["t1.2", "r2_gene"] > 0.95 and stats.loc["t1.2", "n_isoforms"] == 2


def test_residuals_isolate_switched_isoform_signal():
    """On the mixing simulation, the switched isoform's residual tracks its new module."""
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=80, n_modules=4, switch_rate=0.5,
             alpha=0.7, signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=2)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    resid, stats = isoform_residuals(expr, t2g)
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    z = r.module_latents.values
    gaps_sw, gaps_plain = [], []
    for tx in resid.index:
        row = st.loc[tx]
        x = resid.loc[tx].values
        if x.std() == 0:
            continue
        c_new = abs(np.corrcoef(x, z[:, int(row["tx_module"])])[0, 1])
        (gaps_sw if row["is_switched"] == 1 else gaps_plain).append(c_new)
    assert np.mean(gaps_sw) > 0.35
    assert np.mean(gaps_sw) > np.mean(gaps_plain) + 0.15
