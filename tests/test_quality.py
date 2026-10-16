import numpy as np
import pandas as pd

from bsvae.networks.quality import module_enrichment, module_preservation, module_quality


def _blocks(seed, n_samples=60, sizes=(30, 30), noise=0.4):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for k, sz in enumerate(sizes):
        z = rng.normal(size=n_samples)
        for _ in range(sz):
            rows.append(z + noise * rng.normal(size=n_samples)); labels.append(k)
    fids = [f"f{i}" for i in range(len(rows))]
    return pd.DataFrame(np.array(rows), index=fids, columns=[f"s{j}" for j in range(n_samples)]), np.array(labels), fids


def test_module_quality_table():
    expr, labels, fids = _blocks(0)
    labels = labels.copy(); labels[:3] = -1
    q = module_quality(expr, labels, fids)
    assert list(q["module"]) == [0, 1] and q["size"].tolist() == [27, 30]
    assert (q["mean_within_cor"] > 0.6).all() and (q["kme_mean"] > 0.8).all()
    assert (q["n_grey_total"] == 3).all()


def test_module_preservation_detects_preserved_and_lost_modules():
    ref, labels, fids = _blocks(0)
    test = ref.copy()
    rng = np.random.default_rng(1)
    test.iloc[30:] = rng.normal(size=test.iloc[30:].shape)      # module 1 destroyed in the test set
    pres = module_preservation(ref, test, labels, fids, n_perm=100)
    z = pres.set_index("module")["Z_density"]
    assert z[0] > 5 and z[1] < 2
    assert pres.set_index("module")["cor_kME"][0] > 0.5


def test_module_enrichment_fdr():
    labels = np.array([0] * 20 + [1] * 20 + [-1] * 5)
    fids = [f"g{i}" for i in range(45)]
    sets = {"setA": fids[:15], "setB": fids[40:45] + fids[20:22]}
    df = module_enrichment(labels, fids, sets, min_overlap=3)
    top = df.iloc[0]
    assert top["module"] == 0 and top["annotation"] == "setA" and top["q_value"] < 0.01
    assert "q_value" in df.columns and (df["q_value"] <= 1).all()
