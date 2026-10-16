import numpy as np
import pandas as pd
import pytest

from bsvae.networks.targets import (
    loci_to_genes, locus_summary, read_bed, read_gene_annotation, read_id_list,
    resolve_target_transcripts, target_gene_tests, profile_reliability,
)
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params


# ---------------------------------------------------------------------------
# readers and locus mapping
# ---------------------------------------------------------------------------

def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_read_id_list_strips_versions_headers_and_duplicates(tmp_path):
    p = _write(tmp_path, "genes.txt", "gene_id\n# comment\nENSG00000000001.4\nENSG00000000002\tfoo\nENSG00000000001\n\n")
    assert read_id_list(p) == ["ENSG00000000001", "ENSG00000000002"]


def test_read_bed_names_and_defaults(tmp_path):
    p = _write(tmp_path, "loci.bed", "track name=x\nchr11\t113400000\t113400001\trs1\nchr11\t130000000\t130500000\n")
    bed = read_bed(p)
    assert list(bed["locus"]) == ["rs1", "chr11:130000000-130500000"]
    assert bed.loc[0, "end"] - bed.loc[0, "start"] == 1
    with pytest.raises(ValueError):
        read_bed(_write(tmp_path, "bad.bed", "chr1\t5\n"))


def test_read_gene_annotation_gtf_and_bed(tmp_path):
    gtf = _write(tmp_path, "a.gtf", "\n".join([
        "#!genome-build x",
        'chr11\tHAVANA\tgene\t113409605\t113475691\t.\t-\t.\tgene_id "ENSG00000149295.14"; gene_name "DRD2";',
        'chr11\tHAVANA\ttranscript\t113409605\t113475691\t.\t-\t.\tgene_id "ENSG00000149295.14"; transcript_id "ENST1";',
        'chr11\tHAVANA\tgene\t130742842\t130857469\t.\t+\t.\tgene_id "ENSG00000120451.11"; gene_name "SNX19";',
    ]) + "\n")
    ann = read_gene_annotation(gtf)
    assert list(ann["gene_id"]) == ["ENSG00000149295", "ENSG00000120451"]
    assert ann.loc[0, "tss"] == 113475691 and ann.loc[1, "tss"] == 130742841   # minus strand uses end
    assert ann.loc[0, "chrom"] == "11"
    bed = _write(tmp_path, "a.bed", "chrom\tstart\tend\tgene_id\n11\t100\t200\tG1\tSYM1\t+\n11\t500\t600\tG2\n")
    ann2 = read_gene_annotation(bed)
    assert list(ann2["gene_name"]) == ["SYM1", ""] and list(ann2["tss"]) == [100, 500]


def _toy_annotation():
    return pd.DataFrame({
        "gene_id": ["G1", "G2", "G3", "G4"], "gene_name": ["A", "B", "C", "D"],
        "chrom": ["1", "1", "1", "2"], "start": [100, 1000, 5000, 100], "end": [300, 1500, 5200, 300],
        "strand": ["+", "-", "+", "+"],
    }).assign(tss=lambda d: np.where(d["strand"] == "-", d["end"], d["start"]))


def test_loci_to_genes_rules():
    ann = _toy_annotation()
    loci = pd.DataFrame({"locus": ["L"], "chrom": ["chr1"], "start": [1200], "end": [1201]})
    assert list(loci_to_genes(loci, ann, rule="overlap")["gene_id"]) == ["G2"]
    win = loci_to_genes(loci, ann, rule="window", window=1000)
    assert list(win["gene_id"]) == ["G2", "G1"]          # sorted by TSS distance
    assert win.loc[win["gene_id"] == "G2", "distance"].item() == 1500 - 1201 + 1
    close = loci_to_genes(loci, ann, rule="closest", n_closest=3)
    assert list(close["gene_id"]) == ["G2", "G1", "G3"]  # G4 is on another chromosome
    with pytest.raises(ValueError):
        loci_to_genes(loci, ann, rule="nearest")


def test_resolve_targets_by_id_symbol_and_transcript():
    tx_gene = pd.Series({"t1.1": "G1", "t2": "G1", "t3": "G2", "t4": "G3"})
    ann = _toy_annotation()
    sel, unmatched = resolve_target_transcripts(tx_gene, genes=["G1", "b", "ZZZ"], transcripts=["t4.2", "t9"], annotation=ann)
    assert set(sel.index) == {"t1", "t2", "t3", "t4"}
    assert unmatched == ["ZZZ", "t9"]


# ---------------------------------------------------------------------------
# tests on simulated data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iso_sim():
    p = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=300, n_samples=100, n_modules=4, switch_rate=0.3,
             alpha=1.0, signal_scale=3.0, null_fraction=0.0)
    r = simulate_params(p, seed=4)
    expr = r.expr_features_x_samples
    t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"], r.extra_truth["tx2gene.tsv"]["gene_id"]))
    truth = r.modules_hard.set_index("feature_id")["module"].reindex(expr.index).values
    st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
    return expr, t2g, truth, st, r.counts_features_x_samples, r.covariates


@pytest.mark.parametrize("null", ["permutation", "matched", "analytic"])
def test_target_tests_separate_switched_from_unswitched(iso_sim, null):
    expr, t2g, truth, st, counts, cov = iso_sim
    tx_gene = pd.Series(t2g).reindex(expr.index.astype(str))
    sw_genes = st.loc[st["is_switched"] == 1, "gene_id"].unique()[:15]
    multi = st[(st["isoform_rank"] > 0)].groupby("gene_id").size()
    ns_genes = [g for g in multi.index if g not in set(st.loc[st["is_switched"] == 1, "gene_id"])][:15]
    targets, _ = resolve_target_transcripts(tx_gene, genes=list(sw_genes) + ns_genes)
    res = target_gene_tests(expr, t2g, truth, list(expr.index), targets, null=null, n_perm=200, seed=1)
    iso = res["isoforms"]
    assert len(iso) == len(targets) and {"p_value", "q_value", "called", "best_other_module"} <= set(iso.columns)
    is_sw = st.loc[iso.index, "is_switched"].values == 1
    assert iso.loc[is_sw, "q_value"].median() < iso.loc[~is_sw, "q_value"].median()
    recall = iso.loc[is_sw, "called"].mean()
    fp = iso.loc[~is_sw, "called"].mean()
    assert recall > 0.5 and fp < 0.25
    hit = iso[iso["called"] & is_sw]
    assert (hit["best_other_module"].values == st.loc[hit.index, "tx_module"].values).mean() > 0.8
    genes = res["genes"]
    assert set(genes.index) == set(targets.values) and genes["n_isoforms_tested"].sum() == len(iso)
    assert genes.loc[list(sw_genes), "divergent"].mean() > genes.loc[ns_genes, "divergent"].mean()


def test_target_tests_neighbours_reliability_and_condition(iso_sim):
    expr, t2g, truth, st, counts, cov = iso_sim
    tx_gene = pd.Series(t2g).reindex(expr.index.astype(str))
    sw_genes = st.loc[st["is_switched"] == 1, "gene_id"].unique()[:10]
    targets, _ = resolve_target_transcripts(tx_gene, genes=list(sw_genes))
    cond = pd.Series(np.where(np.arange(expr.shape[1]) % 2 == 0, "case", "control"), index=expr.columns)
    res = target_gene_tests(expr, t2g, truth, list(expr.index), targets, null="analytic",
                            counts=counts, condition=cond, n_neighbours=5)
    iso = res["isoforms"]
    assert "reliability" in iso and iso["reliability"].between(0, 1).all()
    ok = iso["kme_disattenuated"].notna()
    assert (iso.loc[ok, "kme_disattenuated"].abs() >= iso.loc[ok, "kme_residual"].abs() - 1e-9).all()
    assert {"kme_case", "kme_control", "z_interaction", "p_interaction"} <= set(iso.columns)
    # no real case/control difference was simulated: interaction p-values are not concentrated near zero
    assert (iso["p_interaction"] < 0.05).mean() < 0.3
    nb = res["neighbours"]
    assert set(nb["feature_id"]) == set(iso.index[iso["called"]])
    assert nb.groupby("feature_id").size().max() <= 5
    # neighbours live in the called isoform's target module
    mod_of = dict(zip(expr.index.astype(str), truth))
    assert all(mod_of[n] == m for n, m in zip(nb["neighbour"], nb["module"]))


def test_single_isoform_targets_and_empty(iso_sim):
    expr, t2g, truth, st, counts, cov = iso_sim
    tx_gene = pd.Series(t2g).reindex(expr.index.astype(str))
    single = st[st["gene_id"].map(st.groupby("gene_id").size()) == 1].index[:5]
    targets, _ = resolve_target_transcripts(tx_gene, transcripts=list(single))
    res = target_gene_tests(expr, t2g, truth, list(expr.index), targets, null="analytic")
    assert res["isoforms"]["single_isoform"].all() and res["isoforms"]["r2_gene"].isna().all()
    with pytest.raises(ValueError):
        target_gene_tests(expr, t2g, truth, list(expr.index), targets, null="bootstrap")
    empty = target_gene_tests(expr, t2g, truth, list(expr.index), pd.Series(dtype=object))
    assert empty["isoforms"].empty


def test_locus_summary_counts_and_concentration(iso_sim):
    expr, t2g, truth, st, counts, cov = iso_sim
    tx_gene = pd.Series(t2g).reindex(expr.index.astype(str))
    genes = list(st["gene_id"].unique()[:6])
    targets, _ = resolve_target_transcripts(tx_gene, genes=genes)
    res = target_gene_tests(expr, t2g, truth, list(expr.index), targets, null="analytic")
    loci_genes = pd.DataFrame({"locus": ["L1"] * 3 + ["L2"] * 3, "gene_id": genes})
    summ = locus_summary(res["isoforms"], loci_genes, truth)
    assert list(summ["locus"]) == ["L1", "L2"] and (summ["n_genes"] == 3).all()
    assert summ["n_isoforms_tested"].sum() == len(res["isoforms"])
    assert summ["p_concentration"].between(0, 1).all()


def test_profile_reliability_rises_with_abundance(iso_sim):
    expr, t2g, truth, st, counts, cov = iso_sim
    rel = profile_reliability(counts, expr, list(expr.index))
    me = expr.mean(axis=1)
    assert rel.between(0, 1).all()
    assert rel[me > me.quantile(0.8)].mean() > rel[me < me.quantile(0.2)].mean()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_target_genes_with_bed(iso_sim, tmp_path):
    from bsvae.cli.networks import cli
    expr, t2g, truth, st, counts, cov = iso_sim
    (tmp_path / "expr.tsv").write_text(expr.to_csv(sep="\t"))
    (tmp_path / "counts.tsv").write_text(counts.to_csv(sep="\t"))
    pd.DataFrame({"transcript_id": list(t2g), "gene_id": list(t2g.values())}).to_csv(tmp_path / "tx2gene.tsv", sep="\t", index=False)
    pd.DataFrame({"feature_id": expr.index, "module": truth}).to_csv(tmp_path / "modules.csv", index=False)
    genes = list(st["gene_id"].unique()[:4])
    # two loci on "chr1": genes laid out 10 kb apart
    ann = pd.DataFrame({"chrom": "chr1", "start": [10_000 * (i + 1) for i in range(len(genes))],
                        "end": [10_000 * (i + 1) + 2000 for i in range(len(genes))], "gene_id": genes,
                        "gene_name": [f"SYM{i}" for i in range(len(genes))], "strand": "+"})
    ann.to_csv(tmp_path / "genes.bed", sep="\t", index=False, header=False)
    (tmp_path / "loci.bed").write_text("chr1\t10500\t10501\trsA\nchr1\t30500\t30501\trsB\n")
    (tmp_path / "list.txt").write_text("SYM3\n")
    cov_df = pd.DataFrame({"dx": np.where(np.arange(expr.shape[1]) % 2 == 0, "case", "control")}, index=expr.columns)
    cov_df.to_csv(tmp_path / "cov.tsv", sep="\t")
    out = tmp_path / "out"
    cli([
        "target-genes", "--dataset", str(tmp_path / "expr.tsv"), "--tx2gene", str(tmp_path / "tx2gene.tsv"),
        "--modules", str(tmp_path / "modules.csv"), "--output-dir", str(out),
        "--bed", str(tmp_path / "loci.bed"), "--annotation", str(tmp_path / "genes.bed"), "--rule", "window",
        "--window", "1000", "--genes", str(tmp_path / "list.txt"), "--null", "analytic",
        "--counts", str(tmp_path / "counts.tsv"), "--covariates", str(tmp_path / "cov.tsv"), "--condition", "dx",
    ])
    lg = pd.read_csv(out / "loci_genes.tsv", sep="\t")
    assert set(lg["gene_id"]) == {genes[0], genes[2]}
    iso = pd.read_csv(out / "target_isoforms.tsv", sep="\t", index_col=0)
    assert set(iso["gene_id"]) == {genes[0], genes[2], genes[3]}
    assert {"kme_disattenuated", "p_interaction"} <= set(iso.columns)
    loci = pd.read_csv(out / "target_loci.tsv", sep="\t")
    assert list(loci["locus"]) == ["rsA", "rsB"]
    assert (out / "target_genes.tsv").exists() and (out / "target_neighbours.tsv").exists()


def test_combine_target_tests_pools_weak_evidence(tmp_path):
    from bsvae.networks.targets import combine_target_tests
    from bsvae.cli.networks import cli
    rng = np.random.default_rng(0)
    iso = [f"t{i}" for i in range(30)]
    tables = {}
    for k in range(6):
        p = rng.uniform(0.05, 1.0, size=30)          # nothing significant alone
        p[:5] = rng.uniform(0.02, 0.08, size=5)      # weak but consistent for five isoforms
        tables[f"r{k}"] = pd.DataFrame({"gene_id": [f"g{i // 2}" for i in range(30)], "best_other_module": rng.integers(0, 4, 30),
                                        "kme_residual": np.where(np.arange(30) < 5, 0.25, 0.05) * rng.choice([-1, 1], 30),
                                        "p_value": p, "q_value": p, "called": False}, index=iso)
    out = combine_target_tests(tables, n_samples={f"r{k}": 150 for k in range(6)})
    assert out.loc[iso[:5], "called"].all() and not out.loc[iso[5:], "called"].any()
    assert (out["n_cohorts"] == 6).all() and out.loc["t0", "modules"].count(";") == 5
    assert out["p_combined"].between(0, 1).all() and (out["q_combined"] >= out["p_combined"] - 1e-12).all()
    # fewer cohorts than min_cohorts are never called
    one = combine_target_tests({"a": tables["r0"]}, min_cohorts=2)
    assert not one["called"].any()
    with pytest.raises(ValueError):
        combine_target_tests({})
    # CLI round trip
    paths = []
    for k, t in tables.items():
        d = tmp_path / k; d.mkdir(); t.to_csv(d / "target_isoforms.tsv", sep="\t"); paths.append(f"{k}={d / 'target_isoforms.tsv'}")
    cli(["target-genes-combine", "--inputs", *paths, "--n-samples", *[f"r{k}=150" for k in range(6)], "--output", str(tmp_path / "combined.tsv")])
    got = pd.read_csv(tmp_path / "combined.tsv", sep="\t", index_col=0)
    assert got["called"].sum() == 5


def test_combine_strube_correction_reduces_evidence_for_dependent_cohorts():
    from bsvae.networks.targets import combine_target_tests
    rng = np.random.default_rng(3)
    iso = [f"t{i}" for i in range(200)]
    shared = rng.normal(size=200)                       # donor-driven component shared by all cohorts
    tables = {}
    for k in range(5):
        z = 0.9 * shared + 0.45 * rng.normal(size=200)  # strongly dependent cohorts
        from scipy import stats as sps
        p = sps.norm.sf(z)
        tables[f"r{k}"] = pd.DataFrame({"gene_id": [f"g{i}" for i in range(200)], "best_other_module": 0,
                                        "kme_residual": 0.3, "p_value": p, "q_value": p, "called": p < 0.05}, index=iso)
    naive = combine_target_tests(tables, correlation="none")
    corr = combine_target_tests(tables, correlation="empirical")
    rho = corr.attrs["cohort_z_correlation"]
    assert rho.values[np.triu_indices(5, 1)].mean() > 0.6
    assert corr["z_combined"].abs().mean() < naive["z_combined"].abs().mean()
    assert corr["called"].sum() < naive["called"].sum()
    strict = combine_target_tests(tables, correlation="none", min_called=3)
    assert strict["called"].sum() <= naive["called"].sum()
    with pytest.raises(ValueError):
        combine_target_tests(tables, correlation="bogus")
