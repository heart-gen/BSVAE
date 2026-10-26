"""
Cross-region pooling of targeted-gene tests under biological heterogeneity and
shared donors.

Six 150-sample "regions" of one simulated truth with rare switched isoforms
(minor isoforms carry ~8 % of their gene) are generated with

  shared     every region carries the same switches,
  partial    each region keeps half of the base switches and adds 5 % of its own,
  specific   no base switch is shared; each region has its own 15 %,

each with independent donors or with the same donors in every region
(donor-level module-factor offsets shared across regions, which makes the
per-region tests dependent).  BSVAE is trained per region, the targeted test
is run per region on every isoform switched in any region plus an equal
number of never-switched genes, and the per-region tables are pooled with
Stouffer's method with and without Strube's dependent-cohort correction and
with an optional minimum number of single-region calls.  ``--min-kme`` is
the effect-size floor of both the per-region tests and the pooled call
(``combine_target_tests``); 0 removes it, as the BrainSEQ application does.

Reported per setting: recall of isoforms switched in 1, 2–3 or 4–6 regions,
the false-positive rate among never-switched isoforms, the observed positive
predictive value among calls, and the PPV projected to 5 % and 15 %
prevalence of true switches, so a false-positive rate is never read as a
false-discovery rate.

Usage:
  poetry run python scripts/benchmark_pooling.py --outdir results/pooling \\
    [--seeds 0,1] [--regions 6] [--epochs 150] [--n-perm 500]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import run_bsvae
from bsvae.networks.targets import combine_target_tests, resolve_target_transcripts, target_gene_tests
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params

logger = logging.getLogger("pooling")
ISO = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=150, n_modules=10, switch_rate=0.15, dominant_isoform_frac=0.9, alpha=1.0)
HET = {"shared": dict(switch_keep_frac=1.0, switch_extra_rate=0.0),
       "partial": dict(switch_keep_frac=0.5, switch_extra_rate=0.05),
       "specific": dict(switch_keep_frac=0.0, switch_extra_rate=0.15)}
DONOR = {"independent": 0.0, "shared_donors": 1.0}


def ppv(recall: float, fpr: float, prev: float) -> float:
    return recall * prev / max(recall * prev + fpr * (1 - prev), 1e-12)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/pooling")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--regions", type=int, default=6)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--min-kme", type=float, default=0.2, help="Per-region effect-size floor.")
    p.add_argument("--heterogeneity", default=",".join(HET))
    p.add_argument("--donors", default=",".join(DONOR))
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    n_reg = int(args.regions)
    tcfg = {"epochs": args.epochs, "warmup_epochs": min(30, max(2, args.epochs // 3)), "transition_epochs": min(10, max(1, args.epochs // 10))}
    rows: List[Dict] = []
    for het in [h.strip() for h in args.heterogeneity.split(",")]:
        for dname in [d.strip() for d in args.donors.split(",")]:
            dsd = DONOR[dname]
            for seed in seeds:
                tables, n_samples, sw_by_region = {}, {}, {}
                for r in range(n_reg):
                    cfg = dict(ISO, sample_seed=100 + r, **HET[het])
                    if dsd > 0:
                        cfg.update(donor_seed=555 + seed, donor_effect_sd=dsd)
                    res = simulate_params(cfg, seed=seed)
                    expr = res.expr_features_x_samples; expr.index = expr.index.astype(str)
                    t2g = dict(zip(res.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), res.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
                    st = res.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
                    sw_by_region[r] = set(st.index[st["is_switched"] == 1])
                    if r == 0:
                        base_st, tx_gene = st, pd.Series(t2g).reindex(expr.index)
                    logger.info("%s | %s | seed %d | region %d: training", het, dname, seed, r)
                    fit = run_bsvae(expr, n_modules=10, seed=seed, workdir=outdir / het / dname / f"seed{seed}" / f"region{r}", train_cfg=tcfg, device=dev)
                    if fit.status != "ok":
                        raise RuntimeError(fit.message)
                    lab = fit.labels.reindex(expr.index).fillna(-1).astype(int).values
                    tables[f"r{r}"] = (expr, t2g, lab)
                    n_samples[f"r{r}"] = expr.shape[1]
                union = set().union(*sw_by_region.values())
                n_sw_regions = {t: sum(t in s for s in sw_by_region.values()) for t in union}
                genes_sw = sorted({base_st.loc[t, "gene_id"] for t in union})
                rng = np.random.default_rng(seed)
                multi = base_st[(base_st["isoform_rank"] > 0) & (base_st["is_null"] == 0)].groupby("gene_id").size().index
                pool = [g for g in multi if g not in set(genes_sw)]
                ns_genes = list(rng.choice(pool, size=min(len(genes_sw), len(pool)), replace=False))
                targets, _ = resolve_target_transcripts(tx_gene, genes=genes_sw + ns_genes)
                tx = list(targets.index)
                n_reg_sw = np.array([n_sw_regions.get(t, 0) for t in tx])
                minor = base_st.loc[tx, "isoform_rank"].values > 0
                null = (n_reg_sw == 0) & minor
                per_region = {}
                for r in range(n_reg):
                    expr, t2g, lab = tables[f"r{r}"]
                    per_region[f"r{r}"] = target_gene_tests(expr, t2g, lab, list(expr.index), targets, null="permutation",
                                                            n_perm=args.n_perm, min_kme=args.min_kme, seed=seed)["isoforms"]

                def score(called: np.ndarray, label: str) -> Dict:
                    row = dict(heterogeneity=het, donors=dname, seed=seed, method=label, min_kme=float(args.min_kme),
                               n_targets=len(tx), n_null=int(null.sum()),
                               fpr=float(called[null].mean()), prevalence=float((n_reg_sw > 0).mean()))
                    for lo, hi, nm in [(1, 1, "sw_1region"), (2, 3, "sw_2to3"), (4, n_reg, f"sw_4to{n_reg}"), (1, n_reg, "sw_any")]:
                        m = (n_reg_sw >= lo) & (n_reg_sw <= hi)
                        row[f"recall_{nm}"] = float(called[m].mean()) if m.any() else np.nan
                        row[f"n_{nm}"] = int(m.sum())
                    tp = int((called & (n_reg_sw > 0)).sum())
                    row["n_called"] = int(called.sum()); row["ppv_observed"] = float(tp / max(called.sum(), 1))
                    for prev in (0.05, 0.15):
                        row[f"ppv_at_prev{prev}"] = ppv(row["recall_sw_any"], row["fpr"], prev)
                    return row

                rows.append(score(per_region["r0"]["called"].reindex(tx).fillna(False).values, "single_region"))
                for corr in ("none", "empirical"):
                    for mc in (0, 2):
                        comb = combine_target_tests(per_region, n_samples=n_samples, correlation=corr, min_called=mc,
                                                    min_kme=args.min_kme)
                        rows.append(score(comb["called"].reindex(tx).fillna(False).values, f"pooled_{corr}_mincalled{mc}"))
                        if corr == "empirical" and mc == 0:
                            rho = comb.attrs["cohort_z_correlation"].values
                            rows[-1]["mean_rho"] = float(rho[np.triu_indices(n_reg, 1)].mean())
                pd.DataFrame(rows).to_csv(outdir / "results.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    cols = [c for c in df.columns if c.startswith("recall_") or c in ("fpr", "ppv_observed", "ppv_at_prev0.05", "ppv_at_prev0.15", "mean_rho")]
    summ = df.groupby(["heterogeneity", "donors", "method"])[cols].mean()
    summ.to_csv(outdir / "summary.tsv", sep="\t")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
