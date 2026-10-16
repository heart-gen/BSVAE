"""
Calibration of the isoform-specific residual test.

Three checks on the transcript-level simulator:

  null        abundance-only scenarios (no switching, no usage regulation): every
              multi-isoform transcript is a true null for isoform-specific
              membership, so the caller's p-values should be uniform.  Reports the
              type-I error at 0.05 and 0.01, the Kolmogorov–Smirnov distance from
              uniform, and the fraction called at the default thresholds, for
              labels from a trained BSVAE (raw and after gene coherence), without,
              with the approximate and with the exact leave-one-gene-out eigengene, and stratified by whether
              the transcript's raw label is the module it is tested against (the
              selection-bias case).
  heldout     samples are split in half: modules are learned on half A and the
              caller is run on both halves with A's labels.  Reports the fraction
              of A-calls that replicate on B (true switches vs unswitched) and the
              null call rate on B for unswitched transcripts.
  ppv         with switching at a given prevalence, the observed positive
              predictive value among calls next to recall and the false-positive
              rate, so the two are never confused.

Usage:
  poetry run python scripts/benchmark_calibration.py --outdir results/calibration \\
    [--seeds 0,1,2] [--epochs 150] [--switch-rates 0.05,0.15]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sps

from bsvae.benchmark.runners import run_bsvae
from bsvae.networks.postprocess import enforce_gene_coherence
from bsvae.utils.isoform import isoform_specific_calls
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params

logger = logging.getLogger("calibration")
BASE = dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=600, n_samples=150, n_modules=10, null_fraction=0.10, usage_rate=0.0)


def train_labels(expr: pd.DataFrame, seed: int, workdir: Path, epochs: int) -> np.ndarray:
    cfg = {"epochs": epochs, "warmup_epochs": min(30, max(2, epochs // 3)), "transition_epochs": min(10, max(1, epochs // 10))}
    res = run_bsvae(expr, n_modules=10, seed=seed, workdir=workdir, train_cfg=cfg)
    if res.status != "ok":
        raise RuntimeError(res.message)
    return res.labels.reindex(expr.index.astype(str)).fillna(-1).astype(int).values


def null_rows(calls: pd.DataFrame, label: str, seed: int, variant: str) -> List[Dict]:
    rows = []
    p = calls["p_value"].values.astype(float)
    sel = calls["label_discordant"].values.astype(bool)
    for stratum, m in (("all", np.ones(len(p), bool)), ("label_in_tested_module", sel), ("label_concordant", ~sel)):
        if m.sum() < 10:
            continue
        pm = p[m]
        rows.append({"check": "null", "seed": seed, "labels": label, "eigengene": variant, "stratum": stratum, "n": int(m.sum()),
                     "type1_0.05": float((pm < 0.05).mean()), "type1_0.01": float((pm < 0.01).mean()),
                     "ks_from_uniform": float(sps.kstest(pm, "uniform").statistic),
                     "called_default": float(calls["called"].values[m].mean())})
    return rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/calibration")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--switch-rates", default="0.05,0.15")
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    rates = [float(x) for x in args.switch_rates.split(",")]
    rows: List[Dict] = []
    for seed in seeds:
        # ---- null calibration ----
        r = simulate_params(dict(BASE, switch_rate=0.0), seed)
        expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
        t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
        fids = list(expr.index)
        logger.info("seed %d | null: training", seed)
        raw = train_labels(expr, seed, outdir / f"seed_{seed}" / "null", args.epochs)
        coh, _ = enforce_gene_coherence(raw, expr, fids, t2g)
        for label, lab in (("raw", raw), ("coherent", coh)):
            for variant, loo in (("plain", False), ("loo_approx", "approx"), ("loo_exact", "exact")):
                calls = isoform_specific_calls(expr, t2g, lab, fids, loo_eigengene=loo)
                calls.to_csv(outdir / f"seed_{seed}" / f"null_calls_{label}_{variant}.tsv", sep="\t")
                new_rows = null_rows(calls, label, seed, variant)
                rows.extend(new_rows)
                logger.info("null | %s | %s: %d strata", label, variant, len(new_rows))
        pd.DataFrame(rows).to_csv(outdir / "results.tsv", sep="\t", index=False)
        # ---- held-out replication and PPV at each switch prevalence ----
        for rate in rates:
            r = simulate_params(dict(BASE, switch_rate=rate), seed)
            expr = r.expr_features_x_samples; expr.index = expr.index.astype(str)
            t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
            st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
            fids = list(expr.index)
            rng = np.random.default_rng(seed)
            cols = rng.permutation(expr.shape[1]); half = expr.shape[1] // 2
            A, B = expr.iloc[:, cols[:half]], expr.iloc[:, cols[half:]]
            logger.info("seed %d | switch %.2f: training on half A", seed, rate)
            labA = train_labels(A, seed, outdir / f"seed_{seed}" / f"heldout_{rate}", args.epochs)
            cA = isoform_specific_calls(A, t2g, labA, fids)
            cB = isoform_specific_calls(B, t2g, labA, fids)
            sw = st.loc[cA.index, "is_switched"].values == 1
            a_called = cA["called"].values; b_called = cB["called"].reindex(cA.index).fillna(False).values
            same_module = (cA["best_other_module"].values == cB["best_other_module"].reindex(cA.index).values)
            rows.append({"check": "heldout", "seed": seed, "switch_rate": rate, "n_switched": int(sw.sum()), "n_unswitched": int((~sw).sum()),
                         "replicated_switched": float((a_called & b_called & same_module)[sw].sum() / max(a_called[sw].sum(), 1)),
                         "replicated_unswitched": float((a_called & b_called & same_module)[~sw].sum() / max(a_called[~sw].sum(), 1)),
                         "null_call_rate_B": float(b_called[~sw].mean()), "recall_B": float(b_called[sw].mean())})
            # full-data PPV
            logger.info("seed %d | switch %.2f: training on all samples", seed, rate)
            lab = train_labels(expr, seed, outdir / f"seed_{seed}" / f"full_{rate}", args.epochs)
            calls = isoform_specific_calls(expr, t2g, lab, fids)
            sw = st.loc[calls.index, "is_switched"].values == 1
            c = calls["called"].values
            rows.append({"check": "ppv", "seed": seed, "switch_rate": rate, "prevalence": float(sw.mean()),
                         "recall": float(c[sw].mean()), "fpr": float(c[~sw].mean()),
                         "ppv_observed": float((c & sw).sum() / max(c.sum(), 1)), "n_called": int(c.sum())})
            pd.DataFrame(rows).to_csv(outdir / "results.tsv", sep="\t", index=False)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    for check in ("null", "heldout", "ppv"):
        sub = df[df["check"] == check].drop(columns=["check"])
        if sub.empty:
            continue
        sub = sub.dropna(axis=1, how="all")
        keys = [k for k in ("labels", "eigengene", "stratum", "switch_rate") if k in sub.columns]
        num = [c for c in sub.columns if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(sub[c])]
        print(f"== {check}")
        print(sub.groupby(keys, dropna=False)[num].mean().round(3).to_string() if keys else sub[num].mean().round(3).to_string())
    df.to_csv(outdir / "results.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
