"""
Defaults sweep for the isoform-specific residual scan.

One *cell* of the scenario grid (feature count, sample count, module count,
module-factor correlation, signal scale) is simulated per seed; the models the
scan needs are trained once per dataset and every post-processing setting is
then evaluated on the saved labels, so the sweep costs one training set per
dataset rather than one per setting:

  type1     abundance-only data (no switching).  Trained: the full-data model
            and one model per sample fold (two folds).  Every setting reports
            the type-I error at 0.05 and 0.01 and the fraction called at the
            default thresholds, overall and by whether the transcript's label
            is the module it is tested against (``label_in_tested_module``,
            the selection-bias stratum) or not (``label_concordant``).
  power     one dataset per switch prevalence.  Trained: the full-data model
            and one model per half (the halves double as cross-fit folds).
            Every setting reports recall, false-positive rate, observed
            positive predictive value and the observed false-discovery rate
            among calls.

Settings (the factorial ``design`` x ``covariates`` x ``max_target_corr`` x
``loo``; ``module_agreement`` additionally for cross-fitting):

  design            ``full`` (one model, leave-one-gene-out eigengene) or
                    ``crossfit`` (two fold models, Stouffer-combined)
  covariates        regress log library size (from the simulated counts) out of
                    residuals and eigengenes, or not
  max_target_corr   exclude target modules whose eigengene correlates with the
                    gene's module above this (``none`` keeps every target)
  loo               ``approx`` or ``exact`` leave-one-gene-out eigengene
                    (cross-fitting uses ``approx`` only)
  module_agreement  cross-fitting only: require both folds to pick the same
                    target module

Output: ``<outdir>/results.tsv`` with one row per (seed, check, setting,
stratum or switch rate) and the cell parameters as columns, and the trained
labels under ``<outdir>/seed_<s>/labels_<name>.npy`` so settings can be
re-evaluated without retraining (``--reuse-labels``).

Usage:
  poetry run python scripts/benchmark_defaults.py --outdir results/defaults/cell \\
    --n-genes 6000 --n-samples 300 --n-modules 30 --factor-corr 0.3 \\
    --seeds 0 --epochs 150 --switch-rates 0.05,0.15
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from bsvae.benchmark.runners import run_bsvae
from bsvae.simulation.scenario import ISOFORM_GENERATOR_DEFAULTS, simulate_params
from bsvae.utils.isoform import cross_fit_calls, isoform_specific_calls

logger = logging.getLogger("defaults")
STRATA = (("all", None), ("label_in_tested_module", True), ("label_concordant", False))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="results/defaults/cell")
    p.add_argument("--seeds", default="0")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--n-genes", type=int, default=1200)
    p.add_argument("--n-samples", type=int, default=150)
    p.add_argument("--n-modules", type=int, default=12)
    p.add_argument("--factor-corr", type=float, default=float(ISOFORM_GENERATOR_DEFAULTS["factor_corr"]))
    p.add_argument("--signal-scale", type=float, default=1.9)
    p.add_argument("--null-fraction", type=float, default=0.10)
    p.add_argument("--dominant-isoform-frac", type=float, default=float(ISOFORM_GENERATOR_DEFAULTS["dominant_isoform_frac"]))
    p.add_argument("--switch-rates", default="0.05,0.15", help="Comma list; empty string skips the power check.")
    p.add_argument("--checks", default="type1,power")
    p.add_argument("--designs", default="full,crossfit")
    p.add_argument("--covariates", default="on,off")
    p.add_argument("--target-corrs", default="none,0.6,0.4,0.3")
    p.add_argument("--loo", default="approx,exact", help="Leave-one-gene-out eigengene variants for the full design.")
    p.add_argument("--min-kme", type=float, default=0.3)
    p.add_argument("--alpha", type=float, default=0.05, help="q-value threshold behind ``called``.")
    p.add_argument("--reuse-labels", action="store_true", default=False,
                   help="Load labels saved by a previous run of this cell instead of retraining.")
    p.add_argument("--no-cuda", action="store_true", default=False)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def _cell(args) -> Dict:
    return {"n_genes": args.n_genes, "n_samples": args.n_samples, "n_modules": args.n_modules,
            "factor_corr": args.factor_corr, "signal_scale": args.signal_scale}


def _params(args, switch_rate: float) -> Dict:
    return dict(ISOFORM_GENERATOR_DEFAULTS, n_genes=args.n_genes, n_samples=args.n_samples, n_modules=args.n_modules,
                factor_corr=args.factor_corr, signal_scale=args.signal_scale, null_fraction=args.null_fraction,
                dominant_isoform_frac=args.dominant_isoform_frac, usage_rate=0.0, switch_rate=switch_rate)


def _settings(args) -> List[Dict]:
    out = []
    tcorrs = [None if t.strip().lower() == "none" else float(t) for t in args.target_corrs.split(",") if t.strip()]
    covs = [c.strip().lower() == "on" for c in args.covariates.split(",") if c.strip()]
    loos = [l.strip() for l in args.loo.split(",") if l.strip()]
    for design in [d.strip() for d in args.designs.split(",") if d.strip()]:
        for cov in covs:
            for tc in tcorrs:
                if design == "full":
                    for loo in loos:
                        out.append({"design": "full", "covariates": cov, "max_target_corr": tc, "loo": loo, "module_agreement": None})
                elif design == "crossfit":
                    for agree in (False, True):
                        out.append({"design": "crossfit", "covariates": cov, "max_target_corr": tc, "loo": "approx", "module_agreement": agree})
                else:
                    raise ValueError(f"unknown design {design}")
    return out


def _setting_cols(s: Dict) -> Dict:
    return {"design": s["design"], "covariates": bool(s["covariates"]),
            "max_target_corr": np.nan if s["max_target_corr"] is None else float(s["max_target_corr"]),
            "loo": s["loo"], "module_agreement": s["module_agreement"]}


class Trainer:
    """Trains (or reloads) BSVAE labels for one dataset; labels are cached on disk per name."""

    def __init__(self, args, seed_dir: Path):
        self.args = args
        self.seed_dir = seed_dir
        seed_dir.mkdir(parents=True, exist_ok=True)

    def labels(self, name: str, expr: pd.DataFrame, seed: int, n_modules: int) -> np.ndarray:
        path = self.seed_dir / f"labels_{name}.npy"
        if self.args.reuse_labels and path.exists():
            lab = np.load(path)
            if len(lab) == expr.shape[0]:
                return lab
        cfg = {"epochs": self.args.epochs, "warmup_epochs": min(30, max(2, self.args.epochs // 3)),
               "transition_epochs": min(10, max(1, self.args.epochs // 10))}
        if self.args.no_cuda:
            cfg["no_cuda"] = True
        logger.info("training %s (%d features x %d samples)", name, *expr.shape)
        res = run_bsvae(expr, n_modules=n_modules, seed=seed, workdir=self.seed_dir / name, train_cfg=cfg)
        if res.status != "ok":
            raise RuntimeError(f"{name}: {res.message}")
        lab = res.labels.reindex(expr.index.astype(str)).fillna(-1).astype(int).values
        np.save(path, lab)
        return lab


def _libsize(counts: pd.DataFrame, columns) -> pd.DataFrame:
    return pd.DataFrame({"log_libsize": np.log(counts.sum(axis=0).reindex(columns).values + 1.0)}, index=columns)


def _calls(setting: Dict, expr, t2g, fids, cov, lab_full, lab_folds, fold, min_kme, alpha) -> pd.DataFrame:
    kw = dict(min_kme=min_kme, alpha=alpha, covariates=cov if setting["covariates"] else None,
              max_target_corr=setting["max_target_corr"])
    if setting["design"] == "full":
        return isoform_specific_calls(expr, t2g, lab_full, fids, loo_eigengene=setting["loo"], **kw)
    return cross_fit_calls(expr, t2g, lab_folds, fold, fids, require_module_agreement=bool(setting["module_agreement"]),
                           loo_eigengene=setting["loo"], **kw)


def null_rows(calls: pd.DataFrame, base: Dict) -> List[Dict]:
    rows = []
    p = calls["p_value"].values.astype(float)
    sel = calls["label_discordant"].values.astype(bool)
    for stratum, want in STRATA:
        m = np.ones(len(p), bool) if want is None else (sel == want)
        if m.sum() < 10:
            continue
        pm = p[m]
        rows.append({**base, "check": "type1", "stratum": stratum, "n": int(m.sum()),
                     "type1_0.05": float((pm < 0.05).mean()), "type1_0.01": float((pm < 0.01).mean()),
                     "ks_from_uniform": float(sps.kstest(pm, "uniform").statistic),
                     "called_default": float(calls["called"].values[m].mean())})
    return rows


def power_row(calls: pd.DataFrame, switched: np.ndarray, base: Dict, rate: float) -> Dict:
    c = calls["called"].values.astype(bool)
    n_called = int(c.sum())
    tp = int((c & switched).sum())
    return {**base, "check": "power", "switch_rate": rate, "n": int(len(c)), "n_switched": int(switched.sum()),
            "prevalence": float(switched.mean()), "recall": float(c[switched].mean()) if switched.any() else np.nan,
            "fpr": float(c[~switched].mean()) if (~switched).any() else np.nan,
            "ppv_observed": float(tp / n_called) if n_called else np.nan,
            "fdr_observed": float((n_called - tp) / n_called) if n_called else np.nan, "n_called": n_called}


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("bsvae").setLevel(logging.WARNING)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rates = [float(x) for x in args.switch_rates.split(",") if x.strip()]
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}
    settings = _settings(args)
    cell = _cell(args)
    rows: List[Dict] = []
    results_path = outdir / "results.tsv"

    def flush():
        pd.DataFrame(rows).to_csv(results_path, sep="\t", index=False)

    for seed in seeds:
        trainer = Trainer(args, outdir / f"seed_{seed}")
        if "type1" in checks:
            r = simulate_params(_params(args, 0.0), seed)
            expr = r.expr_features_x_samples; expr.index = expr.index.astype(str); expr.columns = expr.columns.astype(str)
            t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
            fids = list(expr.index)
            counts = r.counts_features_x_samples; counts.columns = counts.columns.astype(str)
            cov = _libsize(counts, expr.columns)
            designs = {s["design"] for s in settings}
            lab_full = trainer.labels("null_full", expr, seed, args.n_modules) if "full" in designs else None
            fold = np.random.default_rng(seed).permutation(expr.shape[1]) % 2
            lab_folds = None
            if "crossfit" in designs:
                lab_folds = [trainer.labels(f"null_xf{f}", expr.loc[:, fold != f], seed, args.n_modules) for f in (0, 1)]
            for s in settings:
                calls = _calls(s, expr, t2g, fids, cov, lab_full, lab_folds, fold, args.min_kme, args.alpha)
                rows.extend(null_rows(calls, {**cell, "seed": seed, **_setting_cols(s)}))
                logger.info("null | %s", _setting_cols(s))
            flush()
        if "power" in checks:
            for rate in rates:
                r = simulate_params(_params(args, rate), seed)
                expr = r.expr_features_x_samples; expr.index = expr.index.astype(str); expr.columns = expr.columns.astype(str)
                t2g = dict(zip(r.extra_truth["tx2gene.tsv"]["transcript_id"].astype(str), r.extra_truth["tx2gene.tsv"]["gene_id"].astype(str)))
                st = r.extra_truth["switch_table.tsv.gz"].set_index("transcript_id")
                fids = list(expr.index)
                counts = r.counts_features_x_samples; counts.columns = counts.columns.astype(str)
                cov = _libsize(counts, expr.columns)
                rng = np.random.default_rng(seed)
                perm = rng.permutation(expr.shape[1]); half = expr.shape[1] // 2
                fold = np.zeros(expr.shape[1], dtype=int); fold[perm[half:]] = 1
                designs = {s["design"] for s in settings}
                tag = f"{rate:g}"
                lab_full = trainer.labels(f"switch{tag}_full", expr, seed, args.n_modules) if "full" in designs else None
                lab_folds = None
                if "crossfit" in designs:
                    # labels_by_fold[f] are learned on every sample except fold f
                    lab_folds = [trainer.labels(f"switch{tag}_xf{f}", expr.loc[:, fold != f], seed, args.n_modules) for f in (0, 1)]
                for s in settings:
                    calls = _calls(s, expr, t2g, fids, cov, lab_full, lab_folds, fold, args.min_kme, args.alpha)
                    sw = st["is_switched"].reindex(calls.index).fillna(0).values == 1
                    rows.append(power_row(calls, sw, {**cell, "seed": seed, **_setting_cols(s)}, rate))
                    logger.info("power %.2f | %s", rate, _setting_cols(s))
                flush()
    flush()
    df = pd.DataFrame(rows)
    if not df.empty:
        pd.set_option("display.width", 250)
        keys = ["check", "design", "covariates", "max_target_corr", "loo", "module_agreement", "stratum", "switch_rate"]
        keys = [k for k in keys if k in df.columns]
        num = [c for c in ("type1_0.05", "called_default", "recall", "fpr", "ppv_observed", "fdr_observed") if c in df.columns]
        print(df.groupby(keys, dropna=False)[num].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    main()
