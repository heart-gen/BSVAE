"""
Benchmark harness: scenario grid × methods × seeds → tidy results table.

``run_grid`` generates (or reuses) each scenario replicate, runs every
requested method, scores it with the grey-aware metrics and appends one
row per (scenario, rep, method, seed) to ``results.tsv`` immediately, so a
crashed or interrupted run can be resumed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import (
    BSVAE_DEFAULTS,
    POST_DEFAULTS,
    WGCNA_DEFAULTS,
    MethodResult,
    run_bsvae,
    run_wgcna,
    wgcna_available,
)
from bsvae.simulation.metrics import evaluate_labels, gene_level_metrics, normalize_labels, switched_isoform_metrics
from bsvae.simulation.scenario import expand_scenarios, generate_scenario, load_config

logger = logging.getLogger(__name__)

METHODS = ("bsvae", "wgcna")
KNOWN_METHODS = ("bsvae", "bsvae_consensus", "bsvae_isoform", "bsvae_regressed", "wgcna", "wgcna_signed", "wgcna_unsigned", "wgcna_regressed")


# ---------------------------------------------------------------------------
# Truth loading and scoring
# ---------------------------------------------------------------------------

def load_truth(run_dir: str | Path) -> Dict:
    """Load the truth files written by ``generate_scenario``."""
    run_dir = Path(run_dir)
    truth = run_dir / "truth"
    hard = pd.read_csv(truth / "modules_hard.csv")
    out: Dict = {
        "labels": pd.Series(hard["module"].astype(int).values, index=hard["feature_id"].astype(str)),
        "expr_path": run_dir / "expr" / "features_x_samples.tsv.gz",
        "covariates_path": run_dir / "covariates.tsv.gz",
        "metadata": json.loads((run_dir / "metadata.json").read_text()),
    }
    tx2gene_path = truth / "tx2gene.tsv"
    if tx2gene_path.exists():
        t = pd.read_csv(tx2gene_path, sep="\t")
        out["tx2gene"] = dict(zip(t["transcript_id"].astype(str), t["gene_id"].astype(str)))
        g = pd.read_csv(truth / "modules_gene.csv")
        out["truth_gene"] = dict(zip(g["gene_id"].astype(str), g["module"].astype(int)))
    sw = truth / "switch_table.tsv.gz"
    if sw.exists():
        out["switch_table"] = pd.read_csv(sw, sep="\t")
    return out


def evaluate_run(truth: Dict, result: MethodResult) -> Dict[str, float]:
    """Score a MethodResult against loaded truth; NaN metrics when not ok."""
    if result.status != "ok" or result.labels is None:
        return {}
    if result.method == "bsvae_isoform":
        if "switch_table" not in truth:
            return {}
        pred = result.labels.copy(); pred.index = pred.index.astype(str)
        return switched_isoform_metrics(dict(zip(pred.index, normalize_labels(pred.values))), list(pred.index), truth["switch_table"])
    truth_labels: pd.Series = truth["labels"]
    pred = result.labels.copy()
    pred.index = pred.index.astype(str)
    common = truth_labels.index.intersection(pred.index)
    if len(common) < 0.9 * len(truth_labels):
        raise ValueError(
            f"Only {len(common)}/{len(truth_labels)} truth features present in {result.method} output"
        )
    t = truth_labels.loc[common].values
    p = normalize_labels(pred.loc[common].values)
    metrics = evaluate_labels(t, p)
    if "tx2gene" in truth:
        metrics.update(
            gene_level_metrics(
                dict(zip(common, p)), list(common), truth["tx2gene"], truth["truth_gene"],
                truth_tx=dict(zip(truth_labels.index, truth_labels.values)),
            )
        )
    return metrics


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def _flatten_params(params: Dict, prefix: str = "") -> Dict:
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            flat[f"{prefix}{k}"] = json.dumps(v, sort_keys=True)
        elif isinstance(v, (list, tuple)):
            flat[f"{prefix}{k}"] = ";".join(map(str, v))
        else:
            flat[f"{prefix}{k}"] = v
    return flat


def _append_row(path: Path, row: Dict) -> None:
    """Append one row, keeping a single consistent header (columns are the union)."""
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, sep="\t")
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, sep="\t", index=False)


def _done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", usecols=["scenario_id", "rep", "method", "seed"])
    return {tuple(r) for r in df.itertuples(index=False, name=None)}


def run_one(
    run_dir: Path,
    method: str,
    seed: int,
    *,
    k_mode: str = "oracle",
    bsvae_cfg: Optional[Dict] = None,
    wgcna_cfg: Optional[Dict] = None,
    post_cfg: Optional[Dict] = None,
    postprocess: bool = True,
    device=None,
    over_cluster_factor: Optional[float] = None,
    consensus_runs: int = 3,
) -> tuple[MethodResult, Dict]:
    truth = load_truth(run_dir)
    expr = pd.read_csv(truth["expr_path"], sep="\t", index_col=0)
    expr.index = expr.index.astype(str)
    if method.endswith("_regressed"):
        from bsvae.utils.preprocess import regress_covariates
        cov = pd.read_csv(truth["covariates_path"], sep="\t", index_col=0)
        # Known technical covariate only (batch); condition is biology.
        expr = regress_covariates(expr, cov, columns=["batch"])
        method_base = method[: -len("_regressed")]
    else:
        method_base = method
    out_dir = run_dir / "methods" / method / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_modules = int(truth["metadata"]["summary"].get("n_modules", truth["metadata"]["parameters"].get("n_modules", 20)))

    if method_base == "bsvae":
        cfg = dict(bsvae_cfg or {})
        if k_mode == "sweep":
            cfg.setdefault("_k_mode", "sweep")
        tx2gene = None
        if "tx2gene" in truth:
            tx2gene = pd.DataFrame(list(truth["tx2gene"].items()), columns=["transcript_id", "gene_id"])
        kwargs = {}
        if over_cluster_factor is not None:
            kwargs["over_cluster_factor"] = over_cluster_factor
        result = run_bsvae(
            expr, n_modules=n_modules, seed=seed, workdir=out_dir, tx2gene=tx2gene,
            train_cfg={k: v for k, v in cfg.items() if not k.startswith("_")},
            post_cfg=post_cfg, device=device, postprocess=postprocess, **kwargs,
        )
    elif method == "bsvae_consensus":
        from bsvae.networks.postprocess import consensus_assignments
        cfg = {k: v for k, v in (bsvae_cfg or {}).items() if not k.startswith("_")}
        tx2gene = None
        if "tx2gene" in truth:
            tx2gene = pd.DataFrame(list(truth["tx2gene"].items()), columns=["transcript_id", "gene_id"])
        kwargs = {"over_cluster_factor": over_cluster_factor} if over_cluster_factor is not None else {}
        runs, t_total = [], 0.0
        for i in range(int(consensus_runs)):
            sub = run_bsvae(
                expr, n_modules=n_modules, seed=seed * 1000 + i, workdir=out_dir / f"run_{i}",
                tx2gene=tx2gene, train_cfg=cfg, post_cfg=post_cfg, device=device,
                postprocess=postprocess, **kwargs,
            )
            t_total += sub.runtime_s
            if sub.status == "ok":
                runs.append(sub.labels.reindex(expr.index).fillna(-1).astype(int).values)
        if len(runs) < 2:
            result = MethodResult("bsvae_consensus", "failed", runtime_s=t_total, message="fewer than two successful runs")
        else:
            cons = consensus_assignments(runs, min_agreement=0.5, min_module_size=(post_cfg or {}).get("min_module_size", 10))
            result = MethodResult(
                "bsvae_consensus", "ok", labels=pd.Series(cons.labels, index=expr.index.astype(str), name="module"),
                runtime_s=t_total,
                params={"n_runs": cons.n_runs, "run_ari_to_consensus": cons.run_ari,
                        "mean_agreement": float(cons.agreement.mean())},
            )
    elif method == "bsvae_isoform":
        from bsvae.utils.isoform import isoform_specific_calls
        if "tx2gene" not in truth:
            result = MethodResult("bsvae_isoform", "skipped", message="scenario has no transcript structure")
        else:
            cfg = {k: v for k, v in (bsvae_cfg or {}).items() if not k.startswith("_")}
            kwargs = {"over_cluster_factor": over_cluster_factor} if over_cluster_factor is not None else {}
            sub = run_bsvae(expr, n_modules=n_modules, seed=seed, workdir=out_dir, train_cfg=cfg,
                            post_cfg=post_cfg, device=device, postprocess=postprocess, **kwargs)
            if sub.status == "ok":
                fids = [str(f) for f in expr.index]
                lab = sub.labels.reindex(fids).fillna(-1).astype(int).values
                min_kme = float((post_cfg or {}).get("isoform_min_kme", 0.3))
                calls = isoform_specific_calls(expr, truth["tx2gene"], lab, fids, min_kme=min_kme, eigengenes=sub.eigengenes)
                labels = pd.Series(-1, index=fids, name="module")
                called = calls[calls["called"]]
                labels.loc[called.index] = called["best_other_module"].astype(int).values
                calls.to_csv(out_dir / "isoform_specific_calls.csv")
                result = MethodResult("bsvae_isoform", "ok", labels=labels, runtime_s=sub.runtime_s,
                                      params={**sub.params, "isoform_min_kme": min_kme, "n_called": int(len(called))})
            else:
                result = MethodResult("bsvae_isoform", sub.status, runtime_s=sub.runtime_s, message=sub.message)
    elif method_base in ("wgcna", "wgcna_unsigned", "wgcna_signed"):
        cfg = dict(WGCNA_DEFAULTS)
        cfg.update(wgcna_cfg or {})
        if method_base == "wgcna_unsigned":
            cfg["network_type"] = "unsigned"
        elif method_base == "wgcna_signed":
            cfg["network_type"] = "signed"
        result = run_wgcna(expr, workdir=out_dir, seed=seed, **cfg)
    else:
        raise ValueError(f"Unknown method: {method}")

    result.method = method
    metrics = evaluate_run(truth, result) if result.status == "ok" else {}
    if result.labels is not None:
        pd.DataFrame({"feature_id": result.labels.index, "module": result.labels.values}).to_csv(
            out_dir / "modules.csv", index=False,
        )
    (out_dir / "metrics.json").write_text(json.dumps(
        {"status": result.status, "message": result.message, "runtime_s": result.runtime_s,
         "params": result.params, "metrics": metrics}, indent=2, default=str,
    ))
    return result, metrics


def run_grid(
    config: str | Path | Dict,
    outdir: str | Path,
    *,
    methods: Sequence[str] = METHODS,
    reps: Optional[int] = None,
    seeds: Optional[Iterable[int]] = None,
    k_mode: str = "oracle",
    bsvae_cfg: Optional[Dict] = None,
    wgcna_cfg: Optional[Dict] = None,
    post_cfg: Optional[Dict] = None,
    postprocess: bool = True,
    resume: bool = True,
    scenario_ids: Optional[Sequence[str]] = None,
    device=None,
    over_cluster_factor: Optional[float] = None,
    consensus_runs: int = 3,
) -> pd.DataFrame:
    """Run every (scenario, rep, method, seed) and return the tidy results table."""
    cfg = load_config(config) if not isinstance(config, dict) else config
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "results.tsv"
    done = _done_keys(results_path) if resume else set()
    if not resume and results_path.exists():
        results_path.unlink()

    scenarios = expand_scenarios(cfg)
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [(sid, p) for sid, p in scenarios if sid in wanted]
    n_reps = int(cfg.get("reps", 1) if reps is None else reps)
    seed_list = list(seeds) if seeds is not None else [0]
    methods = [m for m in methods]
    unknown = [m for m in methods if m not in KNOWN_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}; choose from {KNOWN_METHODS}")
    if any(m.startswith("wgcna") for m in methods) and not wgcna_available():
        logger.warning("WGCNA (Rscript + WGCNA package) not available; its rows will be marked skipped")

    for sid, params in scenarios:
        for rep in range(n_reps):
            run_dir = outdir / "scenarios" / sid / f"rep_{rep:03d}"
            if not (run_dir / "metadata.json").exists():
                generate_scenario(cfg, sid, rep=rep, outdir=outdir)
            meta = json.loads((run_dir / "metadata.json").read_text())
            factors = _flatten_params(meta["parameters"], prefix="p_")
            summary = _flatten_params(meta["summary"], prefix="s_")
            for method in methods:
                for seed in seed_list:
                    key = (sid, rep, method, seed)
                    if key in done:
                        continue
                    logger.info("Running %s | %s rep %d seed %d", method, sid, rep, seed)
                    t0 = time.perf_counter()
                    result, metrics = run_one(
                        run_dir, method, seed, k_mode=k_mode, bsvae_cfg=bsvae_cfg,
                        wgcna_cfg=wgcna_cfg, post_cfg=post_cfg, postprocess=postprocess, device=device,
                        over_cluster_factor=over_cluster_factor, consensus_runs=consensus_runs,
                    )
                    row = {
                        "scenario_id": sid, "rep": rep, "method": method, "seed": seed,
                        "status": result.status, "message": result.message,
                        "runtime_s": round(result.runtime_s, 2), "wall_s": round(time.perf_counter() - t0, 2),
                        "family": meta["parameters"].get("family", "nb_latent_modules"),
                    }
                    row.update(factors)
                    row.update(summary)
                    row.update(metrics)
                    row.update(_flatten_params({k: v for k, v in result.params.items()
                                                if k in ("soft_power", "n_modules", "n_grey", "n_components_trained",
                                                         "n_runs", "mean_agreement")}, prefix="m_"))
                    _append_row(results_path, row)
                    done.add(key)
    df = pd.read_csv(results_path, sep="\t") if results_path.exists() else pd.DataFrame()
    if not df.empty:
        summarize(df).to_csv(outdir / "summary.tsv", sep="\t", index=False)
    return df


SUMMARY_METRICS = [
    "ari_all", "nmi_all", "ari_assigned", "nmi_assigned", "mean_best_match_jaccard",
    "module_recovery_50", "frac_pred_grey", "grey_recall", "grey_precision", "K_ratio",
    "gene_ari", "split_detection_rate", "isoform_gene_agreement",
    "switched_recall", "switched_precision", "switched_ari", "runtime_s",
]


def summarize(df: pd.DataFrame, by: Sequence[str] = ("scenario_id", "method")) -> pd.DataFrame:
    """Mean ± sd per group for the headline metrics (ok rows only)."""
    ok = df[df["status"] == "ok"] if "status" in df else df
    cols = [c for c in SUMMARY_METRICS if c in ok.columns]
    if ok.empty or not cols:
        return pd.DataFrame(columns=list(by))
    g = ok.groupby(list(by))[cols]
    mean = g.mean().add_suffix("_mean")
    sd = g.std(ddof=0).add_suffix("_sd")
    n = g.size().rename("n_runs")
    return pd.concat([n, mean, sd], axis=1).reset_index()


__all__ = ["run_grid", "run_one", "evaluate_run", "load_truth", "summarize", "METHODS"]
