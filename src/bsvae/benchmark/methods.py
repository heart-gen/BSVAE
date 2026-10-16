"""
Additional comparison methods for the benchmark harness.

* ``run_usage_network``: transcript-usage network — isoform fractions within
  each gene (logit of the linear-scale share, single-isoform genes excluded)
  clustered with WGCNA, so modules reflect co-varying splicing/usage rather
  than abundance.
* ``run_ica``: FastICA on the feature × sample matrix; a feature belongs to
  the component on which its standardised loading is largest, when that
  loading exceeds ``z_threshold`` (otherwise grey).  Components whose module
  would be smaller than ``min_module_size`` are dissolved.
* ``run_nmf``: non-negative matrix factorisation of the (non-negative) log
  matrix; a feature belongs to the basis column that carries the largest
  share of its reconstruction, when that share exceeds ``share_threshold``.
* ``run_acorde``: the acorde isoform co-expression pipeline (Arzalluz-Luque
  et al. 2022) re-implemented in R with ``dynamicTreeCut`` (percentile
  correlation reduces to Pearson correlation with one sample group):
  hierarchical clustering on 1 - |cor|, dynamic tree cut, within-cluster
  correlation filtering, metatranscript merging and forced expansion.

All runners return a :class:`MethodResult` with ``labels`` (feature_id → int,
-1 grey) so the harness scores them with the same metrics as BSVAE and WGCNA.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bsvae.benchmark.runners import MethodResult, run_wgcna

logger = logging.getLogger(__name__)

ACORDE_SCRIPT = Path(__file__).with_name("acorde.R")


def isoform_usage(
    expr: pd.DataFrame,
    tx2gene: Dict[str, str],
    pseudocount: float = 1.0,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """
    Logit isoform usage (transcripts of multi-isoform genes × samples) from a
    log2(x + 1) transcript matrix: share = isoform / gene total on the linear
    scale, clipped to [eps, 1 - eps] before the logit.
    """
    ids = expr.index.astype(str)
    genes = pd.Series([tx2gene.get(t) for t in ids], index=ids)
    counts = genes.value_counts()
    multi = genes.map(counts).fillna(0) >= 2
    sub = expr.loc[multi.values]
    g = genes.loc[sub.index]
    lin = np.maximum(np.power(2.0, sub.values) - pseudocount, 0.0)
    tot = pd.DataFrame(lin, index=g.values, columns=sub.columns).groupby(level=0).sum()
    share = lin / np.maximum(tot.reindex(g.values).values, 1e-12)
    share = np.clip(share, eps, 1.0 - eps)
    return pd.DataFrame(np.log(share / (1.0 - share)), index=sub.index, columns=sub.columns)


def run_usage_network(
    expr: pd.DataFrame,
    tx2gene: Dict[str, str],
    *,
    workdir: str | Path,
    seed: int = 42,
    wgcna_cfg: Optional[Dict] = None,
) -> MethodResult:
    """WGCNA on logit isoform usage; transcripts of single-isoform genes are grey."""
    t0 = time.perf_counter()
    usage = isoform_usage(expr, tx2gene)
    if usage.shape[0] < 20:
        return MethodResult("usage_network", "skipped", message="fewer than 20 multi-isoform transcripts",
                            runtime_s=time.perf_counter() - t0)
    cfg = dict(wgcna_cfg or {})
    cfg.setdefault("network_type", "signed")
    res = run_wgcna(usage, workdir=workdir, seed=seed, **cfg)
    if res.status != "ok":
        res.method = "usage_network"
        return res
    labels = pd.Series(-1, index=expr.index.astype(str), name="module")
    labels.loc[res.labels.index] = res.labels.values
    params = dict(res.params); params["n_usage_transcripts"] = int(usage.shape[0])
    return MethodResult("usage_network", "ok", labels=labels, eigengenes=res.eigengenes,
                        runtime_s=time.perf_counter() - t0, params=params)


def _dissolve_small(labels: np.ndarray, min_size: int) -> np.ndarray:
    out = labels.copy()
    for m, n in zip(*np.unique(out[out >= 0], return_counts=True)):
        if n < min_size:
            out[out == m] = -1
    # relabel 0..M-1
    uniq = sorted(set(out[out >= 0].tolist()))
    remap = {m: i for i, m in enumerate(uniq)}
    return np.array([remap.get(v, -1) for v in out])


def run_ica(
    expr: pd.DataFrame,
    *,
    n_components: int,
    workdir: str | Path,
    seed: int = 42,
    z_threshold: float = 2.0,
    min_module_size: int = 10,
    max_iter: int = 1000,
) -> MethodResult:
    """FastICA modules: feature → component with the largest standardised loading (|z| ≥ threshold)."""
    from sklearn.decomposition import FastICA

    t0 = time.perf_counter()
    params = dict(n_components=int(n_components), z_threshold=z_threshold, min_module_size=min_module_size, seed=seed)
    X = expr.values.astype(np.float64)
    X = (X - X.mean(axis=1, keepdims=True)) / np.where(X.std(axis=1, keepdims=True) == 0, 1.0, X.std(axis=1, keepdims=True))
    k = int(min(n_components, min(X.shape) - 1))
    try:
        ica = FastICA(n_components=k, random_state=int(seed), max_iter=int(max_iter), whiten="unit-variance")
        S = ica.fit_transform(X)                     # features × components (feature loadings)
    except Exception as exc:  # noqa: BLE001
        return MethodResult("ica", "failed", params=params, runtime_s=time.perf_counter() - t0, message=str(exc))
    Z = (S - S.mean(axis=0, keepdims=True)) / np.where(S.std(axis=0, keepdims=True) == 0, 1.0, S.std(axis=0, keepdims=True))
    best = np.abs(Z).argmax(axis=1)
    labels = np.where(np.abs(Z[np.arange(len(Z)), best]) >= float(z_threshold), best, -1)
    labels = _dissolve_small(labels, int(min_module_size))
    out = pd.Series(labels, index=expr.index.astype(str), name="module")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_id": out.index, "module": out.values}).to_csv(Path(workdir) / "modules.csv", index=False)
    params["n_modules"] = int(len(set(labels[labels >= 0])))
    return MethodResult("ica", "ok", labels=out, runtime_s=time.perf_counter() - t0, params=params)


def run_nmf(
    expr: pd.DataFrame,
    *,
    n_components: int,
    workdir: str | Path,
    seed: int = 42,
    share_threshold: float = 0.5,
    min_module_size: int = 10,
    max_iter: int = 500,
) -> MethodResult:
    """NMF modules: feature → basis column carrying the largest share of its reconstruction."""
    from sklearn.decomposition import NMF

    t0 = time.perf_counter()
    params = dict(n_components=int(n_components), share_threshold=share_threshold, min_module_size=min_module_size, seed=seed)
    X = expr.values.astype(np.float64)
    X = X - X.min()
    k = int(min(n_components, min(X.shape)))
    try:
        nmf = NMF(n_components=k, init="nndsvda", random_state=int(seed), max_iter=int(max_iter))
        W = nmf.fit_transform(X)                     # features × components
    except Exception as exc:  # noqa: BLE001
        return MethodResult("nmf", "failed", params=params, runtime_s=time.perf_counter() - t0, message=str(exc))
    H_norm = np.linalg.norm(nmf.components_, axis=1)  # weight each component by its sample-profile norm
    contrib = W * H_norm[None, :]
    share = contrib / np.maximum(contrib.sum(axis=1, keepdims=True), 1e-12)
    best = share.argmax(axis=1)
    labels = np.where(share[np.arange(len(share)), best] >= float(share_threshold), best, -1)
    labels = _dissolve_small(labels, int(min_module_size))
    out = pd.Series(labels, index=expr.index.astype(str), name="module")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature_id": out.index, "module": out.values}).to_csv(Path(workdir) / "modules.csv", index=False)
    params["n_modules"] = int(len(set(labels[labels >= 0])))
    return MethodResult("nmf", "ok", labels=out, runtime_s=time.perf_counter() - t0, params=params)


def acorde_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        proc = subprocess.run([shutil.which("Rscript"), "-e", "quit(status = as.integer(!requireNamespace('dynamicTreeCut', quietly = TRUE)))"],
                              capture_output=True, text=True, timeout=120)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def run_acorde(
    expr: pd.DataFrame,
    *,
    workdir: str | Path,
    seed: int = 42,
    deep_split: int = 3,
    min_cluster_size: int = 20,
    min_cor: float = 0.8,
    lowcor_threshold: Optional[int] = None,
    size_threshold: int = 3,
    merge_height: float = 0.2,
    expand: bool = False,
    signed: bool = False,
    timeout_s: int = 7200,
) -> MethodResult:
    """
    acorde-style isoform co-expression clustering via ``Rscript`` (see ``acorde.R``).

    acorde's within-cluster correlation filter (drop a transcript with three or
    more within-cluster correlations below ``min_cor`` = 0.8) and forced
    expansion of unclustered transcripts were tuned for percentile
    correlations on single-cell data; on bulk Pearson correlations the filter
    empties every cluster at any threshold (within-module correlations of a
    50-member bulk module are rarely all above 0.4), so it is off by default
    (``lowcor_threshold=None``) and unclustered transcripts stay grey
    (``expand=False``).  Pass ``lowcor_threshold=3`` and ``expand=True`` for
    the published settings.
    """
    lowcor = int(lowcor_threshold) if lowcor_threshold is not None else 10 ** 9
    params = dict(deep_split=deep_split, min_cluster_size=min_cluster_size, min_cor=min_cor,
                  lowcor_threshold=lowcor_threshold, size_threshold=size_threshold,
                  merge_height=merge_height, expand=expand, signed=signed, seed=seed)
    if not acorde_available():
        return MethodResult("acorde", "skipped", params=params, message="Rscript or dynamicTreeCut not found")
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    expr_path = workdir / "expr.csv.gz"
    expr.to_csv(expr_path, compression="gzip")
    cmd = [shutil.which("Rscript"), "--vanilla", str(ACORDE_SCRIPT), "--expr", str(expr_path), "--outdir", str(workdir),
           "--deep-split", str(int(deep_split)), "--min-cluster-size", str(int(min_cluster_size)),
           "--min-cor", str(float(min_cor)), "--lowcor-threshold", str(lowcor),
           "--size-threshold", str(int(size_threshold)), "--merge-height", str(float(merge_height)),
           "--expand", "true" if expand else "false", "--signed", "true" if signed else "false", "--seed", str(int(seed))]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return MethodResult("acorde", "failed", params=params, runtime_s=time.perf_counter() - t0, message=f"timeout after {timeout_s}s")
    runtime = time.perf_counter() - t0
    (workdir / "acorde.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        return MethodResult("acorde", "failed", params=params, runtime_s=runtime, message="Rscript failed: " + " | ".join(tail))
    mods = pd.read_csv(workdir / "modules.csv")
    labels = pd.Series(mods["module"].astype(int).values, index=mods["feature_id"].astype(str), name="module")
    try:
        params.update(json.loads((workdir / "params.json").read_text()))
    except Exception:  # pragma: no cover
        pass
    return MethodResult("acorde", "ok", labels=labels, runtime_s=runtime, params=params)


TWN_SCRIPT = Path(__file__).with_name("twn.R")


def twn_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        proc = subprocess.run([shutil.which("Rscript"), "-e", "quit(status = as.integer(!requireNamespace('glasso', quietly = TRUE)))"],
                              capture_output=True, text=True, timeout=120)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def twn_inputs(expr: pd.DataFrame, tx2gene: Dict[str, str], pseudocount: float = 1.0, eps: float = 1e-3):
    """
    TWN node matrices from a log2 transcript matrix: total expression (samples ×
    genes, log2 of the linear-scale isoform sum) and isoform ratios (samples ×
    isoforms of multi-isoform genes, isoform / gene total on the linear scale).
    """
    ids = expr.index.astype(str)
    genes = pd.Series([tx2gene.get(t) for t in ids], index=ids)
    keep = genes.notna().values
    lin = np.maximum(np.power(2.0, expr.values[keep]) - pseudocount, 0.0)
    tot = pd.DataFrame(lin, index=genes.values[keep], columns=expr.columns).groupby(level=0).sum()
    te = np.log2(tot + 1.0).T                                        # samples × genes
    counts = genes.value_counts()
    multi = (genes.map(counts).fillna(0) >= 2).values & keep
    g_multi = genes.values[multi]
    share = np.maximum(np.power(2.0, expr.values[multi]) - pseudocount, 0.0) / np.maximum(tot.reindex(g_multi).values, 1e-12)
    ir = pd.DataFrame(np.clip(share, eps, 1 - eps), index=ids[multi], columns=expr.columns).T   # samples × isoforms
    return te, ir


def run_twn(
    expr: pd.DataFrame,
    tx2gene: Dict[str, str],
    *,
    workdir: str | Path,
    lambda_tt: float = 0.5,
    lambda_ti: float = 0.4,
    lambda_ii: float = 0.4,
    lambda_d: float = 0.0,
    lambda_s: float = 0.05,
    max_iter: int = 100,
    timeout_s: int = 7200,
) -> MethodResult:
    """
    Transcriptome-wide network (Saha et al. 2017) via ``twn.R``: a sparse
    Gaussian graphical model over gene total expression and isoform ratios
    with the paper's penalty structure, solved with glasso (the original uses
    QUIC in MATLAB; the penalty matrix is built identically).  Returns an edge
    list in ``edges`` (no module labels); score with
    :func:`bsvae.benchmark.edges.edge_metrics`.
    """
    params = dict(lambda_tt=lambda_tt, lambda_ti=lambda_ti, lambda_ii=lambda_ii, lambda_d=lambda_d, lambda_s=lambda_s)
    if not twn_available():
        return MethodResult("twn", "skipped", params=params, message="Rscript or glasso not found")
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    te, ir = twn_inputs(expr, tx2gene)
    if ir.shape[1] < 2:
        return MethodResult("twn", "skipped", params=params, message="fewer than two isoform-ratio nodes")
    te.to_csv(workdir / "te.csv.gz", compression="gzip")
    ir.to_csv(workdir / "ir.csv.gz", compression="gzip")
    pd.DataFrame({"transcript_id": list(tx2gene.keys()), "gene_id": list(tx2gene.values())}).to_csv(workdir / "tx2gene.tsv", sep="\t", index=False)
    cmd = [shutil.which("Rscript"), "--vanilla", str(TWN_SCRIPT), "--te", str(workdir / "te.csv.gz"), "--ir", str(workdir / "ir.csv.gz"),
           "--tx2gene", str(workdir / "tx2gene.tsv"), "--outdir", str(workdir),
           "--lambda-tt", str(lambda_tt), "--lambda-ti", str(lambda_ti), "--lambda-ii", str(lambda_ii),
           "--lambda-d", str(lambda_d), "--lambda-s", str(lambda_s), "--max-iter", str(int(max_iter))]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return MethodResult("twn", "failed", params=params, runtime_s=time.perf_counter() - t0, message=f"timeout after {timeout_s}s")
    runtime = time.perf_counter() - t0
    (workdir / "twn.log").write_text(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        return MethodResult("twn", "failed", params=params, runtime_s=runtime, message="Rscript failed: " + " | ".join(tail))
    edges = pd.read_csv(workdir / "edges.tsv", sep="\t")
    try:
        params.update(json.loads((workdir / "params.json").read_text()))
    except Exception:  # pragma: no cover
        pass
    return MethodResult("twn", "ok", edges=edges, runtime_s=runtime, params=params)


__all__ = ["isoform_usage", "run_usage_network", "run_ica", "run_nmf", "run_acorde", "acorde_available",
           "twn_inputs", "run_twn", "twn_available"]
