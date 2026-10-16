"""
Isoform-aware simulation: transcripts nested in genes.

Generator family ``nb_isoform_modules`` (config keys under ``generator``,
all overridable from the grid; see ``ISOFORM_GENERATOR_DEFAULTS``):

n_genes / n_features
    Number of genes (or transcripts; ``n_genes`` wins when both are given).
isoforms_per_gene
    ``{"dist": "geometric", "p": 0.45, "min": 1, "max": 6}`` or an int.
n_modules, module_size_distribution
    Gene-level modules (long-tail Dirichlet sizes by default).
switch_rate
    Fraction of multi-isoform signal genes in which one non-dominant isoform
    follows a different module ("isoform switching").
alpha
    Mixing weight of the switched isoform toward its new module: 1.0 is a
    hard switch; 0.3–0.7 keeps partial co-expression with the gene locus.
    Ground truth assigns the switched transcript to the new module.
null_fraction
    Fraction of genes with no module loading; all their isoforms are grey
    (truth label -1), as in real data where WGCNA leaves features unassigned.
iso_offset_sd, dominant_isoform_frac
    Isoform-specific baseline offset and the share of the gene baseline
    given to the dominant isoform (minor isoforms are lower-expressed).
factor_corr, signal_scale, baseline_*, libsize_log_sd, dispersion,
confounding, nonlinear_mode, dropout_*, celltype_confound
    Inherited from the ``nb_latent_modules`` family.
id_version
    Append ``.1`` to Ensembl-style IDs (exercises version stripping).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from bsvae.simulation.scenario import (
    STREAM_NAMES,
    ScenarioResult,
    _build_module_sizes,
    _edge_truth,
    _latent_factors,
    _nb_observe,
    _signal_eta,
    spawn_streams,
)


def _isoform_counts(cfg, n_genes: int, rng: np.random.Generator) -> np.ndarray:
    if isinstance(cfg, (int, np.integer)):
        return np.full(n_genes, int(cfg), dtype=int)
    cfg = dict(cfg or {})
    lo = int(cfg.get("min", 1))
    hi = int(cfg.get("max", 6))
    dist = str(cfg.get("dist", "geometric"))
    if dist == "geometric":
        p = float(cfg.get("p", 0.45))
        draws = rng.geometric(p, size=n_genes)          # 1, 2, 3, ...
        return np.clip(draws + lo - 1, lo, hi).astype(int)
    if dist == "uniform":
        return rng.integers(lo, hi + 1, size=n_genes)
    raise ValueError(f"Unknown isoforms_per_gene dist: {dist}")


def simulate_isoform_counts(params: Dict, seed: int) -> ScenarioResult:
    """Generate transcript-level NB data with gene-level modules and isoform switching."""
    streams = spawn_streams(seed)
    rs, rf, rl, rb, ra, rn = (streams[n] for n in STREAM_NAMES)

    n_samples = int(params.get("n_samples", 300))
    K = int(params.get("n_modules", 16))
    iso_cfg = params.get("isoforms_per_gene", {"dist": "geometric", "p": 0.45, "min": 1, "max": 6})
    if "n_genes" in params:
        n_genes = int(params["n_genes"])
    else:
        mean_iso = float(iso_cfg) if isinstance(iso_cfg, (int, float)) else 1.0 / float(dict(iso_cfg).get("p", 0.45))
        n_genes = max(K * 2, int(round(int(params.get("n_features", 12000)) / mean_iso)))

    n_iso = _isoform_counts(iso_cfg, n_genes, rs)
    n_tx = int(n_iso.sum())
    gene_of_tx = np.repeat(np.arange(n_genes), n_iso)
    tx_rank = np.concatenate([np.arange(k) for k in n_iso])       # 0 = dominant isoform

    # Gene-level modules with null genes.
    null_fraction = float(params.get("null_fraction", 0.10))
    n_null = int(round(null_fraction * n_genes))
    is_null_gene = np.zeros(n_genes, dtype=bool)
    if n_null > 0:
        is_null_gene[rs.choice(n_genes, size=n_null, replace=False)] = True
    signal_genes = np.where(~is_null_gene)[0]
    sizes = _build_module_sizes(len(signal_genes), K, params.get("module_size_distribution", "long_tail"), rs)
    gene_modules = np.full(n_genes, -1, dtype=int)
    assigned = np.concatenate([np.full(sz, k, dtype=int) for k, sz in enumerate(sizes)])
    rs.shuffle(assigned)
    gene_modules[signal_genes] = assigned

    # Transcript modules with switching.
    tx_modules = gene_modules[gene_of_tx].copy()
    switch_rate = float(params.get("switch_rate", 0.15))
    alpha_mix = float(params.get("alpha", 1.0))
    is_switched = np.zeros(n_tx, dtype=bool)
    gene_starts = np.concatenate([[0], np.cumsum(n_iso)[:-1]])
    multi_signal = [g for g in signal_genes if n_iso[g] >= 2]
    if switch_rate > 0.0 and K > 1:
        for g in multi_signal:
            if rs.random() >= switch_rate:
                continue
            start = gene_starts[g]
            iso = start + int(rs.integers(1, n_iso[g]))          # never the dominant isoform
            other = [m for m in range(K) if m != gene_modules[g]]
            tx_modules[iso] = int(rs.choice(other))
            is_switched[iso] = True

    # Factors and loadings.
    signal_scale = float(params.get("signal_scale", 1.9))
    z = _latent_factors(rf, n_samples, K, float(params.get("factor_corr", 0.3)))
    W = np.zeros((n_tx, K), dtype=np.float64)
    load = np.abs(rl.normal(0.0, signal_scale, size=n_tx))
    sig_tx = tx_modules >= 0
    plain = sig_tx & ~is_switched
    W[np.where(plain)[0], tx_modules[plain]] = load[plain]
    sw = np.where(is_switched)[0]
    if sw.size:
        gm = gene_modules[gene_of_tx[sw]]
        W[sw, gm] += (1.0 - alpha_mix) * load[sw]
        W[sw, tx_modules[sw]] += alpha_mix * load[sw]
    W += rl.normal(0.0, 0.02, size=W.shape)

    hub_frac = float(params.get("hub_fraction", 0.05))
    hub_mult = float(params.get("hub_multiplier", 2.0))
    is_hub = np.zeros(n_tx, dtype=bool)
    if hub_frac > 0:
        for k in range(K):
            members = np.where(tx_modules == k)[0]
            if members.size == 0:
                continue
            n_h = max(1, int(hub_frac * members.size))
            chosen = rs.choice(members, size=min(n_h, members.size), replace=False)
            is_hub[chosen] = True
            W[chosen, k] *= hub_mult

    # Baselines: gene baseline split across isoforms + isoform-specific offset.
    gene_alpha = rb.normal(float(params.get("baseline_log_mean", 2.638)),
                           float(params.get("baseline_log_sd", 1.601)), size=n_genes)
    dom_frac = float(params.get("dominant_isoform_frac", 0.6))
    share = np.ones(n_tx, dtype=np.float64)
    for g in range(n_genes):
        start, k = gene_starts[g], n_iso[g]
        if k == 1:
            share[start] = 1.0
            continue
        rest = rb.dirichlet(np.ones(k - 1)) * (1.0 - dom_frac)
        share[start] = dom_frac
        share[start + 1: start + k] = rest
    alpha = gene_alpha[gene_of_tx] + np.log(np.maximum(share, 1e-3)) + rb.normal(0.0, float(params.get("iso_offset_sd", 0.3)), size=n_tx)

    eta_signal, func_ids = _signal_eta(z, W, str(params.get("nonlinear_mode", "off")), ra)

    affected_modules = set(rs.choice(np.arange(K), size=max(1, K // 5), replace=False)) if K else set()
    cond_mask = np.isin(tx_modules, list(affected_modules))

    obs = _nb_observe(
        eta_signal, alpha, generator_cfg=params,
        confounding=str(params.get("confounding", "none")),
        dropout_mode=str(params.get("dropout_mode", "off")),
        dropout_target=float(params.get("dropout_target", 0.5)),
        cond_feature_mask=cond_mask, rng_arm=ra, rng_noise=rn,
    )

    ver = ".1" if bool(params.get("id_version", False)) else ""
    tx_ids = [f"ENST{i:011d}{ver}" for i in range(n_tx)]
    gene_ids = [f"ENSG{g:011d}{ver}" for g in range(n_genes)]
    sample_ids = [f"sample_{j:04d}" for j in range(n_samples)]
    module_names = [f"M{k + 1:02d}" for k in range(K)]

    expr_sxg = pd.DataFrame(obs["expr"], index=sample_ids, columns=tx_ids)
    expr_gxs = expr_sxg.T
    counts_gxs = pd.DataFrame(obs["counts"].T, index=tx_ids, columns=sample_ids)
    covariates = obs["covariates"]
    covariates.index = pd.Index(sample_ids, name="sample_id")

    modules_hard = pd.DataFrame(
        {"feature_id": tx_ids, "module": tx_modules.astype(int), "is_null": (tx_modules < 0).astype(int)}
    )
    modules_gene = pd.DataFrame({"gene_id": gene_ids, "module": gene_modules.astype(int)})
    tx2gene = pd.DataFrame({"transcript_id": tx_ids, "gene_id": [gene_ids[g] for g in gene_of_tx]})
    switch_table = pd.DataFrame(
        {
            "transcript_id": tx_ids,
            "gene_id": [gene_ids[g] for g in gene_of_tx],
            "isoform_rank": tx_rank,
            "gene_module": gene_modules[gene_of_tx],
            "tx_module": tx_modules,
            "alpha": np.where(is_switched, alpha_mix, 0.0),
            "is_switched": is_switched.astype(int),
            "is_null": (tx_modules < 0).astype(int),
            "is_hub": is_hub.astype(int),
        }
    )
    modules_long = pd.DataFrame(
        [
            {"feature_id": tx_ids[i], "module": module_names[tx_modules[i]], "is_hub": int(is_hub[i]),
             "membership_weight": float(abs(W[i, tx_modules[i]]))}
            for i in range(n_tx) if tx_modules[i] >= 0
        ],
        columns=["feature_id", "module", "is_hub", "membership_weight"],
    )
    module_latents = pd.DataFrame(z, index=sample_ids, columns=module_names)

    memberships = [set([int(m)]) if m >= 0 else set() for m in tx_modules]
    module_members = {k: np.where(tx_modules == k)[0].tolist() for k in range(K)}
    edge_map = _edge_truth(W, memberships, module_members, int(params.get("truth_edge_top_k", 25)))
    edge_list = pd.DataFrame(
        [{"source": tx_ids[i], "target": tx_ids[j], "weight": w, "is_within_module": 1}
         for (i, j), w in edge_map.items()],
        columns=["source", "target", "weight", "is_within_module"],
    )

    gene_metadata = pd.DataFrame(
        {
            "feature_id": tx_ids,
            "gene_id": [gene_ids[g] for g in gene_of_tx],
            "primary_module": [module_names[k] if k >= 0 else "grey" for k in tx_modules],
            "isoform_rank": tx_rank,
            "n_isoforms": n_iso[gene_of_tx],
            "is_switched": is_switched.astype(int),
            "is_null": (tx_modules < 0).astype(int),
            "is_hub_any": is_hub.astype(int),
            "alpha": alpha,
            "dispersion": obs["theta"],
            "batch_affected": np.isin(np.arange(n_tx), obs["batch_affected"]).astype(int),
            "nonlinear_func": func_ids if func_ids is not None else 0,
        }
    )

    metadata = {
        "family": "nb_isoform_modules",
        "n_genes": n_genes,
        "n_transcripts": n_tx,
        "n_features": n_tx,
        "n_samples": n_samples,
        "n_modules": K,
        "seed": seed,
        "mean_isoforms_per_gene": float(n_iso.mean()),
        "switch_rate": switch_rate,
        "alpha": alpha_mix,
        "n_switched_transcripts": int(is_switched.sum()),
        "frac_switched_actual": float(is_switched.mean()),
        "null_fraction": null_fraction,
        "n_null_genes": int(n_null),
        "n_null_transcripts": int((tx_modules < 0).sum()),
        "signal_scale": signal_scale,
        "confounding": str(params.get("confounding", "none")),
        "nonlinear_mode": str(params.get("nonlinear_mode", "off")),
        "dropout_mode": str(params.get("dropout_mode", "off")),
        "module_sizes_gene": sizes.tolist(),
        "module_sizes_tx": [int((tx_modules == k).sum()) for k in range(K)],
        "factor_corr": float(params.get("factor_corr", 0.3)),
    }

    return ScenarioResult(
        expr_features_x_samples=expr_gxs,
        expr_samples_x_features=expr_sxg,
        covariates=covariates,
        modules_hard=modules_hard,
        modules_long=modules_long,
        module_latents=module_latents,
        edge_list=edge_list,
        gene_metadata=gene_metadata,
        metadata=metadata,
        modules_overlap=None,
        extra_truth={
            "tx2gene.tsv": tx2gene,
            "modules_gene.csv": modules_gene,
            "switch_table.tsv.gz": switch_table,
        },
        counts_features_x_samples=counts_gxs,
    )


__all__ = ["simulate_isoform_counts"]
