"""
Scenario-grid simulation of count-based co-expression data.

Generator families
------------------
``nb_latent_modules``
    Features load on one (or, with overlap, two) latent module factors; NB
    counts with library-size variation, optional gene-specific batch
    confounding, gene-specific nonlinearity, cell-type mixing and dropout.
``nb_isoform_modules``
    Transcript-level data with genes made of several isoforms, gene-level
    modules, isoform switching/mixing, and null (grey) genes.  Implemented in
    :mod:`bsvae.simulation.isoform` on top of the helpers here.

Random streams
--------------
Each scenario seed is split into independent child streams (structure,
factors, loadings, baselines, arm-specific effects, observation noise), so
two grid arms that differ only in, say, ``confounding`` share the same
modules, factors and loadings and differ only in the confounding draw.
"""

from __future__ import annotations

import copy
import json
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# GTEx v11 Whole Blood calibration (see scripts/calibrate_nb_from_gtex.py).
DEFAULT_CONFIG: Dict = {
    "version": 2,
    "base_seed": 13,
    "reps": 30,
    "generator": {
        "family": "nb_latent_modules",
        "n_features": 6000,
        "n_modules": 20,
        "module_size_distribution": "long_tail",
        "null_fraction": 0.0,
        "hub_fraction": 0.05,
        "hub_multiplier": 2.0,
        "baseline_log_mean": 2.638,
        "baseline_log_sd": 1.601,
        "libsize_log_sd": 0.364,
        "dispersion": "heterogeneous",
        "signal_scale": 1.9,
        "factor_corr": 0.0,
        "overlap_rate": 0.1,
        "nonlinear_mode": "off",
        "dropout_mode": "off",
        "confounding": "none",
        "confound_feature_frac": 0.3,
        "condition_effect": True,
        "celltype_confound": "none",
        "normalization": "median_ratio",
        "truth_edge_top_k": 25,
    },
    "grid": {
        "n_samples": [100, 300],
        "signal_scale": [1.0, 1.9, 3.0],
        "overlap_rate": [0.0, 0.2],
        "confounding": ["none", "moderate"],
        "nonlinear_mode": ["off", "on"],
    },
    "stress": [
        {
            "name": "high_dropout",
            "dropout_mode": "high",
            "dropout_target": 0.5,
        },
        {
            "name": "high_overlap_nonlinear",
            "overlap_rate": 0.35,
            "nonlinear_mode": "on",
            "signal_scale": 1.2,
            "confounding": "moderate",
        },
    ],
    "outputs": {
        "emit_wgcna_matrix": True,
        "emit_gnvae_folds": True,
        "gnvae_n_folds": 10,
    },
}

ISOFORM_GENERATOR_DEFAULTS: Dict = {
    "family": "nb_isoform_modules",
    "n_genes": 4000,
    "isoforms_per_gene": {"dist": "geometric", "p": 0.45, "min": 1, "max": 6},
    "n_modules": 16,
    "module_size_distribution": "long_tail",
    "switch_rate": 0.15,
    "alpha": 1.0,
    "null_fraction": 0.10,
    "iso_offset_sd": 0.3,
    "dominant_isoform_frac": 0.6,
    "factor_corr": 0.3,
    "id_version": False,
}


def _ensure_list(value):
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: str | Path) -> Dict:
    """Load a YAML (PyYAML, if installed) or JSON scenario config."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ValueError(
            "Config parsing failed: the file is not JSON and PyYAML is not installed. "
            "Install pyyaml or use `bsvae-simulate init-config` to create a JSON starter file."
        ) from exc
    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return cfg


def write_starter_config(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


ISOFORM_STARTER_CONFIG: Dict = {
    "version": 2,
    "base_seed": 13,
    "reps": 5,
    "generator": {
        **{k: v for k, v in DEFAULT_CONFIG["generator"].items()
           if k not in ("family", "n_features", "n_modules", "overlap_rate", "null_fraction")},
        "family": "nb_isoform_modules",
        "n_genes": 4000,
        "n_modules": 16,
        "isoforms_per_gene": {"dist": "geometric", "p": 0.45, "min": 1, "max": 6},
        "switch_rate": 0.15,
        "alpha": 1.0,
        "null_fraction": 0.10,
        "iso_offset_sd": 0.3,
        "dominant_isoform_frac": 0.6,
        "factor_corr": 0.3,
    },
    "grid": {
        "n_samples": [100, 300],
        "switch_rate": [0.0, 0.15, 0.3],
        "alpha": [1.0, 0.5],
    },
    "stress": [],
    "outputs": {"emit_wgcna_matrix": True, "emit_gnvae_folds": False, "gnvae_n_folds": 10},
}


def write_isoform_starter_config(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ISOFORM_STARTER_CONFIG, indent=2), encoding="utf-8")


@dataclass
class ScenarioResult:
    expr_features_x_samples: pd.DataFrame
    expr_samples_x_features: pd.DataFrame
    covariates: pd.DataFrame
    modules_hard: pd.DataFrame
    modules_long: pd.DataFrame
    module_latents: pd.DataFrame
    edge_list: pd.DataFrame
    gene_metadata: pd.DataFrame
    metadata: Dict
    modules_overlap: Optional[pd.DataFrame] = None
    extra_truth: Dict[str, pd.DataFrame] = field(default_factory=dict)
    counts_features_x_samples: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

STREAM_NAMES = ("struct", "factors", "loadings", "baseline", "arm", "noise")


def spawn_streams(seed: int) -> Dict[str, np.random.Generator]:
    """Independent child generators so grid arms are paired at a seed."""
    children = np.random.default_rng(int(seed)).spawn(len(STREAM_NAMES))
    return dict(zip(STREAM_NAMES, children))


def _build_module_sizes(n_features: int, n_modules: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    if n_modules <= 0:
        return np.zeros(0, dtype=int)
    if mode == "balanced":
        base = n_features // n_modules
        rem = n_features % n_modules
        return np.array([base + (1 if i < rem else 0) for i in range(n_modules)], dtype=int)
    if mode == "long_tail":
        frac = rng.dirichlet(np.ones(n_modules) * 0.4)
        sizes = np.floor(frac * n_features).astype(int)
        sizes[sizes == 0] = 1
        diff = n_features - int(sizes.sum())
        if diff > 0:
            sizes[:diff] += 1
        elif diff < 0:
            idx = np.argsort(sizes)[::-1]
            for i in idx:
                if diff == 0:
                    break
                if sizes[i] > 1:
                    sizes[i] -= 1
                    diff += 1
        return sizes
    raise ValueError(f"Unknown module_size_distribution: {mode}")


def _latent_factors(rng: np.random.Generator, n_samples: int, n_modules: int, factor_corr: float) -> np.ndarray:
    """Module factors (S × K); with ``factor_corr`` > 0, modules in the same
    meta-group share a common component (TF programmes co-activating modules)."""
    z_indep = rng.normal(0.0, 1.0, size=(n_samples, n_modules))
    if factor_corr > 0.0 and n_modules >= 2:
        n_meta = max(2, n_modules // 4)
        meta_assign = rng.integers(0, n_meta, size=n_modules)
        z_meta = rng.normal(0.0, 1.0, size=(n_samples, n_meta))
        return np.sqrt(1.0 - factor_corr) * z_indep + np.sqrt(factor_corr) * z_meta[:, meta_assign]
    return z_indep


_NONLINEAR_FUNCS = {
    0: lambda z: z,                                  # identity
    1: lambda z: 1.6 * np.tanh(z),                   # saturating
    2: lambda z: np.maximum(z, 0.0) * 1.7 - 0.68,    # ReLU-like (centred)
    3: lambda z: (z > 0).astype(float) * 2.0 - 1.0,  # step
    4: lambda z: (z ** 2 - 1.0) / np.sqrt(2.0),      # centred square (non-monotone)
}


def _signal_eta(
    z: np.ndarray,
    W: np.ndarray,
    nonlinear_mode: str,
    rng_arm: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Signal part of the log-mean, S × F.

    ``nonlinear_mode="on"`` applies a feature-specific link f_g to the factors
    before the loading, so features of one module relate to each other
    nonlinearly.  ``"factor"`` reproduces the old behaviour of transforming
    the factor itself (every member stays linear in the same transformed
    factor).  ``"off"`` is linear.
    """
    if nonlinear_mode == "on":
        n_features = W.shape[0]
        func_ids = rng_arm.integers(0, len(_NONLINEAR_FUNCS), size=n_features)
        eta = np.zeros((z.shape[0], n_features), dtype=np.float64)
        for fid, f in _NONLINEAR_FUNCS.items():
            cols = np.where(func_ids == fid)[0]
            if cols.size == 0:
                continue
            eta[:, cols] = f(z) @ W[cols].T
        return eta, func_ids
    if nonlinear_mode == "factor":
        z_nl = np.zeros_like(z)
        z_nl[:, 0::2] = np.tanh(z[:, 0::2])
        z_nl[:, 1::2] = (z[:, 1::2] > 0).astype(float)
        return (z + 0.6 * z_nl) @ W.T, None
    if nonlinear_mode in ("off", "linear", None):
        return z @ W.T, None
    raise ValueError(f"Unknown nonlinear_mode: {nonlinear_mode}")


def _nb_observe(
    eta_signal: np.ndarray,
    alpha: np.ndarray,
    *,
    generator_cfg: Dict,
    confounding: str,
    dropout_mode: str,
    dropout_target: float,
    cond_feature_mask: Optional[np.ndarray],
    rng_arm: np.random.Generator,
    rng_noise: np.random.Generator,
) -> Dict:
    """
    Turn a signal log-mean (S × F) and per-feature baselines into log2-CPM.

    Adds (from the arm stream) gene-specific batch effects, a condition
    effect, cell-type composition effects and dropout; samples NB counts
    (noise stream); then CPM-normalises and log2-transforms.
    """
    n_samples, n_features = eta_signal.shape
    eta = eta_signal + alpha[None, :]

    condition = rng_arm.integers(0, 2, size=n_samples)
    if confounding == "none":
        batch = np.zeros(n_samples, dtype=int)
        batch_sd = 0.0
    elif confounding in ("moderate", "strong"):
        batch = rng_arm.integers(0, 3, size=n_samples)
        batch_sd = 0.4 if confounding == "moderate" else 0.8
    else:
        raise ValueError(f"Unknown confounding level: {confounding}")

    # Gene-specific batch effects: a random subset of features each gets its
    # own effect per batch, so the structure survives CPM normalisation (a
    # per-sample scalar shared by all features would be removed by it).
    frac = float(generator_cfg.get("confound_feature_frac", 0.3))
    n_aff = int(round(frac * n_features)) if confounding != "none" else 0
    affected = rng_arm.choice(n_features, size=n_aff, replace=False) if n_aff > 0 else np.zeros(0, dtype=int)
    if n_aff > 0:
        n_batches = int(batch.max()) + 1
        eff = rng_arm.normal(0.0, batch_sd, size=(n_batches, n_aff))
        eta[:, affected] += eff[batch]

    beta_cond = np.zeros(n_features, dtype=np.float64)
    if bool(generator_cfg.get("condition_effect", True)) and cond_feature_mask is not None:
        n_c = int(cond_feature_mask.sum())
        if n_c > 0:
            beta_cond[cond_feature_mask] = rng_arm.normal(0.4, 0.1, size=n_c)
            eta = eta + condition[:, None] * beta_cond[None, :]

    ct_fractions = None
    celltype_confound = generator_cfg.get("celltype_confound", "none")
    if celltype_confound != "none":
        n_ct = int(generator_cfg.get("n_celltypes", 3))
        ct_effect_sd = 0.4 if celltype_confound == "moderate" else 0.8
        ct_conc = 2.0 if celltype_confound == "moderate" else 1.0
        n_ct_markers = max(10, int(0.15 * n_features))
        ct_fractions = rng_arm.dirichlet(np.ones(n_ct) * ct_conc, size=n_samples)
        ct_signatures = np.zeros((n_ct, n_features), dtype=np.float64)
        for c in range(n_ct):
            markers = rng_arm.choice(n_features, size=min(n_ct_markers, n_features), replace=False)
            ct_signatures[c, markers] = rng_arm.normal(0.0, ct_effect_sd, size=markers.size)
        eta = eta + ct_fractions @ ct_signatures

    libsize = np.exp(rng_noise.normal(0.0, float(generator_cfg.get("libsize_log_sd", 0.364)), size=n_samples))
    mu = np.exp(np.clip(eta, -10, 10)) * libsize[:, None]

    dispersion_mode = generator_cfg.get("dispersion", "heterogeneous")
    if dispersion_mode == "heterogeneous":
        log_theta = -0.5 + 0.3 * alpha + rng_noise.normal(0.0, 0.3, size=n_features)
        theta = np.exp(log_theta).clip(0.3, 20.0)
    elif dispersion_mode == "low":
        theta = np.full(n_features, 20.0)
    elif dispersion_mode == "high":
        theta = np.full(n_features, 1.5)
    else:  # "medium"
        theta = np.full(n_features, 5.0)

    p = theta[None, :] / (theta[None, :] + mu)
    counts = rng_noise.negative_binomial(theta[None, :], p).astype(np.float32)

    if dropout_mode in {"logistic", "high"}:
        target = dropout_target if dropout_mode == "high" else 0.2
        log_mu = np.log1p(mu)
        mid = float(np.quantile(log_mu, target))
        probs = 1.0 / (1.0 + np.exp((log_mu - mid) / 0.7))
        mask = rng_arm.random(probs.shape) < probs
        counts[mask] = 0.0

    expr = normalize_counts(counts, method=str(generator_cfg.get("normalization", "median_ratio")))

    covariates = pd.DataFrame(
        {"condition": condition, "batch": batch, "libsize": libsize}
    )
    return {
        "expr": expr,
        "counts": counts,
        "covariates": covariates,
        "theta": theta,
        "libsize": libsize,
        "batch_affected": affected,
        "beta_cond": beta_cond,
        "ct_fractions": ct_fractions,
    }


def normalize_counts(counts: np.ndarray, method: str = "median_ratio") -> np.ndarray:
    """
    Library-size normalise a samples × features count matrix and return
    log2(normalised counts per million + 1).

    ``"cpm"`` divides by the total count, which makes every feature without
    module signal co-vary against the large modules (a compositional
    artefact).  ``"median_ratio"`` uses DESeq-style size factors (median over
    features of the ratio to the geometric-mean reference, computed on
    features expressed in every sample), which is robust to a few strongly
    varying modules and is what real pipelines use.
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.maximum(counts.sum(axis=1), 1.0)
    if method == "cpm":
        size = totals / 1e6
    elif method == "median_ratio":
        expressed = (counts > 0).all(axis=0)
        if expressed.sum() >= 10:
            logc = np.log(counts[:, expressed])
            ref = logc.mean(axis=0)
            size = np.exp(np.median(logc - ref[None, :], axis=1))
        else:  # too few universally expressed features: fall back to totals
            size = totals / np.exp(np.mean(np.log(totals)))
        # Rescale so values are on a CPM-like scale (same median library).
        size = size / np.exp(np.mean(np.log(size))) * (np.exp(np.mean(np.log(totals))) / 1e6)
    else:
        raise ValueError(f"Unknown normalization: {method}")
    return np.log2(counts / size[:, None] + 1.0)


def _edge_truth(W: np.ndarray, memberships: List[set], module_members: Dict[int, List[int]], top_k: int) -> Dict[Tuple[int, int], float]:
    """Top-k within-module edges per feature by |W_i · W_j|, vectorised per module."""
    edge_map: Dict[Tuple[int, int], float] = {}
    if top_k <= 0:
        return edge_map
    for k, members in module_members.items():
        if len(members) < 2:
            continue
        idx = np.asarray(members)
        Wk = W[idx]
        sims = np.abs(Wk @ Wk.T)
        np.fill_diagonal(sims, -np.inf)
        kk = min(top_k, len(idx) - 1)
        top = np.argpartition(sims, -kk, axis=1)[:, -kk:]
        for r in range(len(idx)):
            i = int(idx[r])
            for c in top[r]:
                j = int(idx[c])
                w = float(sims[r, c])
                a, b = (i, j) if i < j else (j, i)
                if w > edge_map.get((a, b), -np.inf):
                    edge_map[(a, b)] = w
    return edge_map


# ---------------------------------------------------------------------------
# nb_latent_modules family
# ---------------------------------------------------------------------------

def _simulate_counts(
    n_features: int,
    n_samples: int,
    n_modules: int,
    signal_scale: float,
    overlap_rate: float,
    confounding: str,
    nonlinear_mode: str,
    dropout_mode: str,
    dropout_target: float,
    seed: int,
    generator_cfg: Dict,
) -> ScenarioResult:
    streams = spawn_streams(seed)
    rs, rf, rl, rb, ra, rn = (streams[n] for n in STREAM_NAMES)

    feature_ids = [f"feature_{i:05d}" for i in range(n_features)]
    sample_ids = [f"sample_{j:04d}" for j in range(n_samples)]
    module_names = [f"M{k + 1:02d}" for k in range(n_modules)]

    # Null (background) features carry no module loading: truth label -1.
    null_fraction = float(generator_cfg.get("null_fraction", 0.0))
    n_null = int(round(null_fraction * n_features))
    is_null = np.zeros(n_features, dtype=bool)
    if n_null > 0:
        is_null[rs.choice(n_features, size=n_null, replace=False)] = True
    signal_idx = np.where(~is_null)[0]

    module_sizes = _build_module_sizes(
        n_features=len(signal_idx), n_modules=n_modules,
        mode=generator_cfg.get("module_size_distribution", "long_tail"), rng=rs,
    )
    primary = np.full(n_features, -1, dtype=int)
    assigned = np.concatenate([np.full(sz, k, dtype=int) for k, sz in enumerate(module_sizes)])
    rs.shuffle(assigned)
    primary[signal_idx] = assigned

    memberships = [set([int(primary[i])]) if primary[i] >= 0 else set() for i in range(n_features)]
    n_overlap = int(overlap_rate * len(signal_idx))
    if n_overlap > 0 and n_modules > 1:
        overlap_idx = rs.choice(signal_idx, size=n_overlap, replace=False)
        for i in overlap_idx:
            choices = [k for k in range(n_modules) if k != primary[i]]
            memberships[int(i)].add(int(rs.choice(choices)))

    module_members = {k: [i for i, m in enumerate(memberships) if k in m] for k in range(n_modules)}

    hub_frac = float(generator_cfg.get("hub_fraction", 0.05))
    hub_mult = float(generator_cfg.get("hub_multiplier", 2.0))
    hub_by_module = {k: set() for k in range(n_modules)}
    for k in range(n_modules):
        members = module_members[k]
        if not members:
            continue
        n_h = max(1, int(hub_frac * len(members)))
        chosen = rs.choice(members, size=min(n_h, len(members)), replace=False)
        hub_by_module[k] = set(int(i) for i in chosen)

    z = _latent_factors(rf, n_samples, n_modules, float(generator_cfg.get("factor_corr", 0.0)))

    W = np.zeros((n_features, n_modules), dtype=np.float64)
    for g in range(n_features):
        for k in memberships[g]:
            scale = signal_scale * (hub_mult if g in hub_by_module[k] else 1.0)
            W[g, k] = abs(rl.normal(0.0, scale))
    W += rl.normal(0.0, 0.02, size=W.shape)

    alpha = rb.normal(
        float(generator_cfg.get("baseline_log_mean", 2.638)),
        float(generator_cfg.get("baseline_log_sd", 1.601)),
        size=n_features,
    )

    eta_signal, func_ids = _signal_eta(z, W, nonlinear_mode, ra)

    affected_modules = set(rs.choice(np.arange(n_modules), size=max(1, n_modules // 5), replace=False)) if n_modules else set()
    cond_mask = np.array([any(k in affected_modules for k in memberships[g]) for g in range(n_features)])

    obs = _nb_observe(
        eta_signal, alpha, generator_cfg=generator_cfg, confounding=confounding,
        dropout_mode=dropout_mode, dropout_target=dropout_target,
        cond_feature_mask=cond_mask, rng_arm=ra, rng_noise=rn,
    )
    expr_sxg = pd.DataFrame(obs["expr"], index=sample_ids, columns=feature_ids)
    expr_gxs = expr_sxg.T
    counts_gxs = pd.DataFrame(obs["counts"].T, index=feature_ids, columns=sample_ids)
    covariates = obs["covariates"]
    covariates.index = pd.Index(sample_ids, name="sample_id")

    modules_hard = pd.DataFrame(
        {"feature_id": feature_ids, "module": primary.astype(int), "is_null": is_null.astype(int)}
    )
    modules_overlap = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "modules": [";".join(module_names[k] for k in sorted(m)) if m else "grey" for m in memberships],
            "n_memberships": [len(m) for m in memberships],
        }
    )

    modules_long_rows = []
    for g in range(n_features):
        memberships_g = sorted(memberships[g])
        denom = max(1, len(memberships_g))
        for k in memberships_g:
            modules_long_rows.append(
                {
                    "feature_id": feature_ids[g],
                    "module": module_names[k],
                    "is_hub": int(g in hub_by_module[k]),
                    "membership_weight": float(abs(W[g, k]) / denom),
                }
            )
    modules_long = pd.DataFrame(modules_long_rows, columns=["feature_id", "module", "is_hub", "membership_weight"])
    module_latents = pd.DataFrame(z, index=sample_ids, columns=module_names)

    edge_map = _edge_truth(W, memberships, module_members, int(generator_cfg.get("truth_edge_top_k", 25)))
    edge_list = pd.DataFrame(
        [{"source": feature_ids[i], "target": feature_ids[j], "weight": w, "is_within_module": 1}
         for (i, j), w in edge_map.items()],
        columns=["source", "target", "weight", "is_within_module"],
    )

    gene_metadata = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "primary_module": [module_names[k] if k >= 0 else "grey" for k in primary],
            "n_memberships": [len(m) for m in memberships],
            "is_hub_any": [int(any(g in hub_by_module[k] for k in memberships[g])) for g in range(n_features)],
            "is_null": is_null.astype(int),
            "alpha": alpha,
            "dispersion": obs["theta"],
            "batch_affected": np.isin(np.arange(n_features), obs["batch_affected"]).astype(int),
            "nonlinear_func": func_ids if func_ids is not None else 0,
        }
    )

    metadata = {
        "family": "nb_latent_modules",
        "n_features": n_features,
        "n_samples": n_samples,
        "n_modules": n_modules,
        "seed": seed,
        "signal_scale": signal_scale,
        "overlap_rate": overlap_rate,
        "null_fraction": null_fraction,
        "n_null": int(n_null),
        "confounding": confounding,
        "nonlinear_mode": nonlinear_mode,
        "dropout_mode": dropout_mode,
        "dropout_target": dropout_target,
        "module_sizes": module_sizes.tolist(),
        "factor_corr": float(generator_cfg.get("factor_corr", 0.0)),
        "celltype_confound": generator_cfg.get("celltype_confound", "none"),
        "dispersion_mode": generator_cfg.get("dispersion", "heterogeneous"),
        "n_batch_affected": int(len(obs["batch_affected"])),
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
        modules_overlap=modules_overlap,
        counts_features_x_samples=counts_gxs,
    )


# ---------------------------------------------------------------------------
# Grid handling
# ---------------------------------------------------------------------------

def _write_gnvae_splits(expr_gxs: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> None:
    """Sample-level (column) folds for sample-wise VAEs such as GNVAE."""
    rng = np.random.default_rng(seed)
    n = expr_gxs.shape[1]
    order = rng.permutation(n)

    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1

    current = 0
    for fold_id, fold_size in enumerate(fold_sizes):
        test_idx = order[current: current + fold_size]
        train_idx = np.concatenate([order[:current], order[current + fold_size:]])
        current += fold_size

        fold_dir = output_dir / "gnvae" / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        expr_gxs.iloc[:, train_idx].to_csv(fold_dir / "X_train.tsv.gz", sep="\t", compression="gzip")
        expr_gxs.iloc[:, test_idx].to_csv(fold_dir / "X_test.tsv.gz", sep="\t", compression="gzip")


def _scenario_id(base_idx: int, factor_dict: Dict) -> str:
    ordered = [f"{k}-{factor_dict[k]}" for k in sorted(factor_dict)]
    return f"S{base_idx:03d}__" + "__".join(ordered)


def expand_scenarios(config: Dict) -> List[Tuple[str, Dict]]:
    gen = copy.deepcopy(config.get("generator", {}))
    if gen.get("family") == "nb_isoform_modules":
        merged = copy.deepcopy(ISOFORM_GENERATOR_DEFAULTS)
        merged.update(gen)
        gen = merged
    grid = config.get("grid", {})

    keys = sorted(grid.keys())
    values = [_ensure_list(grid[k]) for k in keys]

    scenarios: List[Tuple[str, Dict]] = []
    idx = 1
    for combo in itertools.product(*values):
        factors = dict(zip(keys, combo))
        params = copy.deepcopy(gen)
        params.update(factors)
        sid = _scenario_id(idx, factors)
        scenarios.append((sid, params))
        idx += 1

    for stress in config.get("stress", []):
        params = copy.deepcopy(gen)
        params.update(stress)
        stress_name = stress.get("name", f"stress_{idx:03d}")
        sid = f"S{idx:03d}__{stress_name}"
        scenarios.append((sid, params))
        idx += 1

    return scenarios


def simulate_params(params: Dict, seed: int) -> ScenarioResult:
    """Dispatch on ``params["family"]`` and run the generator."""
    family = str(params.get("family", "nb_latent_modules"))
    if family == "nb_latent_modules":
        return _simulate_counts(
            n_features=int(params.get("n_features", 6000)),
            n_samples=int(params.get("n_samples", 300)),
            n_modules=int(params.get("n_modules", 20)),
            signal_scale=float(params.get("signal_scale", 1.9)),
            overlap_rate=float(params.get("overlap_rate", 0.1)),
            confounding=str(params.get("confounding", "none")),
            nonlinear_mode=str(params.get("nonlinear_mode", "off")),
            dropout_mode=str(params.get("dropout_mode", "off")),
            dropout_target=float(params.get("dropout_target", 0.5)),
            seed=seed,
            generator_cfg=params,
        )
    if family == "nb_isoform_modules":
        from bsvae.simulation.isoform import simulate_isoform_counts
        return simulate_isoform_counts(params=params, seed=seed)
    raise ValueError(f"Unknown generator family: {family}")


def generate_scenario(config: Dict, scenario_id: str, rep: int, outdir: str | Path, base_seed: int | None = None) -> Path:
    scenarios = dict(expand_scenarios(config))
    if scenario_id not in scenarios:
        available = ", ".join(sorted(scenarios.keys())[:10])
        raise ValueError(f"Unknown scenario_id: {scenario_id}. Examples: {available}")

    params = copy.deepcopy(scenarios[scenario_id])
    seed0 = int(config.get("base_seed", 13) if base_seed is None else base_seed)
    seed = seed0 + rep

    result = simulate_params(params, seed)

    base = Path(outdir) / "scenarios" / scenario_id / f"rep_{rep:03d}"
    (base / "expr").mkdir(parents=True, exist_ok=True)
    (base / "truth").mkdir(parents=True, exist_ok=True)

    outputs_cfg = config.get("outputs", {})
    expr_gxs_path = base / "expr" / "features_x_samples.tsv.gz"
    expr_sxg_path = base / "expr" / "samples_x_features.tsv.gz"
    cov_path = base / "covariates.tsv.gz"

    result.expr_features_x_samples.to_csv(expr_gxs_path, sep="\t", compression="gzip")
    if result.counts_features_x_samples is not None and bool(outputs_cfg.get("emit_counts", True)):
        result.counts_features_x_samples.to_csv(base / "expr" / "counts_features_x_samples.tsv.gz",
                                                sep="\t", compression="gzip")
    emit_wgcna = bool(outputs_cfg.get("emit_wgcna_matrix", True))
    if emit_wgcna:
        result.expr_samples_x_features.to_csv(expr_sxg_path, sep="\t", compression="gzip")
    result.covariates.to_csv(cov_path, sep="\t", compression="gzip")

    modules_hard_path = base / "truth" / "modules_hard.csv"
    result.modules_hard.to_csv(modules_hard_path, index=False)
    result.modules_long.to_csv(base / "truth" / "modules_long.csv", index=False)
    result.module_latents.to_csv(base / "truth" / "module_latents.tsv.gz", sep="\t", compression="gzip")
    result.edge_list.to_csv(base / "truth" / "edge_list.tsv.gz", sep="\t", index=False, compression="gzip")
    result.gene_metadata.to_csv(base / "truth" / "gene_metadata.tsv.gz", sep="\t", index=False, compression="gzip")
    if result.modules_overlap is not None:
        result.modules_overlap.to_csv(base / "truth" / "modules_overlap.csv", index=False)
    extra_paths = {}
    for name, df in result.extra_truth.items():
        sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
        p = base / "truth" / name
        df.to_csv(p, sep=sep, index=False, compression="gzip" if name.endswith(".gz") else None)
        extra_paths[name] = str(p)

    if bool(outputs_cfg.get("emit_gnvae_folds", True)):
        _write_gnvae_splits(
            expr_gxs=result.expr_features_x_samples,
            output_dir=base,
            n_folds=int(outputs_cfg.get("gnvae_n_folds", 10)),
            seed=seed,
        )

    metadata = {
        "config_version": config.get("version", 1),
        "scenario_id": scenario_id,
        "rep": rep,
        "seed": seed,
        "parameters": params,
        "summary": result.metadata,
    }
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

    method_inputs = {
        "bsvae_dataset": str(expr_gxs_path),
        "bsvae_ground_truth": str(modules_hard_path),
        "gnvae_gene_expression_filename": str(expr_gxs_path),
    }
    if emit_wgcna:
        method_inputs["wgcna_expr"] = str(expr_sxg_path)
    if "tx2gene.tsv" in extra_paths:
        method_inputs["bsvae_tx2gene"] = extra_paths["tx2gene.tsv"]
    if "modules_gene.csv" in extra_paths:
        method_inputs["bsvae_ground_truth_gene"] = extra_paths["modules_gene.csv"]
    (base / "method_inputs.json").write_text(json.dumps(method_inputs, indent=2), encoding="utf-8")

    return base


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def generate_grid(config: Dict, outdir: str | Path, reps: int | None = None, base_seed: int | None = None) -> Dict:
    scenarios = expand_scenarios(config)
    n_reps = int(config.get("reps", 30) if reps is None else reps)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for scenario_id, params in scenarios:
        for rep in range(n_reps):
            run_dir = generate_scenario(
                config=config, scenario_id=scenario_id, rep=rep, outdir=outdir, base_seed=base_seed,
            )
            manifest_rows.append(
                {
                    "scenario_id": scenario_id,
                    "rep": rep,
                    "run_dir": str(run_dir),
                    "family": params.get("family", "nb_latent_modules"),
                    "n_features": int(params.get("n_features", params.get("n_genes", 6000))),
                    "n_samples": int(params.get("n_samples", 300)),
                    "n_modules": int(params.get("n_modules", 20)),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)

    summary = {
        "n_scenarios": len(scenarios),
        "n_reps": n_reps,
        "n_runs": len(manifest_rows),
        "manifest": str(manifest_path),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def validate_grid(grid_dir: str | Path) -> Dict:
    grid = Path(grid_dir)
    manifest_path = grid / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path, sep="\t")
    checks = []
    for _, row in manifest.head(50).iterrows():
        run_dir = Path(row["run_dir"])
        required = [
            run_dir / "expr" / "features_x_samples.tsv.gz",
            run_dir / "covariates.tsv.gz",
            run_dir / "truth" / "modules_hard.csv",
            run_dir / "method_inputs.json",
        ]
        checks.append(all(p.exists() for p in required))

    ok = bool(all(checks)) if checks else True
    result = {
        "manifest_rows": int(manifest.shape[0]),
        "checked_rows": int(min(50, manifest.shape[0])),
        "all_required_files_present": ok,
    }
    (grid / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
