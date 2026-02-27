from __future__ import annotations

import copy
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_CONFIG: Dict = {
    "version": 1,
    "base_seed": 13,
    "reps": 30,
    "generator": {
        "family": "nb_latent_modules",
        "n_features": 6000,
        "n_modules": 20,
        "module_size_distribution": "long_tail",
        "hub_fraction": 0.05,
        "hub_multiplier": 2.0,
        "baseline_log_mean": -2.0,
        "baseline_log_sd": 0.6,
        "libsize_log_sd": 0.5,
        "dispersion": "medium",
        "signal_scale": 0.8,
        "overlap_rate": 0.1,
        "nonlinear_mode": "off",
        "dropout_mode": "off",
        "truth_edge_top_k": 25,
    },
    "grid": {
        "n_samples": [100, 300],
        "signal_scale": [0.4, 0.8, 1.2],
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
            "signal_scale": 0.6,
            "confounding": "moderate",
        },
    ],
    "outputs": {
        "emit_wgcna_matrix": True,
        "emit_gnvae_folds": True,
        "gnvae_n_folds": 10,
    },
}


def _ensure_list(value):
    if isinstance(value, list):
        return value
    return [value]


def load_config(path: str | Path) -> Dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Config parsing failed. `bsvae-simulate` expects JSON-compatible YAML. "
            "Use `bsvae-simulate init-config` to create a starter file."
        ) from exc
    return cfg


def write_starter_config(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")


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


def _build_module_sizes(n_features: int, n_modules: int, mode: str, rng: np.random.Generator) -> np.ndarray:
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
    rng = np.random.default_rng(seed)

    feature_ids = [f"feature_{i:05d}" for i in range(n_features)]
    sample_ids = [f"sample_{j:04d}" for j in range(n_samples)]
    module_names = [f"M{k + 1:02d}" for k in range(n_modules)]

    module_sizes = _build_module_sizes(
        n_features=n_features,
        n_modules=n_modules,
        mode=generator_cfg.get("module_size_distribution", "long_tail"),
        rng=rng,
    )

    primary = np.concatenate([np.full(sz, k, dtype=int) for k, sz in enumerate(module_sizes)])
    rng.shuffle(primary)

    memberships = [set([int(primary[i])]) for i in range(n_features)]
    n_overlap = int(overlap_rate * n_features)
    if n_overlap > 0:
        overlap_idx = rng.choice(n_features, size=n_overlap, replace=False)
        for i in overlap_idx:
            choices = [k for k in range(n_modules) if k != primary[i]]
            memberships[int(i)].add(int(rng.choice(choices)))

    module_members = {k: [i for i, m in enumerate(memberships) if k in m] for k in range(n_modules)}

    hub_frac = float(generator_cfg.get("hub_fraction", 0.05))
    hub_mult = float(generator_cfg.get("hub_multiplier", 2.0))
    hub_by_module = {k: set() for k in range(n_modules)}
    for k in range(n_modules):
        members = module_members[k]
        if not members:
            continue
        n_h = max(1, int(hub_frac * len(members)))
        chosen = rng.choice(members, size=min(n_h, len(members)), replace=False)
        hub_by_module[k] = set(int(i) for i in chosen)

    z_lin = rng.normal(0.0, 1.0, size=(n_samples, n_modules))
    if nonlinear_mode == "on":
        z_nl = np.zeros_like(z_lin)
        z_nl[:, 0::2] = np.tanh(z_lin[:, 0::2])
        z_nl[:, 1::2] = (z_lin[:, 1::2] > 0).astype(float)
        z = z_lin + 0.6 * z_nl
    else:
        z = z_lin

    W = np.zeros((n_features, n_modules), dtype=np.float32)
    for g in range(n_features):
        for k in memberships[g]:
            scale = signal_scale
            if g in hub_by_module[k]:
                scale *= hub_mult
            W[g, k] = rng.normal(0.0, scale)
    W += rng.normal(0.0, 0.02, size=W.shape)

    alpha = rng.normal(
        float(generator_cfg.get("baseline_log_mean", -2.0)),
        float(generator_cfg.get("baseline_log_sd", 0.6)),
        size=n_features,
    )

    condition = rng.integers(0, 2, size=n_samples)
    if confounding == "none":
        batch = np.zeros(n_samples, dtype=int)
        batch_eff = np.array([0.0], dtype=np.float32)
    elif confounding == "moderate":
        batch = rng.integers(0, 3, size=n_samples)
        batch_eff = rng.normal(0.0, 0.4, size=3)
    elif confounding == "strong":
        batch = rng.integers(0, 3, size=n_samples)
        batch_eff = rng.normal(0.0, 0.8, size=3)
    else:
        raise ValueError(f"Unknown confounding level: {confounding}")

    libsize = np.exp(rng.normal(0.0, float(generator_cfg.get("libsize_log_sd", 0.5)), size=n_samples))

    affected_modules = set(rng.choice(np.arange(n_modules), size=max(1, n_modules // 5), replace=False))
    cond_gene_mask = np.array([any(k in affected_modules for k in memberships[g]) for g in range(n_features)])
    beta_cond = np.zeros(n_features, dtype=np.float32)
    beta_cond[cond_gene_mask] = rng.normal(0.4, 0.1, size=cond_gene_mask.sum())

    eta = z @ W.T
    eta = eta + alpha[None, :]
    eta = eta + condition[:, None] * beta_cond[None, :]
    eta = eta + batch_eff[batch][:, None]

    mu = np.exp(np.clip(eta, -10, 10)) * libsize[:, None]

    dispersion_mode = generator_cfg.get("dispersion", "medium")
    if dispersion_mode == "low":
        theta = np.full(n_features, 20.0)
    elif dispersion_mode == "high":
        theta = np.full(n_features, 1.5)
    else:
        theta = np.full(n_features, 5.0)

    p = theta[None, :] / (theta[None, :] + mu)
    counts = rng.negative_binomial(theta[None, :], p).astype(np.float32)

    if dropout_mode in {"logistic", "high"}:
        target = dropout_target if dropout_mode == "high" else 0.2
        log_mu = np.log1p(mu)
        mid = float(np.quantile(log_mu, target))
        probs = 1.0 / (1.0 + np.exp((log_mu - mid) / 0.7))
        mask = rng.random(probs.shape) < probs
        counts[mask] = 0.0

    cpm = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0) * 1e6
    expr = np.log2(cpm + 1.0)

    expr_sxg = pd.DataFrame(expr, index=sample_ids, columns=feature_ids)
    expr_gxs = expr_sxg.T

    covariates = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": condition,
            "batch": batch,
            "libsize": libsize,
        }
    ).set_index("sample_id")

    modules_hard = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "module": primary.astype(int),
        }
    )

    modules_long_rows = []
    for g in range(n_features):
        memberships_g = sorted(list(memberships[g]))
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
    modules_long = pd.DataFrame(modules_long_rows)

    module_latents = pd.DataFrame(z, index=sample_ids, columns=module_names)

    top_k = int(generator_cfg.get("truth_edge_top_k", 25))
    edge_map: Dict[Tuple[int, int], float] = {}
    for i in range(n_features):
        candidates = set()
        for k in memberships[i]:
            candidates.update(module_members[k])
        candidates.discard(i)
        if not candidates:
            continue
        cand_idx = np.fromiter(candidates, dtype=int)
        sims = np.abs(W[cand_idx] @ W[i])
        if cand_idx.size > top_k:
            keep = np.argpartition(sims, -top_k)[-top_k:]
            cand_idx = cand_idx[keep]
            sims = sims[keep]
        for j, w in zip(cand_idx, sims):
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a == b:
                continue
            old = edge_map.get((a, b), -np.inf)
            if float(w) > old:
                edge_map[(a, b)] = float(w)
    edge_rows = [
        {
            "source": feature_ids[i],
            "target": feature_ids[j],
            "weight": w,
            "is_within_module": 1,
        }
        for (i, j), w in edge_map.items()
    ]
    edge_list = pd.DataFrame(edge_rows)

    gene_metadata = pd.DataFrame(
        {
            "feature_id": feature_ids,
            "primary_module": [module_names[k] for k in primary],
            "n_memberships": [len(m) for m in memberships],
            "is_hub_any": [int(any(g in hub_by_module[k] for k in memberships[g])) for g in range(n_features)],
            "alpha": alpha,
            "dispersion": theta,
        }
    )

    metadata = {
        "n_features": n_features,
        "n_samples": n_samples,
        "n_modules": n_modules,
        "seed": seed,
        "signal_scale": signal_scale,
        "overlap_rate": overlap_rate,
        "confounding": confounding,
        "nonlinear_mode": nonlinear_mode,
        "dropout_mode": dropout_mode,
        "dropout_target": dropout_target,
        "module_sizes": module_sizes.tolist(),
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
    )


def _write_gnvae_splits(expr_gxs: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = expr_gxs.shape[0]
    order = rng.permutation(n)

    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1

    current = 0
    for fold_id, fold_size in enumerate(fold_sizes):
        test_idx = order[current : current + fold_size]
        train_idx = np.concatenate([order[:current], order[current + fold_size :]])
        current += fold_size

        fold_dir = output_dir / "gnvae" / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        expr_gxs.iloc[train_idx].to_csv(fold_dir / "X_train.tsv.gz", sep="\t", compression="gzip")
        expr_gxs.iloc[test_idx].to_csv(fold_dir / "X_test.tsv.gz", sep="\t", compression="gzip")


def _scenario_id(base_idx: int, factor_dict: Dict) -> str:
    ordered = [f"{k}-{factor_dict[k]}" for k in sorted(factor_dict)]
    return f"S{base_idx:03d}__" + "__".join(ordered)


def expand_scenarios(config: Dict) -> List[Tuple[str, Dict]]:
    gen = copy.deepcopy(config.get("generator", {}))
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


def generate_scenario(config: Dict, scenario_id: str, rep: int, outdir: str | Path, base_seed: int | None = None) -> Path:
    scenarios = dict(expand_scenarios(config))
    if scenario_id not in scenarios:
        available = ", ".join(sorted(scenarios.keys())[:10])
        raise ValueError(f"Unknown scenario_id: {scenario_id}. Examples: {available}")

    params = copy.deepcopy(scenarios[scenario_id])

    seed0 = int(config.get("base_seed", 13) if base_seed is None else base_seed)
    seed = seed0 + rep

    result = _simulate_counts(
        n_features=int(params.get("n_features", 6000)),
        n_samples=int(params.get("n_samples", 300)),
        n_modules=int(params.get("n_modules", 20)),
        signal_scale=float(params.get("signal_scale", 0.8)),
        overlap_rate=float(params.get("overlap_rate", 0.1)),
        confounding=str(params.get("confounding", "none")),
        nonlinear_mode=str(params.get("nonlinear_mode", "off")),
        dropout_mode=str(params.get("dropout_mode", "off")),
        dropout_target=float(params.get("dropout_target", 0.5)),
        seed=seed,
        generator_cfg=params,
    )

    base = Path(outdir) / "scenarios" / scenario_id / f"rep_{rep:03d}"
    (base / "expr").mkdir(parents=True, exist_ok=True)
    (base / "truth").mkdir(parents=True, exist_ok=True)

    expr_gxs_path = base / "expr" / "features_x_samples.tsv.gz"
    expr_sxg_path = base / "expr" / "samples_x_features.tsv.gz"
    cov_path = base / "covariates.tsv.gz"

    result.expr_features_x_samples.to_csv(expr_gxs_path, sep="\t", compression="gzip")
    result.expr_samples_x_features.to_csv(expr_sxg_path, sep="\t", compression="gzip")
    result.covariates.to_csv(cov_path, sep="\t", compression="gzip")

    modules_hard_path = base / "truth" / "modules_hard.csv"
    modules_long_path = base / "truth" / "modules_long.csv"
    latents_path = base / "truth" / "module_latents.tsv.gz"
    edge_list_path = base / "truth" / "edge_list.tsv.gz"
    gene_meta_path = base / "truth" / "gene_metadata.tsv.gz"

    result.modules_hard.to_csv(modules_hard_path, index=False)
    result.modules_long.to_csv(modules_long_path, index=False)
    result.module_latents.to_csv(latents_path, sep="\t", compression="gzip")
    result.edge_list.to_csv(edge_list_path, sep="\t", index=False, compression="gzip")
    result.gene_metadata.to_csv(gene_meta_path, sep="\t", index=False, compression="gzip")

    outputs_cfg = config.get("outputs", {})
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
    (base / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    method_inputs = {
        "bsvae_dataset": str(expr_gxs_path),
        "bsvae_ground_truth": str(modules_hard_path),
        "wgcna_expr": str(expr_sxg_path),
        "gnvae_gene_expression_filename": str(expr_gxs_path),
    }
    (base / "method_inputs.json").write_text(json.dumps(method_inputs, indent=2), encoding="utf-8")

    return base


def generate_grid(config: Dict, outdir: str | Path, reps: int | None = None, base_seed: int | None = None) -> Dict:
    scenarios = expand_scenarios(config)
    n_reps = int(config.get("reps", 30) if reps is None else reps)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for scenario_id, params in scenarios:
        for rep in range(n_reps):
            run_dir = generate_scenario(
                config=config,
                scenario_id=scenario_id,
                rep=rep,
                outdir=outdir,
                base_seed=base_seed,
            )
            manifest_rows.append(
                {
                    "scenario_id": scenario_id,
                    "rep": rep,
                    "run_dir": str(run_dir),
                    "n_features": int(params.get("n_features", 6000)),
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
            run_dir / "expr" / "samples_x_features.tsv.gz",
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
