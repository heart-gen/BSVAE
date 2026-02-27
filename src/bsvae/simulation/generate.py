"""
Simulation of omics data with known block-diagonal module structure.

simulate_omics_data(...)
    Returns (X, ground_truth, metadata) suitable for benchmarking.

The data-generating process
---------------------------
1. Assign each of n_features to one of n_modules uniformly.
2. For each module k, sample a module "factor" score f_k ∈ R^n_samples
   from N(0, 1).
3. Compute true expression for feature i in module k:
     x_i = f_k * sqrt(rho_within) + epsilon_shared * sqrt(rho_between)
           + epsilon_i * sqrt(noise_var)
   where rho_within + rho_between + noise_var = 1 (variance decomposition).
4. Add noise: X += N(0, noise_std).

This gives a block-diagonal correlation structure:
  Cor(x_i, x_j) = rho_within   if i, j in same module
  Cor(x_i, x_j) = rho_between  otherwise
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def simulate_omics_data(
    n_features: int = 500,
    n_samples: int = 200,
    n_modules: int = 10,
    within_module_correlation: float = 0.8,
    between_module_correlation: float = 0.0,
    noise_std: float = 0.2,
    seed: int = 13,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simulate omics data with block-diagonal covariance structure.

    Parameters
    ----------
    n_features : int
        Total number of features (genes, isoforms, etc.). Default: 500.
    n_samples : int
        Number of samples. Default: 200.
    n_modules : int
        Number of ground-truth modules. Default: 10.
    within_module_correlation : float
        Pearson correlation between features in the same module. Default: 0.8.
    between_module_correlation : float
        Pearson correlation between features in different modules. Default: 0.0.
    noise_std : float
        Standard deviation of independent feature noise. Default: 0.2.
    seed : int
        Random seed. Default: 13.

    Returns
    -------
    X : np.ndarray, shape (n_features, n_samples)
        Simulated expression matrix.
    ground_truth : np.ndarray, shape (n_features,), dtype int
        Module assignment for each feature (0-indexed).
    metadata : dict
        Simulation parameters and summary statistics.
    """
    rng = np.random.default_rng(seed)

    # --- Assign features to modules (as equal-sized as possible) ---
    base_size = n_features // n_modules
    remainder = n_features % n_modules
    sizes = [base_size + (1 if i < remainder else 0) for i in range(n_modules)]
    ground_truth = np.concatenate(
        [np.full(s, k, dtype=int) for k, s in enumerate(sizes)]
    )
    # Shuffle so modules aren't trivially contiguous
    perm = rng.permutation(n_features)
    ground_truth = ground_truth[perm]

    # --- Variance decomposition ---
    # Total variance = 1.0; decompose into within-module, between-module, noise
    rho_w = within_module_correlation
    rho_b = between_module_correlation
    noise_var = max(1.0 - rho_w - rho_b, noise_std ** 2)

    # --- Module factor scores: one per module (F_k ∈ R^n_samples) ---
    module_factors = rng.standard_normal((n_modules, n_samples))  # (K, N)

    # --- Shared cross-module factor (for between-module correlation) ---
    shared_factor = rng.standard_normal((1, n_samples))           # (1, N)

    # --- Assemble X ---
    X = np.zeros((n_features, n_samples), dtype=np.float32)
    for i in range(n_features):
        k = ground_truth[i]
        within_signal = module_factors[k] * np.sqrt(rho_w)
        between_signal = shared_factor[0] * np.sqrt(rho_b)
        noise = rng.standard_normal(n_samples) * np.sqrt(noise_var)
        X[i] = within_signal + between_signal + noise

    # Add extra observation noise
    X += rng.standard_normal((n_features, n_samples)).astype(np.float32) * noise_std

    metadata = dict(
        n_features=n_features,
        n_samples=n_samples,
        n_modules=n_modules,
        within_module_correlation=within_module_correlation,
        between_module_correlation=between_module_correlation,
        noise_std=noise_std,
        seed=seed,
        module_sizes=sizes,
    )

    return X, ground_truth, metadata
