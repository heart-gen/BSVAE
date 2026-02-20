"""
Evaluation metrics for module discovery benchmarking.

Functions
---------
compute_ari(labels_true, labels_pred)
    Adjusted Rand Index.
compute_nmi(labels_true, labels_pred)
    Normalized Mutual Information.
benchmark_methods(labels_true, predictions_dict)
    Compute ARI + NMI for multiple methods.
compute_module_enrichment_score(modules, annotation_sets, feature_ids)
    Fisher's exact test for GO/pathway enrichment per module.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Adjusted Rand Index between ground-truth and predicted module assignments.

    Parameters
    ----------
    labels_true : array-like, shape (N,)
    labels_pred : array-like, shape (N,)

    Returns
    -------
    ari : float in [-1, 1]; 1.0 = perfect agreement.
    """
    return float(adjusted_rand_score(labels_true, labels_pred))


def compute_nmi(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    average_method: str = "arithmetic",
) -> float:
    """
    Normalized Mutual Information between ground-truth and predicted assignments.

    Parameters
    ----------
    labels_true : array-like, shape (N,)
    labels_pred : array-like, shape (N,)
    average_method : str
        Normalisation method for sklearn NMI. Default: "arithmetic".

    Returns
    -------
    nmi : float in [0, 1]; 1.0 = perfect agreement.
    """
    return float(
        normalized_mutual_info_score(labels_true, labels_pred, average_method=average_method)
    )


def benchmark_methods(
    labels_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute ARI and NMI for multiple prediction methods.

    Parameters
    ----------
    labels_true : array-like, shape (N,)
        Ground-truth module labels.
    predictions : dict[str → array-like]
        Method name → predicted labels.

    Returns
    -------
    results : dict[str → {"ari": float, "nmi": float}]
    """
    results = {}
    for method_name, labels_pred in predictions.items():
        results[method_name] = {
            "ari": compute_ari(labels_true, labels_pred),
            "nmi": compute_nmi(labels_true, labels_pred),
        }
    return results


def compute_module_enrichment_score(
    module_assignments: np.ndarray,
    annotation_sets: Dict[str, List[str]],
    feature_ids: Sequence[str],
    min_overlap: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Fisher's exact test for annotation set enrichment per module.

    Parameters
    ----------
    module_assignments : array-like, shape (N,)
        Predicted module labels.
    annotation_sets : dict[str → list of str]
        Annotation set name → list of annotated feature IDs (e.g., GO terms).
    feature_ids : sequence of str
        Feature IDs corresponding to module_assignments.
    min_overlap : int
        Minimum overlap required to test enrichment. Default: 3.

    Returns
    -------
    results : dict[module_id → dict[annotation_name → {"p_value": float, "odds_ratio": float}]]
    """
    from scipy.stats import fisher_exact

    feature_ids = list(feature_ids)
    fid_set = set(feature_ids)
    id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    modules = np.asarray(module_assignments)
    unique_modules = np.unique(modules)
    N = len(feature_ids)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for k in unique_modules:
        module_mask = modules == k
        module_features = set(f for f, m in zip(feature_ids, module_mask) if m)
        n_module = len(module_features)
        results[str(k)] = {}

        for ann_name, ann_features in annotation_sets.items():
            ann_set = set(ann_features) & fid_set
            overlap = module_features & ann_set
            if len(overlap) < min_overlap:
                continue

            # 2×2 contingency table
            a = len(overlap)              # in module AND annotation
            b = len(ann_set) - a         # in annotation but NOT module
            c = n_module - a             # in module but NOT annotation
            d = N - a - b - c            # neither

            table = [[a, b], [c, d]]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
            results[str(k)][ann_name] = {
                "p_value": float(p_value),
                "odds_ratio": float(odds_ratio),
                "overlap": int(a),
                "annotation_size": len(ann_set),
                "module_size": n_module,
            }

    return results
