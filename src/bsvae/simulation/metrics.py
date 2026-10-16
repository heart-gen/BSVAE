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

Grey-aware metrics (label -1 / "grey" = unassigned, as in WGCNA)
----------------------------------------------------------------
ari_nmi_all, ari_nmi_assigned, grey_metrics, best_match_metrics, k_metrics,
evaluate_labels, gene_level_metrics.
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


# ---------------------------------------------------------------------------
# Grey-aware metrics
# ---------------------------------------------------------------------------

GREY = -1
_GREY_NAMES = {"grey", "gray", "-1", "unassigned", "none", "nan"}


def normalize_labels(labels) -> np.ndarray:
    """
    Convert arbitrary labels (ints, WGCNA colour strings, "grey") to an int
    array where unassigned is ``GREY`` (-1) and other labels are 0..M-1.
    """
    arr = np.asarray(labels)
    if arr.dtype.kind in "iu":
        out = arr.astype(np.int64)
        out[out < 0] = GREY
        return out
    if arr.dtype.kind == "f":
        out = np.where(np.isnan(arr), GREY, arr).astype(np.int64)
        out[out < 0] = GREY
        return out
    strs = np.array([str(x).strip() for x in arr])
    is_grey = np.array([x.lower() in _GREY_NAMES for x in strs])
    uniq = sorted(set(strs[~is_grey]))
    code = {u: i for i, u in enumerate(uniq)}
    out = np.array([GREY if g else code[x] for x, g in zip(strs, is_grey)], dtype=np.int64)
    return out


def ari_nmi_all(labels_true, labels_pred) -> Dict[str, float]:
    """ARI/NMI over every feature, grey treated as a label of its own."""
    t = normalize_labels(labels_true)
    p = normalize_labels(labels_pred)
    return {"ari_all": compute_ari(t, p), "nmi_all": compute_nmi(t, p)}


def ari_nmi_assigned(labels_true, labels_pred) -> Dict[str, float]:
    """ARI/NMI restricted to features assigned (non-grey) in both truth and prediction."""
    t = normalize_labels(labels_true)
    p = normalize_labels(labels_pred)
    keep = (t != GREY) & (p != GREY)
    n_dropped = int((~keep).sum())
    if keep.sum() < 2:
        return {"ari_assigned": float("nan"), "nmi_assigned": float("nan"), "n_dropped": n_dropped}
    return {
        "ari_assigned": compute_ari(t[keep], p[keep]),
        "nmi_assigned": compute_nmi(t[keep], p[keep]),
        "n_dropped": n_dropped,
    }


def grey_metrics(labels_true, labels_pred) -> Dict[str, float]:
    """Precision/recall of the unassigned class against null (grey) truth."""
    t = normalize_labels(labels_true) == GREY
    p = normalize_labels(labels_pred) == GREY
    tp = int((t & p).sum())
    return {
        "frac_true_grey": float(t.mean()) if len(t) else float("nan"),
        "frac_pred_grey": float(p.mean()) if len(p) else float("nan"),
        "grey_precision": tp / p.sum() if p.sum() else float("nan"),
        "grey_recall": tp / t.sum() if t.sum() else float("nan"),
        "n_pred_grey": int(p.sum()),
    }


COVERAGE_GREY_FRACS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def coverage_metrics(labels_true, labels_pred, score, grey_fracs=COVERAGE_GREY_FRACS) -> Dict[str, float]:
    """
    Assigned-only ARI as a function of coverage: the lowest-``score`` assigned
    features are abstained on until the unassigned fraction reaches each target
    in ``grey_fracs``, and ARI is recomputed on what remains.  ``score`` is a
    per-feature confidence (own-module kME in the harness).  Methods that abstain
    on a fixed subset (acorde) are a single point on this curve, so a method that
    assigns everything can be compared with them at equal coverage; targets below
    the method's own grey fraction are NaN.  Keys ``ari_at_grey_<pct>``.
    """
    t = normalize_labels(labels_true)
    p = normalize_labels(labels_pred)
    s = np.asarray(score, dtype=float)
    assigned = p != GREY
    n = len(p)
    out: Dict[str, float] = {}
    order = np.argsort(np.where(assigned, np.nan_to_num(s, nan=-np.inf), np.inf), kind="stable")
    for g in grey_fracs:
        key = f"ari_at_grey_{int(round(g * 100)):02d}"
        n_target = int(round(g * n))
        n_drop = n_target - int((~assigned).sum())
        if n_drop < 0:
            out[key] = float("nan")
            continue
        keep = assigned.copy()
        keep[order[:n_drop]] = False
        keep &= t != GREY
        out[key] = compute_ari(t[keep], p[keep]) if keep.sum() >= 2 else float("nan")
    return out


def best_match_metrics(labels_true, labels_pred) -> Dict[str, float]:
    """
    Per-true-module best-match Jaccard / precision / recall / F1 (grey excluded),
    plus the fraction of true modules recovered at Jaccard ≥ 0.5 and ≥ 0.25.
    """
    t = normalize_labels(labels_true)
    p = normalize_labels(labels_pred)
    true_sets: Dict[int, set] = {}
    pred_sets: Dict[int, set] = {}
    for i, (a, b) in enumerate(zip(t, p)):
        if a != GREY:
            true_sets.setdefault(int(a), set()).add(i)
        if b != GREY:
            pred_sets.setdefault(int(b), set()).add(i)
    if not true_sets:
        return {k: float("nan") for k in (
            "mean_best_match_jaccard", "mean_best_match_f1", "mean_best_match_precision",
            "mean_best_match_recall", "module_recovery_50", "module_recovery_25")}
    jac, prec, rec, f1s = [], [], [], []
    for ts in true_sets.values():
        bj = bp = br = bf = 0.0
        for ps in pred_sets.values():
            inter = len(ts & ps)
            if inter == 0:
                continue
            j = inter / len(ts | ps)
            if j > bj:
                pr = inter / len(ps); rc = inter / len(ts)
                bj, bp, br, bf = j, pr, rc, 2 * pr * rc / (pr + rc)
        jac.append(bj); prec.append(bp); rec.append(br); f1s.append(bf)
    n = len(jac)
    return {
        "mean_best_match_jaccard": float(np.mean(jac)),
        "mean_best_match_f1": float(np.mean(f1s)),
        "mean_best_match_precision": float(np.mean(prec)),
        "mean_best_match_recall": float(np.mean(rec)),
        "module_recovery_50": float(sum(j >= 0.5 for j in jac) / n),
        "module_recovery_25": float(sum(j >= 0.25 for j in jac) / n),
    }


def k_metrics(labels_true, labels_pred) -> Dict[str, float]:
    t = normalize_labels(labels_true)
    p = normalize_labels(labels_pred)
    k_true = len(set(t[t != GREY]))
    k_pred = len(set(p[p != GREY]))
    sizes = np.bincount(p[p != GREY]) if k_pred else np.array([0])
    sizes = sizes[sizes > 0] if k_pred else sizes
    return {
        "n_modules_true": int(k_true),
        "n_modules_pred": int(k_pred),
        "K_ratio": float(k_pred / k_true) if k_true else float("nan"),
        "K_abs_diff": int(abs(k_pred - k_true)),
        "module_size_min": int(sizes.min()) if k_pred else 0,
        "module_size_median": float(np.median(sizes)) if k_pred else 0.0,
        "module_size_max": int(sizes.max()) if k_pred else 0,
    }


def evaluate_labels(labels_true, labels_pred) -> Dict[str, float]:
    """All grey-aware metrics for one (truth, prediction) pair of aligned label vectors."""
    out: Dict[str, float] = {}
    out.update(ari_nmi_all(labels_true, labels_pred))
    out.update(ari_nmi_assigned(labels_true, labels_pred))
    out.update(grey_metrics(labels_true, labels_pred))
    out.update(best_match_metrics(labels_true, labels_pred))
    out.update(k_metrics(labels_true, labels_pred))
    return out


def gene_level_metrics(
    pred_tx: "Dict[str, int] | Sequence[int]",
    tx_ids: Sequence[str],
    tx2gene: Dict[str, str],
    truth_gene: Dict[str, int],
    truth_tx: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Gene-level recovery from transcript-level predictions.

    * ``gene_ari`` / ``gene_nmi``: majority vote of each gene's non-grey
      isoform labels against the gene truth (ties → smallest label).
    * ``false_split_rate``: among genes with no switched transcript and at
      least two assigned transcripts, the fraction whose transcripts are
      spread over two or more predicted modules (abundance confounding:
      divergence that is not in the truth);
    * ``split_detection_rate``: among genes whose transcript truth places at
      least one isoform in a module different from the gene module
      ("switched" genes), the fraction where the prediction also places that
      isoform in a module different from the majority of its siblings.
    * ``isoform_gene_agreement``: fraction of genes with ≥2 assigned isoforms
      whose assigned isoforms all share one predicted module.
    """
    from collections import Counter, defaultdict

    if isinstance(pred_tx, dict):
        pred_map = {str(k): int(v) for k, v in pred_tx.items()}
    else:
        pred_map = {str(t): int(v) for t, v in zip(tx_ids, pred_tx)}
    by_gene: Dict[str, List[str]] = defaultdict(list)
    for t in tx_ids:
        g = tx2gene.get(str(t))
        if g is not None:
            by_gene[str(g)].append(str(t))

    gene_true, gene_pred = [], []
    agree_n = agree_ok = 0
    split_n = split_ok = 0
    plain_n = plain_split = 0
    for g, txs in by_gene.items():
        if g not in truth_gene:
            continue
        votes = [pred_map[t] for t in txs if pred_map.get(t, GREY) != GREY]
        if votes:
            c = Counter(votes)
            top = max(c.values())
            maj = min(k for k, v in c.items() if v == top)
            gene_true.append(int(truth_gene[g])); gene_pred.append(maj)
            if len(votes) >= 2:
                agree_n += 1
                agree_ok += int(len(c) == 1)
        if truth_tx is not None and len(txs) >= 2:
            tg = int(truth_gene[g])
            switched = [t for t in txs if int(truth_tx.get(t, tg)) != tg]
            if switched:
                split_n += 1
                siblings = [pred_map[t] for t in txs if t not in switched and pred_map.get(t, GREY) != GREY]
                if siblings:
                    sib_maj = Counter(siblings).most_common(1)[0][0]
                    detected = any(pred_map.get(t, GREY) != GREY and pred_map[t] != sib_maj for t in switched)
                else:
                    detected = False
                split_ok += int(detected)
            else:
                assigned = [pred_map[t] for t in txs if pred_map.get(t, GREY) != GREY]
                if len(assigned) >= 2:
                    plain_n += 1
                    plain_split += int(len(set(assigned)) > 1)
    out = {
        "gene_ari": compute_ari(np.array(gene_true), np.array(gene_pred)) if len(gene_true) > 1 else float("nan"),
        "gene_nmi": compute_nmi(np.array(gene_true), np.array(gene_pred)) if len(gene_true) > 1 else float("nan"),
        "n_genes_scored": len(gene_true),
        "isoform_gene_agreement": agree_ok / agree_n if agree_n else float("nan"),
        "n_switched_genes": split_n,
        "split_detection_rate": split_ok / split_n if split_n else float("nan"),
        "n_unswitched_multi_genes": plain_n,
        "false_split_rate": plain_split / plain_n if plain_n else float("nan"),
    }
    return out


def switched_isoform_metrics(
    pred_tx: "Dict[str, int] | Sequence[int]",
    tx_ids: Sequence[str],
    switch_table: "pd.DataFrame",
) -> Dict[str, float]:
    """
    Isoform-specific module recovery from residual-module labels.

    ``pred_tx`` labels transcripts whose *isoform-specific* (gene-programme-
    removed) profile falls in a module; everything else is grey.  Against the
    simulator's switch table this gives

    * ``switched_recall``: fraction of switched transcripts placed in a module,
    * ``switched_precision``: fraction of module members that are switched,
    * ``switched_ari``: ARI between residual-module labels and the true
      transcript module, over switched transcripts placed in a module,
    * ``n_switched_true`` / ``n_called``.
    """
    import pandas as pd

    if isinstance(pred_tx, dict):
        pred = {str(k): int(v) for k, v in pred_tx.items()}
    else:
        pred = {str(t): int(v) for t, v in zip(tx_ids, pred_tx)}
    st = switch_table.set_index(switch_table["transcript_id"].astype(str))
    tx = [t for t in st.index if t in pred]
    is_sw = st.loc[tx, "is_switched"].values.astype(int) == 1
    called = np.array([pred[t] != GREY for t in tx])
    n_sw, n_called = int(is_sw.sum()), int(called.sum())
    tp = int((is_sw & called).sum())
    out = {
        "n_switched_true": n_sw,
        "n_called": n_called,
        "switched_recall": tp / n_sw if n_sw else float("nan"),
        "switched_precision": tp / n_called if n_called else float("nan"),
    }
    both = is_sw & called
    if both.sum() >= 2:
        truth = st.loc[np.array(tx)[both], "tx_module"].values.astype(int)
        labels = np.array([pred[t] for t in np.array(tx)[both]])
        out["switched_ari"] = compute_ari(truth, labels)
    else:
        out["switched_ari"] = float("nan")
    return out
