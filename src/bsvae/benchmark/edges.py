"""
Edge-level evaluation for network methods whose output is an edge list
rather than module labels (TWN), and edge lists derived from module labels
for comparison.

Truth comes from the simulator: gene modules (``modules_gene.csv``) and the
switch table (``switch_table.tsv.gz``).  Three metric groups:

* ``tt``: gene–gene (total expression) edges.  Precision = fraction of
  predicted edges joining two genes of the same true module; recall is
  measured against the top-degree truth, the within-module pairs listed in
  ``truth/edge_list.tsv.gz`` aggregated to gene pairs.
* ``ti``: gene–isoform-ratio edges for isoforms with transcript-specific
  regulation (switched or usage-regulated).  An edge is correct when the
  gene belongs to the isoform's target module; recall is the fraction of
  such isoforms with at least one correct edge.
* ``ti_null``: the same for unswitched isoforms, where any cross-module
  gene edge is a false positive (abundance confounding at the edge level).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def edges_from_labels(
    expr: pd.DataFrame,
    labels: pd.Series,
    tx2gene: Dict[str, str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Gene-level edge list from transcript module labels, for comparison with
    TWN: genes inherit the majority module of their assigned transcripts, and
    within each module every gene is joined to its ``top_k`` most correlated
    module members (gene profiles are the linear-scale isoform sums).
    """
    ids = expr.index.astype(str)
    genes = pd.Series([tx2gene.get(t) for t in ids], index=ids)
    lab = labels.reindex(ids).fillna(-1).astype(int)
    lin = np.maximum(np.power(2.0, expr.values) - 1.0, 0.0)
    keep = genes.notna().values
    gx = pd.DataFrame(lin[keep], index=genes.values[keep], columns=expr.columns).groupby(level=0).sum()
    gx = np.log2(gx + 1.0)
    gene_mod: Dict[str, int] = {}
    for g, sub in lab[keep].groupby(genes.values[keep]):
        v = sub[sub >= 0]
        gene_mod[g] = int(v.value_counts().idxmax()) if len(v) else -1
    rows = []
    for m in sorted({v for v in gene_mod.values() if v >= 0}):
        members = [g for g, v in gene_mod.items() if v == m]
        if len(members) < 2:
            continue
        X = gx.reindex(members).values
        Xc = X - X.mean(axis=1, keepdims=True)
        sd = np.sqrt((Xc ** 2).sum(axis=1)); sd[sd == 0] = 1.0
        C = (Xc / sd[:, None]) @ (Xc / sd[:, None]).T
        np.fill_diagonal(C, -np.inf)
        k = min(int(top_k), len(members) - 1)
        for i, g in enumerate(members):
            for j in np.argpartition(-C[i], k - 1)[:k]:
                a, b = sorted((g, members[j]))
                rows.append((a, b, 1, float(C[i, j])))
    out = pd.DataFrame(rows, columns=["node1", "node2", "edge_type", "weight"]).drop_duplicates(["node1", "node2"])
    return out


def edge_metrics(
    edges: pd.DataFrame,
    truth_gene: Dict[str, int],
    switch_table: Optional[pd.DataFrame] = None,
    truth_edges: Optional[pd.DataFrame] = None,
    tx2gene: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Score an edge list (``node1, node2, edge_type, weight``); see module docstring."""
    out: Dict[str, float] = {}
    e = edges.copy()
    e["node1"] = e["node1"].astype(str); e["node2"] = e["node2"].astype(str)
    tt = e[e["edge_type"] == 1]
    out["n_edges_tt"] = int(len(tt))
    if len(tt):
        m1 = tt["node1"].map(truth_gene); m2 = tt["node2"].map(truth_gene)
        known = m1.notna() & m2.notna()
        same = (m1[known] == m2[known]) & (m1[known] >= 0)
        out["tt_precision"] = float(same.mean()) if known.any() else np.nan
    else:
        out["tt_precision"] = np.nan
    if truth_edges is not None and tx2gene is not None and len(truth_edges):
        te = truth_edges.copy()
        g1 = te["source"].astype(str).map(tx2gene); g2 = te["target"].astype(str).map(tx2gene)
        pairs = {tuple(sorted((a, b))) for a, b in zip(g1, g2) if isinstance(a, str) and isinstance(b, str) and a != b}
        pred = {tuple(sorted((a, b))) for a, b in zip(tt["node1"], tt["node2"])}
        out["n_truth_gene_edges"] = len(pairs)
        out["tt_recall"] = float(len(pairs & pred) / len(pairs)) if pairs else np.nan
    if switch_table is not None:
        st = switch_table.set_index(switch_table["transcript_id"].astype(str))
        ti = e[e["edge_type"] == 2]
        iso_col = np.where(ti["node1"].isin(st.index), ti["node1"], ti["node2"])
        gene_col = np.where(ti["node1"].isin(st.index), ti["node2"], ti["node1"])
        ti = pd.DataFrame({"iso": iso_col, "gene": gene_col})
        ti = ti[ti["iso"].isin(st.index)]
        gm = ti["gene"].map(truth_gene)
        is_sw = st.loc[ti["iso"], "is_switched"].values == 1
        tgt = st.loc[ti["iso"], "tx_module"].values
        own = st.loc[ti["iso"], "gene_module"].values
        out["n_edges_ti"] = int(len(ti))
        sw_edges = ti[is_sw]
        out["ti_switched_precision"] = float((gm[is_sw].values == tgt[is_sw]).mean()) if is_sw.any() else np.nan
        sw_iso = st.index[(st["is_switched"] == 1) & (st["isoform_rank"] > 0)]
        hit = set(ti["iso"][is_sw & (gm.values == tgt)])
        out["ti_switched_recall"] = float(len(hit & set(sw_iso)) / len(sw_iso)) if len(sw_iso) else np.nan
        ns = ~is_sw & (own >= 0)
        out["ti_unswitched_cross_module_rate"] = float((gm[ns].values != own[ns]).mean()) if ns.any() else np.nan
        ns_iso = st.index[(st["is_switched"] == 0) & (st["isoform_rank"] > 0) & (st["gene_module"] >= 0)]
        cross = set(ti["iso"][ns & (gm.values != own)])
        out["ti_unswitched_false_rate"] = float(len(cross & set(ns_iso)) / len(ns_iso)) if len(ns_iso) else np.nan
    return out


__all__ = ["edges_from_labels", "edge_metrics"]
