"""
Targeted-gene mode: interrogate a trained, transcriptome-wide module solution
for a small set of genes of interest (candidate genes, GWAS loci).

Nothing is re-trained.  For every isoform of a target gene the
gene-programme-removed residual (see :mod:`bsvae.utils.isoform`) is correlated
with each module eigengene, and the best module other than the gene's own is
tested against a calibrated null.  Because only the target isoforms are
tested, multiplicity is controlled within the target set and rare isoforms
whose correlations are too weak for a transcriptome-wide call become
reportable.

Target selection
----------------
* a gene list (Ensembl IDs, or symbols when an annotation is supplied),
* a transcript list,
* a BED file of loci plus a gene annotation (GTF or BED-like table), with
  genes picked per locus by overlap, by a window around the locus, or as the
  N closest genes by TSS distance.
"""

from __future__ import annotations

import gzip
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from bsvae.utils.hierarchy import strip_feature_version

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input readers
# ---------------------------------------------------------------------------

def _open(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")


def read_id_list(path: str | Path, keep_version: bool = False) -> List[str]:
    """One identifier per line (first whitespace/comma-delimited column); ``#`` lines skipped."""
    ids: List[str] = []
    with _open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = re.split(r"[\s,]+", line)[0]
            if tok.lower() in {"gene_id", "gene", "transcript_id", "transcript", "feature_id", "id", "symbol"}:
                continue
            ids.append(strip_feature_version(tok, keep_version))
    return list(dict.fromkeys(ids))


def read_bed(path: str | Path) -> pd.DataFrame:
    """
    Loci as a BED file (``chrom start end [name ...]``, 0-based half-open).
    Returns ``locus, chrom, start, end`` with ``locus`` the name column or
    ``chrom:start-end``.
    """
    rows = []
    with _open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
            if len(parts) < 3:
                raise ValueError(f"BED line has fewer than three columns: {line!r}")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            name = parts[3] if len(parts) > 3 and parts[3] not in ("", ".") else f"{chrom}:{start}-{end}"
            rows.append({"locus": name, "chrom": chrom, "start": start, "end": end})
    if not rows:
        raise ValueError(f"No loci found in {path}")
    return pd.DataFrame(rows)


def _norm_chrom(c: str) -> str:
    c = str(c)
    return c[3:] if c.lower().startswith("chr") else c


def read_gene_annotation(path: str | Path, keep_version: bool = False) -> pd.DataFrame:
    """
    Gene coordinates from a GTF/GFF (``gene`` features) or a BED-like table
    (``chrom start end gene_id [gene_name] [strand]``, header optional).
    Returns ``gene_id, gene_name, chrom, start, end, strand, tss``.
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    rows = []
    if ".gtf" in suffixes or ".gff" in suffixes or ".gff3" in suffixes:
        with _open(path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                attrs = parts[8]
                gid = re.search(r'gene_id[ =]"?([^";]+)"?', attrs)
                if not gid:
                    continue
                gname = re.search(r'gene_name[ =]"?([^";]+)"?', attrs)
                rows.append({
                    "gene_id": strip_feature_version(gid.group(1), keep_version),
                    "gene_name": gname.group(1) if gname else "",
                    "chrom": _norm_chrom(parts[0]), "start": int(parts[3]) - 1, "end": int(parts[4]),
                    "strand": parts[6] if parts[6] in "+-" else "+",
                })
    else:
        with _open(path) as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.rstrip("\n").split("\t") if "\t" in line else line.split()
                if len(parts) < 4:
                    raise ValueError(f"annotation line needs chrom, start, end, gene_id: {line!r}")
                if parts[0].lower() in ("chrom", "chr", "chromosome"):
                    continue
                rows.append({
                    "gene_id": strip_feature_version(parts[3], keep_version),
                    "gene_name": parts[4] if len(parts) > 4 and parts[4] not in (".", "") else "",
                    "chrom": _norm_chrom(parts[0]), "start": int(parts[1]), "end": int(parts[2]),
                    "strand": parts[5] if len(parts) > 5 and parts[5] in "+-" else "+",
                })
    if not rows:
        raise ValueError(f"No gene records found in {path}")
    ann = pd.DataFrame(rows).drop_duplicates("gene_id")
    ann["tss"] = np.where(ann["strand"] == "-", ann["end"], ann["start"])
    return ann.reset_index(drop=True)


def loci_to_genes(
    loci: pd.DataFrame,
    annotation: pd.DataFrame,
    rule: str = "overlap",
    window: int = 500_000,
    n_closest: int = 3,
) -> pd.DataFrame:
    """
    Map loci to genes.  ``rule``:

    * ``"overlap"``: genes whose body overlaps the locus;
    * ``"window"``: genes whose body overlaps the locus extended by ``window``
      on both sides (a one-base BED of index SNPs gives "within ``window`` of
      the SNP");
    * ``"closest"``: the ``n_closest`` genes with the smallest TSS distance to
      the locus (zero when the TSS lies inside it).

    Returns one row per (locus, gene) with ``distance`` (TSS to locus, bp) and
    the rule used.
    """
    if rule not in ("overlap", "window", "closest"):
        raise ValueError("rule must be 'overlap', 'window' or 'closest'")
    rows = []
    ann_by_chrom = {c: g for c, g in annotation.groupby("chrom")}
    for loc in loci.itertuples(index=False):
        chrom = _norm_chrom(loc.chrom)
        genes = ann_by_chrom.get(chrom)
        if genes is None:
            logger.warning("Locus %s: chromosome %s not in annotation", loc.locus, loc.chrom)
            continue
        pad = int(window) if rule == "window" else 0
        lo, hi = int(loc.start) - pad, int(loc.end) + pad
        tss = genes["tss"].values
        dist = np.where(tss < int(loc.start), int(loc.start) - tss, np.where(tss >= int(loc.end), tss - int(loc.end) + 1, 0))
        if rule in ("overlap", "window"):
            hit = (genes["start"].values < hi) & (genes["end"].values > lo)
            sel = np.where(hit)[0]
        else:
            sel = np.argsort(dist, kind="stable")[: int(n_closest)]
        for i in sel:
            g = genes.iloc[i]
            rows.append({"locus": loc.locus, "chrom": loc.chrom, "locus_start": int(loc.start), "locus_end": int(loc.end),
                         "gene_id": g["gene_id"], "gene_name": g["gene_name"], "distance": int(dist[i]), "rule": rule})
    out = pd.DataFrame(rows, columns=["locus", "chrom", "locus_start", "locus_end", "gene_id", "gene_name", "distance", "rule"])
    return out.sort_values(["locus", "distance", "gene_id"]).reset_index(drop=True)


def resolve_target_transcripts(
    tx_gene: pd.Series,
    genes: Optional[Iterable[str]] = None,
    transcripts: Optional[Iterable[str]] = None,
    annotation: Optional[pd.DataFrame] = None,
    keep_version: bool = False,
) -> Tuple[pd.Series, List[str]]:
    """
    Transcripts of the target genes (plus explicitly listed transcripts) that
    are present in ``tx_gene`` (feature_id → gene_id for the expression
    matrix).  Gene symbols are resolved through ``annotation`` when given.
    Returns (feature_id → gene_id for the targets, list of unmatched inputs).
    """
    tx_gene = tx_gene.copy()
    tx_gene.index = [strip_feature_version(f, keep_version) for f in tx_gene.index]
    unmatched: List[str] = []
    target_genes: set = set()
    if genes:
        known = set(tx_gene.dropna().astype(str).values)
        sym2id: Dict[str, str] = {}
        if annotation is not None and "gene_name" in annotation:
            sym2id = {str(n).upper(): g for g, n in zip(annotation["gene_id"], annotation["gene_name"]) if n}
        for g in genes:
            gid = strip_feature_version(g, keep_version)
            if gid in known:
                target_genes.add(gid)
            elif gid.upper() in sym2id and sym2id[gid.upper()] in known:
                target_genes.add(sym2id[gid.upper()])
            else:
                unmatched.append(g)
    sel = tx_gene[tx_gene.isin(target_genes)] if target_genes else tx_gene.iloc[0:0]
    if transcripts:
        ids = [strip_feature_version(t, keep_version) for t in transcripts]
        present = [t for t in ids if t in tx_gene.index]
        unmatched.extend([t for t, i in zip(transcripts, ids) if i not in tx_gene.index])
        sel = pd.concat([sel, tx_gene.loc[present]])
    sel = sel[~sel.index.duplicated()]
    return sel, unmatched


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _zscore_rows(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    sd = Xc.std(axis=1, ddof=1, keepdims=True)
    return Xc / np.where(sd == 0, 1.0, sd)


def _standardize_cols(E: np.ndarray) -> np.ndarray:
    Ec = E - E.mean(axis=0, keepdims=True)
    sd = Ec.std(axis=0, ddof=1, keepdims=True)
    return Ec / np.where(sd == 0, 1.0, sd)


def profile_reliability(
    counts: pd.DataFrame,
    expr: pd.DataFrame,
    feature_ids: Sequence[str],
    theta: float = 20.0,
    n_draws: int = 64,
    seed: int = 0,
) -> pd.Series:
    """
    Fraction of each log2 profile's variance that is not sampling noise, from
    the raw counts.  For every sample the observed count is taken as the
    negative-binomial mean (technical dispersion ``1 / theta``; biological
    variation is signal and deliberately not estimated), ``n_draws`` counts
    are re-drawn and transformed exactly like the expression matrix
    (``log2(c / s + 1)`` with the size factor recovered from the matrix), and
    the mean across samples of the variance across draws is the noise
    variance.  Reliability is ``max(var - noise, 0) / var``; it is computed
    for the listed features only, so it is cheap for a target set.  Used to
    disattenuate kME: ``kme / sqrt(reliability)``.
    """
    ids = [str(f) for f in feature_ids]
    C = counts.reindex(ids).values.astype(np.float64)
    X = expr.reindex(ids).values.astype(np.float64)
    lin = np.power(2.0, X) - 1.0
    ratio = np.where(lin > 0, C / np.maximum(lin, 1e-12), np.nan)
    s_eff = np.nanmedian(ratio, axis=0)
    if not np.all(np.isfinite(s_eff) & (s_eff > 0)):
        from bsvae.utils.preprocess import size_factors
        s_eff = np.where(np.isfinite(s_eff) & (s_eff > 0), s_eff, size_factors(counts.values))
    rng = np.random.default_rng(int(seed))
    mu = np.maximum(C, 0.5)
    pnb = float(theta) / (float(theta) + mu)
    draws = rng.negative_binomial(float(theta), pnb[None, :, :], size=(int(n_draws),) + mu.shape).astype(np.float64)
    xs = np.log2(draws / s_eff[None, None, :] + 1.0)
    noise_var = xs.var(axis=0, ddof=1).mean(axis=1)
    total = X.var(axis=1, ddof=1)
    rel = np.where(total > 0, np.maximum(total - noise_var, 0.0) / np.where(total > 0, total, 1.0), np.nan)
    return pd.Series(rel, index=ids, name="reliability")


def target_gene_tests(
    expr: pd.DataFrame,
    tx2gene: "pd.Series | Dict[str, str]",
    labels: np.ndarray,
    feature_ids: Sequence[str],
    targets: pd.Series,
    *,
    eigengenes: Optional[pd.DataFrame] = None,
    null: str = "permutation",
    n_perm: int = 1000,
    n_matched: int = 200,
    min_kme: float = 0.2,
    alpha: float = 0.05,
    counts: Optional[pd.DataFrame] = None,
    condition: Optional[pd.Series] = None,
    n_neighbours: int = 20,
    residualize_eigengenes: bool = True,
    leave_one_out: bool = True,
    keep_version: bool = False,
    seed: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Test every isoform of the target genes for membership of a module other
    than its gene's own, against the transcriptome-wide module solution.

    Parameters
    ----------
    expr, tx2gene, labels, feature_ids
        As for :func:`bsvae.utils.isoform.isoform_specific_calls`.
    targets
        feature_id → gene_id for the target transcripts
        (:func:`resolve_target_transcripts`).
    null
        ``"permutation"``: the residual is permuted over samples ``n_perm``
        times and the maximum |kME| over the other modules forms the null
        (handles best-module selection and non-Gaussian profiles);
        ``"matched"``: the observed best-other |kME| of the ``n_matched``
        non-target isoforms, assigned to their gene's own module, closest in
        mean expression and zero fraction (also absorbs residual leakage); ``"analytic"``: t-test as in the
        transcriptome-wide caller.
    counts
        Raw counts (features × samples) to estimate each profile's
        reliability and report a disattenuated kME.
    condition
        Two-level sample grouping (index = sample IDs); the best-other kME is
        then also estimated per group and the difference tested with
        Fisher's z (``p_interaction``).

    min_kme, alpha
        Effect-size floor and BH FDR within the target set.  The floor is
        lower than the transcriptome-wide caller's (0.2 vs 0.3): with a few
        hundred samples the permutation test, not the floor, should decide,
        and on the rare-isoform simulations 0.2 roughly doubles recall of
        low-abundance switched isoforms at 500 samples for a false-call rate
        under 10 % among unswitched isoforms (0.0 reaches 0.6 recall but
        17–21 % false calls; 0.3 reproduces the transcriptome-wide caller).

    Returns a dict with ``isoforms`` (one row per target isoform),
    ``genes`` (per-gene summary) and ``neighbours`` (top module members
    correlated with each called isoform's residual).
    """
    from collections import Counter

    from bsvae.utils.isoform import (
        _bh_adjust, _residualize_eigengenes, isoform_residuals, map_transcripts_to_genes,
    )
    from bsvae.networks.postprocess import module_eigengenes

    if null not in ("permutation", "matched", "analytic"):
        raise ValueError("null must be 'permutation', 'matched' or 'analytic'")
    rng = np.random.default_rng(int(seed))
    expr = expr.copy(); expr.index = expr.index.astype(str)
    fids = [str(f) for f in feature_ids]
    labels = np.asarray(labels).astype(np.int64)
    lab = pd.Series(labels, index=fids)
    if eigengenes is None:
        eigengenes = module_eigengenes(expr, labels, fids)
    eigengenes = eigengenes.dropna(axis=1, how="any")
    cols = [int(c) for c in eigengenes.columns]
    E = _standardize_cols(eigengenes.values.astype(np.float64))
    S = expr.shape[1]
    targets = targets.copy(); targets.index = targets.index.astype(str)
    targets = targets[targets.index.isin(expr.index)]
    if len(targets) == 0 or len(cols) == 0:
        empty = pd.DataFrame()
        return {"isoforms": empty, "genes": empty, "neighbours": empty}

    tx_gene = map_transcripts_to_genes(fids, tx2gene, keep_version)
    gene_mod: Dict[str, int] = {}
    for g, idx in tx_gene.dropna().groupby(tx_gene.dropna()).groups.items():
        votes = [int(v) for v in lab.loc[list(idx)] if v >= 0]
        gene_mod[g] = Counter(votes).most_common(1)[0][0] if votes else -1

    # residual profiles: multi-isoform transcripts get the sibling residual, single-isoform
    # transcripts their own centred profile
    resid_all, stats_all = isoform_residuals(expr, tx2gene, keep_version=keep_version, leave_one_out=leave_one_out)
    profiles = np.zeros((len(targets), S)); single = np.zeros(len(targets), bool); r2 = np.full(len(targets), np.nan)
    n_iso = np.ones(len(targets), int)
    for i, t in enumerate(targets.index):
        if t in resid_all.index:
            profiles[i] = resid_all.loc[t].values; r2[i] = stats_all.loc[t, "r2_gene"]; n_iso[i] = int(stats_all.loc[t, "n_isoforms"])
        else:
            profiles[i] = expr.loc[t].values - expr.loc[t].values.mean(); single[i] = True
    Z = _zscore_rows(profiles)
    gm = np.array([gene_mod.get(g, -1) for g in targets.values])
    # isoform's own transcriptome-wide label
    own = lab.reindex(targets.index).fillna(-1).astype(int).values

    def kme_matrix(Zrows: np.ndarray, gmods: np.ndarray) -> np.ndarray:
        K = np.zeros((Zrows.shape[0], len(cols)))
        for g in np.unique(gmods):
            rows_idx = np.where(gmods == g)[0]
            if residualize_eigengenes and g in cols:
                Eg = _standardize_cols(_residualize_eigengenes(eigengenes, cols.index(int(g))).values)
            else:
                Eg = E
            K[rows_idx] = Zrows[rows_idx] @ Eg / (S - 1)
        return K

    K = kme_matrix(Z, gm)
    kme_own = np.array([K[i, cols.index(int(g))] if g in cols else np.nan for i, g in enumerate(gm)])
    K_other = K.copy()
    for i, g in enumerate(gm):
        if g in cols:
            K_other[i, cols.index(int(g))] = 0.0
    best_j = np.abs(K_other).argmax(axis=1)
    best = K_other[np.arange(len(targets)), best_j]

    # null distribution of the best-other |kME|
    if null == "analytic":
        from scipy import stats as sps
        rr = np.minimum(best ** 2, 1 - 1e-12)
        tval = np.abs(best) * np.sqrt(max(S - 2, 1) / (1 - rr))
        p = np.minimum(2.0 * sps.t.sf(tval, max(S - 2, 1)) * len(cols), 1.0)
    elif null == "permutation":
        p = np.zeros(len(targets))
        for g in np.unique(gm):
            rows_idx = np.where(gm == g)[0]
            Eg = _standardize_cols(_residualize_eigengenes(eigengenes, cols.index(int(g))).values) if (residualize_eigengenes and g in cols) else E
            other = [j for j in range(len(cols)) if cols[j] != g]
            for i in rows_idx:
                perms = np.array([rng.permutation(S) for _ in range(int(n_perm))])
                null_k = (Z[i][perms] @ Eg[:, other]) / (S - 1)
                null_max = np.abs(null_k).max(axis=1)
                p[i] = (1.0 + (null_max >= abs(best[i])).sum()) / (n_perm + 1.0)
    else:  # matched
        # non-target multi-isoform transcripts that sit in their gene's own module (so transcripts the
        # transcriptome-wide run already found divergent do not inflate the null)
        tset = set(targets.index)
        pool = [t for t, g in zip(stats_all.index, stats_all["gene_id"].values)
                if t not in tset and lab.get(t, -1) >= 0 and lab.get(t, -1) == gene_mod.get(g, -2)]
        if len(pool) < 20:
            raise ValueError("matched null needs at least 20 non-target multi-isoform transcripts in their gene's module")
        pool_expr = expr.reindex(pool)
        pool_feat = np.column_stack([pool_expr.mean(axis=1).values, (pool_expr.values == 0).mean(axis=1)])
        pool_Z = _zscore_rows(resid_all.loc[pool].values)
        pool_gm = np.array([gene_mod.get(g, -1) for g in stats_all.loc[pool, "gene_id"].values])
        pool_K = kme_matrix(pool_Z, pool_gm)
        for i, g in enumerate(pool_gm):
            if g in cols:
                pool_K[i, cols.index(int(g))] = 0.0
        pool_best = np.abs(pool_K).max(axis=1)
        tgt_expr = expr.reindex(targets.index)
        tgt_feat = np.column_stack([tgt_expr.mean(axis=1).values, (tgt_expr.values == 0).mean(axis=1)])
        scale = pool_feat.std(axis=0, ddof=1); scale[scale == 0] = 1.0
        p = np.zeros(len(targets))
        for i in range(len(targets)):
            d = (((pool_feat - tgt_feat[i]) / scale) ** 2).sum(axis=1)
            near = np.argsort(d, kind="stable")[: int(n_matched)]
            p[i] = (1.0 + (pool_best[near] >= abs(best[i])).sum()) / (len(near) + 1.0)
    q = _bh_adjust(p)
    called = (np.abs(best) >= float(min_kme)) & (q < float(alpha))

    out = pd.DataFrame({
        "gene_id": targets.values,
        "n_isoforms": n_iso,
        "single_isoform": single,
        "mean_expr": expr.reindex(targets.index).mean(axis=1).values,
        "zero_fraction": (expr.reindex(targets.index).values == 0).mean(axis=1),
        "gene_module": gm,
        "isoform_module": own,
        "label_discordant": (own >= 0) & (gm >= 0) & (own != gm),
        "kme_gene_module": kme_own,
        "best_other_module": [cols[j] for j in best_j],
        "kme_residual": best,
        "p_value": p,
        "q_value": q,
        "r2_gene": r2,
        "called": called,
    }, index=targets.index)
    out.index.name = "feature_id"

    if counts is not None:
        rel = profile_reliability(counts, expr, list(targets.index))
        out["reliability"] = rel.values
        out["kme_disattenuated"] = np.clip(best / np.sqrt(np.where(rel.values > 0.05, rel.values, np.nan)), -1.0, 1.0)

    if condition is not None:
        cond = condition.copy(); cond.index = cond.index.astype(str)
        cond = cond.reindex([str(c) for c in expr.columns])
        levels = list(pd.unique(cond.dropna()))
        if len(levels) != 2:
            raise ValueError(f"condition must have exactly two levels, got {levels}")
        masks = [(cond == l).values for l in levels]
        k_grp = []
        for mk in masks:
            Zg = _zscore_rows(profiles[:, mk])
            Kg = np.zeros(len(targets))
            for i, g in enumerate(gm):
                Eg = eigengenes.values[mk]
                if residualize_eigengenes and g in cols:
                    Eg = _residualize_eigengenes(pd.DataFrame(Eg, columns=eigengenes.columns), cols.index(int(g))).values
                Eg = _standardize_cols(Eg)
                Kg[i] = Zg[i] @ Eg[:, best_j[i]] / (mk.sum() - 1)
            k_grp.append(Kg)
        n1, n2 = masks[0].sum(), masks[1].sum()
        z1, z2 = np.arctanh(np.clip(k_grp[0], -0.999, 0.999)), np.arctanh(np.clip(k_grp[1], -0.999, 0.999))
        zdiff = (z1 - z2) / np.sqrt(1.0 / max(n1 - 3, 1) + 1.0 / max(n2 - 3, 1))
        from scipy import stats as sps
        out[f"kme_{levels[0]}"] = k_grp[0]
        out[f"kme_{levels[1]}"] = k_grp[1]
        out["z_interaction"] = zdiff
        out["p_interaction"] = 2.0 * sps.norm.sf(np.abs(zdiff))

    # per-gene summary
    gene_rows = []
    for g, sub in out.groupby("gene_id"):
        mods = sorted({int(m) for m in sub["isoform_module"] if m >= 0})
        gene_rows.append({
            "gene_id": g, "n_isoforms_tested": len(sub), "gene_module": int(sub["gene_module"].iloc[0]),
            "isoform_modules": ";".join(map(str, mods)), "n_called": int(sub["called"].sum()),
            "divergent": bool(len(mods) > 1 or sub["called"].any()),
            "best_isoform": sub["q_value"].idxmin(), "min_q": float(sub["q_value"].min()),
        })
    genes = pd.DataFrame(gene_rows).set_index("gene_id") if gene_rows else pd.DataFrame()

    # neighbours: module members most correlated with each called isoform's residual
    nb_rows = []
    if n_neighbours > 0 and called.any():
        Xz = _zscore_rows(expr.values.astype(np.float64))
        for i in np.where(called)[0]:
            m = int(cols[best_j[i]])
            members = np.where(labels == m)[0]
            if members.size == 0:
                continue
            r = Xz[members] @ Z[i] / (S - 1)
            kme_m = Xz[members] @ E[:, cols.index(m)] / (S - 1)
            top = np.argsort(-np.abs(r), kind="stable")[: int(n_neighbours)]
            for j in top:
                nb_rows.append({"feature_id": targets.index[i], "module": m, "neighbour": fids[members[j]],
                                "cor_residual": float(r[j]), "neighbour_kme": float(kme_m[j])})
    neighbours = pd.DataFrame(nb_rows, columns=["feature_id", "module", "neighbour", "cor_residual", "neighbour_kme"])
    logger.info("Targeted-gene tests: %d isoforms of %d genes, %d called (kME ≥ %.2f, FDR < %.2f, %s null)",
                len(out), out["gene_id"].nunique(), int(called.sum()), min_kme, alpha, null)
    return {"isoforms": out, "genes": genes, "neighbours": neighbours}


def locus_summary(isoforms: pd.DataFrame, loci_genes: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Per-locus summary: isoforms tested and called, the modules they point to,
    and a binomial p-value for the most frequent target module against the
    module's share of all assigned features.
    """
    from scipy import stats as sps

    labels = np.asarray(labels)
    assigned = labels[labels >= 0]
    share = {int(m): float((assigned == m).mean()) for m in np.unique(assigned)}
    rows = []
    for locus, sub in loci_genes.groupby("locus"):
        iso = isoforms[isoforms["gene_id"].isin(sub["gene_id"])]
        called = iso[iso["called"]]
        mods = pd.Series(np.concatenate([iso["isoform_module"].values[iso["isoform_module"].values >= 0],
                                         called["best_other_module"].values]) if len(iso) else [], dtype=int)
        top_mod, top_n, p = -1, 0, np.nan
        if len(mods):
            vc = mods.value_counts()
            top_mod, top_n = int(vc.index[0]), int(vc.iloc[0])
            p = float(sps.binomtest(top_n, len(mods), share.get(top_mod, 0.0), alternative="greater").pvalue) if share.get(top_mod, 0) > 0 else np.nan
        rows.append({"locus": locus, "n_genes": sub["gene_id"].nunique(), "n_genes_tested": iso["gene_id"].nunique(),
                     "n_isoforms_tested": len(iso), "n_called": len(called),
                     "called_isoforms": ";".join(called.index), "modules_hit": ";".join(map(str, sorted(set(mods.tolist())))),
                     "top_module": top_mod, "top_module_count": top_n, "p_concentration": p})
    return pd.DataFrame(rows)


def combine_target_tests(
    tables: Dict[str, pd.DataFrame],
    n_samples: Optional[Dict[str, int]] = None,
    alpha: float = 0.05,
    min_kme: float = 0.25,
    min_cohorts: int = 2,
    correlation: str = "none",
    min_called: int = 0,
) -> pd.DataFrame:
    """
    Combine per-cohort (or per-tissue / per-region) targeted-gene tables into
    one evidence table per isoform.

    ``tables`` maps a cohort name to the ``target_isoforms.tsv`` table from
    :func:`target_gene_tests` on that cohort.  Each cohort tests "does this
    isoform's residual track a module other than its gene's?" against its own
    module solution, so the p-values are combined with Stouffer's method
    (weights sqrt(n_samples), or equal) for that question, which does not
    require the modules to be aligned across cohorts.  Module identity is
    reported per cohort (``modules``) together with the number of cohorts in
    which the isoform was called and the mean |kME|.

    ``correlation="empirical"`` applies Strube's correction for dependent
    cohorts (the same donors contributing to several brain regions): the
    pairwise correlation of the per-isoform z-scores between cohorts is
    estimated over all tested isoforms and enters the denominator
    (sqrt(sum w^2 + 2 sum_{i<j} w_i w_j rho_ij)), so shared donor effects do
    not inflate the combined evidence.  ``min_called`` additionally requires
    that many single-cohort calls.

    The combined call requires BH FDR < ``alpha`` across isoforms, mean |kME|
    ≥ ``min_kme`` and at least ``min_cohorts`` cohorts contributing.  On
    simulations of six 150-sample regions of one truth with rare (8 % of
    gene) switched isoforms, pooling raises recall from 0.71 to 0.83 (hard
    switches) and 0.47 to 0.63 (partial) at a 5–6 % false-call rate among
    unswitched isoforms, and recall of the lowest-abundance third from 0.52
    to 0.71; the 0.25 floor is what keeps the false-call rate from growing
    with the number of regions (0.2 gives 0.92 recall but 16 % false calls).
    """
    from scipy import stats as sps

    from bsvae.utils.isoform import _bh_adjust

    if not tables:
        raise ValueError("no tables to combine")
    names = list(tables)
    w = {k: float(np.sqrt(n_samples[k])) if n_samples and k in n_samples else 1.0 for k in names}
    frames = []
    for k, t in tables.items():
        t = t.copy(); t.index = t.index.astype(str)
        t = t[["gene_id", "best_other_module", "kme_residual", "p_value", "called"]].copy()
        t["cohort"] = k
        frames.append(t)
    long = pd.concat(frames)
    if correlation not in ("none", "empirical"):
        raise ValueError("correlation must be 'none' or 'empirical'")
    rho = pd.DataFrame(0.0, index=names, columns=names)
    if correlation == "empirical" and len(names) > 1:
        zmat = long.assign(z=sps.norm.isf(np.clip(long["p_value"].astype(float), 1e-300, 1.0))).pivot_table(
            index=long.index, columns="cohort", values="z")
        # the dependence of interest is between null tests, so estimate it on isoforms that are
        # not nominally significant in any cohort (fall back to all isoforms when too few)
        null_like = zmat[(zmat.max(axis=1) < sps.norm.isf(0.05)) | zmat.isna().any(axis=1)]
        src = null_like if len(null_like) >= 20 else zmat
        rho = src.corr(min_periods=5).fillna(0.0).clip(lower=0.0)
        np.fill_diagonal(rho.values, 1.0)
    rows = []
    for tx, g in long.groupby(level=0):
        p = np.clip(g["p_value"].values.astype(float), 1e-300, 1.0)
        z = sps.norm.isf(p)                                   # one-sided: evidence for membership
        cs = list(g["cohort"])
        ws = np.array([w[c] for c in cs])
        denom2 = float((ws ** 2).sum())
        if correlation == "empirical" and len(cs) > 1:
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    denom2 += 2.0 * ws[i] * ws[j] * float(rho.loc[cs[i], cs[j]])
        zc = float((ws * z).sum() / np.sqrt(denom2))
        rows.append({
            "feature_id": tx, "gene_id": g["gene_id"].iloc[0], "n_cohorts": int(len(g)),
            "n_called": int(g["called"].sum()), "mean_abs_kme": float(g["kme_residual"].abs().mean()),
            "max_abs_kme": float(g["kme_residual"].abs().max()), "z_combined": zc,
            "p_combined": float(sps.norm.sf(zc)),
            "modules": ";".join(f"{c}:{int(m)}" for c, m in zip(g["cohort"], g["best_other_module"])),
        })
    out = pd.DataFrame(rows).set_index("feature_id")
    out["q_combined"] = _bh_adjust(out["p_combined"].values)
    out["called"] = ((out["q_combined"] < float(alpha)) & (out["mean_abs_kme"] >= float(min_kme))
                     & (out["n_cohorts"] >= int(min_cohorts)) & (out["n_called"] >= int(min_called)))
    out.attrs["cohort_z_correlation"] = rho
    return out.sort_values("p_combined")


__all__ = [
    "read_id_list", "read_bed", "read_gene_annotation", "loci_to_genes", "resolve_target_transcripts",
    "profile_reliability", "target_gene_tests", "locus_summary", "combine_target_tests",
]
