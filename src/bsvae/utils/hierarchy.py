"""
Hierarchy utilities for isoform-aware module discovery.

Functions
---------
load_tx2gene(path=None, feature_ids=None)
    Load or infer transcript → gene mapping.

group_isoforms_by_gene(tx2gene, feature_ids)
    Return dict[gene_id → list of dataset-level feature indices]
    (only genes with ≥ 2 isoforms).

GeneGroupedBatchSampler
    Batch sampler that places all isoforms of sampled genes in the same
    batch (a permutation of the dataset per epoch), so the hierarchical
    loss sees complete isoform groups.

IsoformStratifiedSampler
    Legacy sampler kept for backward compatibility.  It draws features
    uniformly from the multi-isoform pool, which rarely puts isoforms of
    the same gene in one batch; prefer GeneGroupedBatchSampler.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# tx2gene loading
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"\.\d+$")


def strip_feature_version(feature_id: str, keep_version: bool = False) -> str:
    """
    Normalise a transcript/gene identifier.

    * GENCODE pipe-delimited IDs (``ENST…|ENSG…|…``) are reduced to the first field.
    * Ensembl version suffixes (``ENST00000123456.5`` → ``ENST00000123456``) are
      removed unless ``keep_version=True``.
    """
    fid = str(feature_id).strip()
    if "|" in fid:
        fid = fid.split("|", 1)[0]
    if not keep_version:
        fid = _VERSION_RE.sub("", fid)
    return fid


def _looks_like_header(first_row: Sequence[str]) -> bool:
    """Heuristic: a header row contains no Ensembl-style ID and is alphabetic."""
    cells = [str(c).strip() for c in first_row]
    if any(re.match(r"^ENS[A-Z]*[GT]\d+", c, re.IGNORECASE) for c in cells):
        return False
    return all(re.match(r"^[A-Za-z_][A-Za-z0-9_ .-]*$", c) for c in cells if c)


def load_tx2gene(
    path: Optional[str] = None,
    feature_ids: Optional[Sequence[str]] = None,
    keep_version: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.Series:
    """
    Load a transcript-to-gene mapping.

    Parameters
    ----------
    path : str
        TSV/CSV with (transcript_id, gene_id) as named columns or as the
        first two columns.  A header row is auto-detected, so headerless
        tximport-style files work too.
    feature_ids : sequence of str, optional
        Dataset feature IDs.  When given, the match rate between the
        mapping and the dataset is checked and a warning is emitted if
        fewer than half of the features are covered.
    keep_version : bool
        Keep Ensembl version suffixes on both sides. Default: strip them,
        so ``ENST…​.5`` in the data matches ``ENST…`` in the mapping.

    Returns
    -------
    tx2gene : pd.Series
        Index = normalised transcript_id, values = normalised gene_id.

    Notes
    -----
    Ensembl transcript and gene identifiers do **not** share numeric
    suffixes, so the mapping cannot be inferred from transcript IDs alone;
    a mapping file is required.
    """
    log = logger or logging.getLogger(__name__)
    if path is None:
        raise ValueError(
            "A transcript-to-gene mapping file is required (--tx2gene). Ensembl "
            "transcript and gene IDs do not share numeric suffixes, so the mapping "
            "cannot be inferred from the transcript IDs."
        )

    sep = "\t"
    suffixes = [s.lower() for s in Path(path).suffixes]
    if ".csv" in suffixes:
        sep = ","
    raw = pd.read_csv(path, sep=sep, header=None, dtype=str, comment="#")
    if raw.shape[1] < 2:
        raise ValueError(f"tx2gene file {path} must have at least two columns")

    first = list(raw.iloc[0])
    if _looks_like_header(first):
        header = [str(c).strip() for c in first]
        df = raw.iloc[1:].copy()
        df.columns = header
        if "transcript_id" in df.columns and "gene_id" in df.columns:
            df = df[["transcript_id", "gene_id"]]
        else:
            df = df.iloc[:, :2]
            df.columns = ["transcript_id", "gene_id"]
    else:
        df = raw.iloc[:, :2].copy()
        df.columns = ["transcript_id", "gene_id"]

    df = df.dropna()
    df["transcript_id"] = df["transcript_id"].map(lambda v: strip_feature_version(v, keep_version))
    df["gene_id"] = df["gene_id"].map(lambda v: strip_feature_version(v, keep_version))
    df = df.drop_duplicates("transcript_id", keep="first")
    tx2gene = df.set_index("transcript_id")["gene_id"]

    if feature_ids is not None and len(feature_ids) > 0:
        norm_ids = {strip_feature_version(f, keep_version) for f in feature_ids}
        matched = len(norm_ids & set(tx2gene.index))
        rate = matched / max(len(norm_ids), 1)
        if rate < 0.5:
            log.warning(
                "tx2gene covers only %.1f%% of dataset features (%d/%d). Check that "
                "the identifiers match (version suffixes are stripped automatically).",
                100 * rate, matched, len(norm_ids),
            )
        else:
            log.info("tx2gene covers %.1f%% of dataset features", 100 * rate)
    return tx2gene


def group_isoforms_by_gene(
    tx2gene: pd.Series,
    feature_ids: Sequence[str],
    keep_version: bool = False,
) -> Dict[str, List[int]]:
    """
    Group dataset feature indices by gene (only genes with ≥ 2 isoforms).

    Parameters
    ----------
    tx2gene : pd.Series
        Index = transcript_id, values = gene_id.
    feature_ids : sequence of str
        Ordered list of feature IDs in the dataset.

    Returns
    -------
    groups : dict[gene_id → list[int]]
        Only genes with ≥ 2 isoforms present in feature_ids.
    """
    id_to_idx: Dict[str, int] = {}
    for i, fid in enumerate(feature_ids):
        # Normalise the dataset IDs the same way as the mapping so versioned
        # and pipe-delimited identifiers still match; first occurrence wins.
        id_to_idx.setdefault(strip_feature_version(fid, keep_version), i)
    groups: Dict[str, List[int]] = {}

    for tx_id, gene_id in tx2gene.items():
        idx = id_to_idx.get(strip_feature_version(tx_id, keep_version))
        if idx is not None:
            groups.setdefault(str(gene_id), []).append(idx)

    return {gene: sorted(set(idxs)) for gene, idxs in groups.items() if len(set(idxs)) >= 2}


# ---------------------------------------------------------------------------
# IsoformStratifiedSampler
# ---------------------------------------------------------------------------

class IsoformStratifiedSampler(Sampler):
    """
    Over-samples multi-isoform gene batches.

    With probability ``p_multi`` (default: 0.5) the batch is drawn from the
    pool of features belonging to multi-isoform genes; otherwise uniform.

    Parameters
    ----------
    n_features : int
        Total number of features in the dataset.
    multi_isoform_indices : list of int
        Dataset-level indices of features that belong to genes with ≥ 2 isoforms.
    batch_size : int
    p_multi : float
        Probability of drawing a full batch from multi-isoform features.
    num_samples : int or None
        Total samples per epoch.  Defaults to n_features.
    seed : int
    """

    def __init__(
        self,
        n_features: int,
        multi_isoform_indices: List[int],
        batch_size: int = 128,
        p_multi: float = 0.5,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.n_features = n_features
        self.multi = np.array(multi_isoform_indices)
        self.all_idx = np.arange(n_features)
        self.batch_size = batch_size
        self.p_multi = p_multi
        self._num_samples = num_samples or n_features
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        indices = []
        while len(indices) < self._num_samples:
            use_multi = len(self.multi) > 0 and rng.random() < self.p_multi
            pool = self.multi if use_multi else self.all_idx
            batch = rng.choice(pool, size=min(self.batch_size, self._num_samples - len(indices)),
                               replace=True)
            indices.extend(batch.tolist())

        return iter(indices[: self._num_samples])


# ---------------------------------------------------------------------------
# GeneGroupedBatchSampler
# ---------------------------------------------------------------------------

class GeneGroupedBatchSampler(Sampler):
    """
    Batch sampler that keeps isoforms of the same gene together.

    Each epoch visits every feature exactly once (a permutation of the
    dataset).  With probability ``p_multi`` a batch is filled gene by gene:
    whole isoform groups are appended until ``batch_size`` is reached (the
    group that would overflow starts the next batch).  Otherwise the batch
    is drawn from the remaining features at random.  Features that belong
    to no multi-isoform gene are treated as singleton groups.

    Parameters
    ----------
    n_features : int
    gene_groups : dict[gene_id → list of dataset indices] (≥ 2 isoforms each)
    batch_size : int
    p_multi : float
        Probability that a batch is built from complete gene groups.
    drop_last : bool
    seed : int
    """

    def __init__(
        self,
        n_features: int,
        gene_groups: Dict[str, List[int]],
        batch_size: int = 128,
        p_multi: float = 0.5,
        drop_last: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.n_features = int(n_features)
        self.batch_size = int(batch_size)
        self.p_multi = float(p_multi)
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0

        grouped = set()
        self.groups: List[List[int]] = []
        for idxs in gene_groups.values():
            idxs = sorted({int(i) for i in idxs if 0 <= int(i) < self.n_features})
            if len(idxs) >= 2:
                self.groups.append(idxs)
                grouped.update(idxs)
        self.singletons = [i for i in range(self.n_features) if i not in grouped]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_features // self.batch_size
        return (self.n_features + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        # Units are whole gene groups plus singleton features, in random order.
        units: List[List[int]] = [list(g) for g in self.groups] + [[i] for i in self.singletons]
        order = rng.permutation(len(units))
        multi_units = [units[i] for i in order if len(units[i]) > 1]
        single_units = [units[i] for i in order if len(units[i]) == 1]

        batches: List[List[int]] = []
        current: List[int] = []
        while multi_units or single_units:
            use_multi = bool(multi_units) and (not single_units or rng.random() < self.p_multi)
            if use_multi:
                unit = multi_units.pop()
                if current and len(current) + len(unit) > self.batch_size:
                    batches.append(current)
                    current = []
                current.extend(unit)
            else:
                unit = single_units.pop()
                current.extend(unit)
            # Oversized gene groups are split across consecutive batches.
            while len(current) >= self.batch_size:
                batches.append(current[: self.batch_size])
                current = current[self.batch_size:]
        if current and not self.drop_last:
            batches.append(current)
        elif current and self.drop_last and len(current) == self.batch_size:
            batches.append(current)

        return iter(batches)
