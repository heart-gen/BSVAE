"""
Hierarchy utilities for isoform-aware module discovery.

Functions
---------
load_tx2gene(path=None, feature_ids=None)
    Load or infer transcript → gene mapping.

group_isoforms_by_gene(tx2gene, feature_ids)
    Return dict[gene_id → list of dataset-level feature indices]
    (only genes with ≥ 2 isoforms).

IsoformStratifiedSampler
    Over-samples multi-isoform gene batches to enable L_hier computation.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


# ---------------------------------------------------------------------------
# tx2gene loading
# ---------------------------------------------------------------------------

def load_tx2gene(
    path: Optional[str] = None,
    feature_ids: Optional[Sequence[str]] = None,
) -> pd.Series:
    """
    Load or infer a transcript-to-gene mapping.

    Parameters
    ----------
    path : str or None
        TSV with columns (transcript_id, gene_id).  If None, the mapping
        is inferred from Ensembl IDs assuming ENST/ENSG share the same
        numeric suffix (e.g., ENST00000123456 → ENSG00000123456).
    feature_ids : sequence of str or None
        Feature identifiers used for inference (required if path is None).

    Returns
    -------
    tx2gene : pd.Series
        Index = transcript_id, values = gene_id.
    """
    if path is not None:
        df = pd.read_csv(path, sep="\t")
        # Accept (transcript_id, gene_id) or first two columns
        cols = list(df.columns)
        if "transcript_id" in cols and "gene_id" in cols:
            df = df[["transcript_id", "gene_id"]]
        else:
            df = df.iloc[:, :2]
            df.columns = ["transcript_id", "gene_id"]
        return df.set_index("transcript_id")["gene_id"]

    # Infer from Ensembl IDs
    if feature_ids is None:
        raise ValueError("Either path or feature_ids must be provided.")

    mapping = {}
    _enst_re = re.compile(r"ENST(\d+)", re.IGNORECASE)
    _ensg_re = re.compile(r"ENSG(\d+)", re.IGNORECASE)

    for fid in feature_ids:
        m = _enst_re.search(fid)
        if m:
            suffix = m.group(1)
            mapping[fid] = f"ENSG{suffix}"
        else:
            # Not a recognisable transcript ID — map to itself
            mapping[fid] = fid

    return pd.Series(mapping, name="gene_id")


def group_isoforms_by_gene(
    tx2gene: pd.Series,
    feature_ids: Sequence[str],
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
    id_to_idx = {fid: i for i, fid in enumerate(feature_ids)}
    groups: Dict[str, List[int]] = {}

    for tx_id, gene_id in tx2gene.items():
        if tx_id in id_to_idx:
            groups.setdefault(gene_id, []).append(id_to_idx[tx_id])

    return {gene: idxs for gene, idxs in groups.items() if len(idxs) >= 2}


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
