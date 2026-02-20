"""
OmicsDataset — feature-level dataset for GMMModuleVAE.

Each data point is ONE FEATURE (e.g., a gene or isoform) represented
by its sample-level profile x_f ∈ R^N.

Supported formats (auto-detected by file extension):
  .csv / .csv.gz   → pandas (comma separator)
  .tsv / .tsv.gz   → pandas (tab separator)
  .h5 / .hdf5      → h5py (lazy-load; expects features × samples)
  .h5ad            → anndata (optional dependency)

__getitem__ returns (profile: Tensor shape (n_samples,), feature_id: str).

Backward-compatibility shim
----------------------------
GeneExpression is kept as an alias so that code that calls
  GeneExpression(gene_expression_filename=...) still works during the
  transition period.  It is deprecated; use OmicsDataset directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------

def _infer_format(path: str) -> str:
    """Infer file format from extension."""
    suffixes = [s.lower() for s in Path(path).suffixes]
    if ".h5ad" in suffixes:
        return "h5ad"
    if ".h5" in suffixes or ".hdf5" in suffixes:
        return "hdf5"
    if ".tsv" in suffixes:
        return "tsv"
    return "csv"


def _load_matrix(path: str):
    """
    Load a features × samples matrix.

    Returns
    -------
    data : np.ndarray, shape (n_features, n_samples)
    feature_ids : list of str
    sample_ids : list of str
    """
    fmt = _infer_format(path)

    if fmt == "h5ad":
        try:
            import anndata as ad
        except ImportError as e:
            raise ImportError(
                "anndata is required to load .h5ad files: pip install anndata"
            ) from e
        adata = ad.read_h5ad(path)
        # AnnData is samples × features by default; transpose to features × samples
        data = adata.X.T
        if hasattr(data, "toarray"):
            data = data.toarray()
        data = np.array(data, dtype=np.float32)
        feature_ids = list(adata.var_names)
        sample_ids = list(adata.obs_names)
        return data, feature_ids, sample_ids

    if fmt == "hdf5":
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required to load HDF5 files: pip install h5py"
            ) from e
        with h5py.File(path, "r") as f:
            # Try common key names
            for key in ["data", "matrix", "X", list(f.keys())[0]]:
                if key in f:
                    data = f[key][:]
                    break
            feature_ids = (
                [s.decode() if isinstance(s, bytes) else s for s in f["features"][:]]
                if "features" in f else [str(i) for i in range(data.shape[0])]
            )
            sample_ids = (
                [s.decode() if isinstance(s, bytes) else s for s in f["samples"][:]]
                if "samples" in f else [str(i) for i in range(data.shape[1])]
            )
        return np.array(data, dtype=np.float32), feature_ids, sample_ids

    # CSV / TSV
    import pandas as pd
    sep = "\t" if fmt == "tsv" else ","
    df = pd.read_csv(path, index_col=0, sep=sep)
    data = df.values.astype(np.float32)          # (n_features, n_samples)
    feature_ids = list(df.index.astype(str))
    sample_ids = list(df.columns.astype(str))
    return data, feature_ids, sample_ids


# ---------------------------------------------------------------------------
# OmicsDataset
# ---------------------------------------------------------------------------

class OmicsDataset(Dataset):
    """
    Feature-level omics dataset for GMMModuleVAE.

    Parameters
    ----------
    path : str
        Path to the expression matrix (features × samples).
        Supported: CSV, TSV, HDF5, h5ad.
    feature_subset : list of str, optional
        If provided, only load this subset of features (by ID).
    logger : logging.Logger, optional
    """

    def __init__(
        self,
        path: str,
        feature_subset: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.path = path
        self.logger = logger or _LOG

        self.logger.info("Loading omics data from %s", path)
        data, feature_ids, sample_ids = _load_matrix(path)

        self.sample_ids = sample_ids
        self.n_samples = len(sample_ids)

        if feature_subset is not None:
            idx_map = {fid: i for i, fid in enumerate(feature_ids)}
            keep = [idx_map[f] for f in feature_subset if f in idx_map]
            missing = [f for f in feature_subset if f not in idx_map]
            if missing:
                self.logger.warning(
                    "%d requested features not found in data", len(missing)
                )
            data = data[keep]
            feature_ids = [feature_ids[i] for i in keep]

        self.feature_ids: List[str] = feature_ids
        self.data = torch.from_numpy(data)   # (n_features, n_samples)

        self.logger.info(
            "Loaded %d features × %d samples", len(feature_ids), self.n_samples
        )

    def __len__(self) -> int:
        return len(self.feature_ids)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        profile : torch.Tensor, shape (n_samples,)
        feature_id : str
        """
        return self.data[idx], self.feature_ids[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_omics_dataloader(
    path: str,
    batch_size: int = 128,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    num_workers: int = 0,
    feature_subset: Optional[List[str]] = None,
    sampler: Optional[Sampler] = None,
    logger: Optional[logging.Logger] = None,
) -> DataLoader:
    """Create a DataLoader for OmicsDataset."""
    pin_memory = pin_memory and torch.cuda.is_available()
    dataset = OmicsDataset(path, feature_subset=feature_subset, logger=logger)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------

class GeneExpression(Dataset):
    """
    Deprecated: use OmicsDataset directly.

    Legacy sample-level dataset kept for backward compatibility.
    Each item is one *sample's* expression profile (samples × genes mode).
    """

    def __init__(
        self,
        root="/",
        gene_expression_filename=None,
        gene_expression_dir=None,
        fold_id=0,
        train=True,
        random_state=13,
        logger=None,
        **kwargs,
    ):
        import warnings
        warnings.warn(
            "GeneExpression is deprecated. Use OmicsDataset for feature-level training.",
            DeprecationWarning,
            stacklevel=2,
        )
        import pandas as pd
        from sklearn.model_selection import KFold

        self.logger = logger or _LOG

        if not (gene_expression_filename or gene_expression_dir) or \
                (gene_expression_filename and gene_expression_dir):
            raise ValueError(
                "Provide exactly one of gene_expression_filename or gene_expression_dir."
            )

        if gene_expression_filename:
            path = gene_expression_filename
            suffixes = [s.lower() for s in Path(path).suffixes]
            sep = "\t" if ".tsv" in suffixes else ","
            full_df = pd.read_csv(path, index_col=0, sep=sep)

            sample_ids = np.array(full_df.columns)
            n_samples = len(sample_ids)
            n_splits = min(10, n_samples)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            all_splits = list(kf.split(sample_ids))
            train_idx, test_idx = all_splits[fold_id]
            chosen = sample_ids[train_idx] if train else sample_ids[test_idx]
            dfx = full_df[chosen]
        else:
            fname = "X_train" if train else "X_test"
            path = _find_split_file(gene_expression_dir, fname)
            if path is None:
                raise FileNotFoundError(f"{fname} not found in {gene_expression_dir}")
            suffixes = [s.lower() for s in Path(path).suffixes]
            sep = "\t" if ".tsv" in suffixes else ","
            dfx = pd.read_csv(path, index_col=0, sep=sep)

        self.data = torch.from_numpy(dfx.T.values.astype(np.float32))
        self.genes = list(dfx.index)
        self.samples = list(dfx.columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.samples[idx]


def _find_split_file(directory: str, base_name: str) -> Optional[str]:
    for base_ext in (".csv", ".tsv"):
        for comp in ("", ".gz", ".bz2", ".zip", ".xz"):
            candidate = os.path.join(directory, f"{base_name}{base_ext}{comp}")
            if os.path.exists(candidate):
                return candidate
    return None


def get_dataloaders(dataset="omics", root=None, shuffle=True, pin_memory=True,
                    batch_size=128, drop_last=False,
                    logger=logging.getLogger(__name__), **kwargs):
    """Legacy factory — delegates to get_omics_dataloader."""
    path = kwargs.get("gene_expression_filename") or kwargs.get("path")
    if path is None:
        raise ValueError("Provide path or gene_expression_filename")
    return get_omics_dataloader(
        path=path,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        logger=logger,
    )
