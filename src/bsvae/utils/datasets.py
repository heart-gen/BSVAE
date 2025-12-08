import os
import abc
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Registry of supported datasets
DATASETS_DICT = {"genenet": "GeneExpression"}
DATASETS = list(DATASETS_DICT.keys())


def get_dataset(dataset):
    """Return the dataset class corresponding to the given name."""
    dataset = dataset.lower()
    try:
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError(f"Unknown dataset: {dataset}. "
                         f"Available: {list(DATASETS_DICT.keys())}")


def get_dataloaders(dataset, root=None, shuffle=True, pin_memory=True,
                    batch_size=128, drop_last=False,
                    logger=logging.getLogger(__name__), **kwargs):
    """
    Generic data loader wrapper.

    Parameters
    ----------
    dataset : {"genenet", ...}
        Dataset name.
    root : str, optional
        Root directory (used by some datasets).
    kwargs :
        Passed to Dataset constructor and DataLoader.
    """
    pin_memory = pin_memory and torch.cuda.is_available()
    DatasetClass = get_dataset(dataset)
    dataset_instance = DatasetClass(root=root or "/", logger=logger, **kwargs)
    return DataLoader(dataset_instance,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory,
                      drop_last=drop_last)


class BaseDataset(Dataset, abc.ABC):
    """Abstract base class for datasets."""
    def __init__(self, root, logger=logging.getLogger(__name__)):
        self.root = root
        self.logger = logger

    def __len__(self):
        return len(self.data)

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def download(self):
        pass


class GeneExpression(BaseDataset):
    """
    Gene expression dataset (genenet).

    Two modes:
    1. Splitting Mode:
       Provide `gene_expression_filename` (CSV/TSV: genes × samples).
       -> Creates reproducible 10-fold splits.
    2. Pre-split Mode:
       Provide `gene_expression_dir` containing 'X_train.[csv|tsv]' and 'X_test.[csv|tsv]'.

    Parameters
    ----------
    gene_expression_filename : str, optional
        Path to CSV/TSV file with full expression matrix.
    gene_expression_dir : str, optional
        Directory containing 'X_train.[csv|tsv]' and 'X_test.[csv|tsv]'.
    fold_id : int, default=0
        Which CV fold to use (0–9).
    train : bool, default=True
        Whether to load train (True) or test (False) split.
    random_state : int, default=13
        Random seed for CV splitting.
    """
    def __init__(self, root="/",
                 gene_expression_filename=None,
                 gene_expression_dir=None,
                 fold_id=0, train=True,
                 random_state=13,
                 **kwargs):
        super().__init__(root, **kwargs)

        if not (gene_expression_filename or gene_expression_dir) or \
           (gene_expression_filename and gene_expression_dir):
            raise ValueError("Please provide either `gene_expression_filename` "
                             "or `gene_expression_dir`, but not both.")

        if gene_expression_filename:
            self.logger.info(f"Loading and splitting from {gene_expression_filename}")
            full_df = self._read_expression_file(gene_expression_filename)

            # Split across samples (columns) so each dataset item corresponds to a
            # sample profile (num_genes features).
            sample_ids = np.array(full_df.columns)
            n_samples = len(sample_ids)
            n_splits = min(10, n_samples)
            if n_splits < 2:
                raise ValueError(
                    "At least two samples are required to create train/test splits; "
                    f"found {n_samples}."
                )
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            all_splits = list(kf.split(sample_ids))
            if not (0 <= fold_id < n_splits):
                raise ValueError(
                    f"fold_id must be between 0 and {n_splits - 1}, got {fold_id}"
                )
            train_idx, test_idx = all_splits[fold_id]
            chosen_samples = sample_ids[train_idx] if train else sample_ids[test_idx]
            self.dfx = full_df[chosen_samples]
        else:
            self.logger.info(f"Loading pre-split data from {gene_expression_dir}")
            fname = "X_train" if train else "X_test"
            path = self._find_split_file(gene_expression_dir, fname)
            if not path:
                raise FileNotFoundError(
                    f"Expected {fname} with .csv or .tsv extension (optionally "
                    f"compressed) not found in {gene_expression_dir}."
                )
            self.dfx = self._read_expression_file(path)

        # Convert to tensor: training happens at the **sample** level, so transpose
        # the matrix (samples × genes).
        self.data = torch.from_numpy(self.dfx.T.values.astype(np.float32))
        self.genes = list(self.dfx.index)
        self.samples = list(self.dfx.columns)

    def __getitem__(self, idx):
        """
        Return one sample’s expression profile and its identifier.

        Returns
        -------
        profile : torch.Tensor
            Expression vector for the sample (num_genes,).
        sample_id : str
            Identifier of the sample.
        """
        return self.data[idx], self.samples[idx]

    def download(self):
        """No-op (not applicable)."""
        pass

    def _read_expression_file(self, path):
        """Read a CSV or TSV expression matrix with gene IDs as index."""
        suffixes = [s.lower() for s in Path(path).suffixes]

        # Preserve backward compatibility with compressed files (e.g., .csv.gz)
        if ".tsv" in suffixes:
            sep = "\t"
        else:
            # Default to CSV separator even if the extension is missing so pandas
            # can still infer compression and handle the file.
            sep = ","

        return pd.read_csv(path, index_col=0, sep=sep)

    def _find_split_file(self, directory, base_name):
        """Locate a split file with CSV/TSV extension (optionally compressed)."""

        compression_exts = ["", ".gz", ".bz2", ".zip", ".xz"]
        for base_ext in (".csv", ".tsv"):
            for comp_ext in compression_exts:
                candidate = os.path.join(directory, f"{base_name}{base_ext}{comp_ext}")
                if os.path.exists(candidate):
                    return candidate
        return None
