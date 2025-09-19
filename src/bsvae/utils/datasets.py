import os
import abc
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
       Provide `gene_expression_filename` (CSV: genes × samples).
       -> Creates reproducible 10-fold splits.
    2. Pre-split Mode:
       Provide `gene_expression_dir` containing 'X_train.csv' and 'X_test.csv'.

    Parameters
    ----------
    gene_expression_filename : str, optional
        Path to CSV file with full expression matrix.
    gene_expression_dir : str, optional
        Directory containing 'X_train.csv' and 'X_test.csv'.
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
            full_df = pd.read_csv(gene_expression_filename, index_col=0)
            kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
            all_splits = list(kf.split(full_df))
            if not (0 <= fold_id < 10):
                raise ValueError(f"fold_id must be between 0 and 9, got {fold_id}")
            train_idx, test_idx = all_splits[fold_id]
            self.dfx = full_df.iloc[train_idx] if train else full_df.iloc[test_idx]
        else:
            self.logger.info(f"Loading pre-split data from {gene_expression_dir}")
            fname = "X_train.csv" if train else "X_test.csv"
            path = os.path.join(gene_expression_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Expected {path} not found.")
            self.dfx = pd.read_csv(path, index_col=0)

        # Convert to tensor
        self.data = torch.from_numpy(self.dfx.values.astype(np.float32))
        self.genes = list(self.dfx.index)
        self.samples = list(self.dfx.columns)

    def __getitem__(self, idx):
        """
        Return one gene’s expression profile and its identifier.

        Returns
        -------
        profile : torch.Tensor
            Expression vector for the gene (num_samples,).
        gene_id : str
            Identifier of the gene (e.g., Ensembl ID).
        """
        return self.data[idx], self.genes[idx]

    def download(self):
        """No-op (not applicable)."""
        pass
