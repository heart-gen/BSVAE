"""Utilities for exporting encoder latents.

The functions here are intentionally lightweight and unit-test friendly. They
handle device placement, batching, and optional AnnData/HDF export for
integration with downstream single-cell or bulk workflows.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_latents(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    disable_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract latent mean (``mu``) and log-variance (``logvar``) for a dataset.

    Parameters
    ----------
    model
        Trained model containing an ``encoder`` attribute.
    dataloader
        Iterable yielding ``(expression, sample_id)`` pairs where expression has
        shape ``(batch, G)``.
    device
        Torch device. Defaults to CUDA if available.
    disable_progress
        Placeholder flag to allow tqdm in the future without changing the API.

    Returns
    -------
    mu : np.ndarray
        Array of shape ``(n_samples, K)``.
    logvar : np.ndarray
        Array of shape ``(n_samples, K)``.
    sample_ids : list of str
        Identifiers provided by the dataloader.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mu_list: List[np.ndarray] = []
    logvar_list: List[np.ndarray] = []
    sample_ids: List[str] = []

    with torch.no_grad():
        for batch, ids in dataloader:
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
            # DataLoader with strings returns a tuple; ensure list of str
            if isinstance(ids, (list, tuple)):
                sample_ids.extend([str(i) for i in ids])
            else:
                sample_ids.extend([str(ids)])

    mu_arr = np.concatenate(mu_list, axis=0)
    logvar_arr = np.concatenate(logvar_list, axis=0)
    return mu_arr, logvar_arr, sample_ids


def save_latents(
    mu: np.ndarray,
    logvar: np.ndarray,
    sample_ids: Sequence[str],
    output_path: str,
    format: Optional[str] = None,
) -> None:
    """Save latent statistics to ``.csv`` or ``.h5ad``.

    Parameters
    ----------
    mu
        Latent mean array ``(n_samples, K)``.
    logvar
        Latent log-variance array ``(n_samples, K)``.
    sample_ids
        Identifiers for each sample.
    output_path
        Destination path. Extension is inferred when ``format`` is ``None``.
    format
        Override file format (``"csv"`` or ``"h5ad"``).
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format is None:
        ext = path.suffix.lower().lstrip(".")
        format = ext or "csv"

    mu_df = pd.DataFrame(mu, index=sample_ids)
    mu_df.index.name = "sample_id"
    logvar_df = pd.DataFrame(logvar, index=sample_ids)
    logvar_df.index.name = "sample_id"

    if format.lower() == "csv":
        combined = pd.concat({"mu": mu_df, "logvar": logvar_df}, axis=1)
        combined.to_csv(path)
    elif format.lower() == "h5ad":
        try:
            import anndata as ad

            adata = ad.AnnData(mu_df)
            adata.obsm["mu"] = mu_df.values
            adata.obsm["logvar"] = logvar_df.values
            adata.write_h5ad(path)
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("anndata is required to write .h5ad files") from exc
    else:
        raise ValueError(f"Unsupported format: {format}")


def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Move a tensor to ``device`` if provided."""

    if device is None:
        return tensor
    return tensor.to(device)


def collate_batches(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Stack a list of tensors along the batch dimension."""

    return torch.cat(list(tensors), dim=0)


__all__ = ["extract_latents", "save_latents", "to_device", "collate_batches"]
