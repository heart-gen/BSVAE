"""
Model persistence for GMMModuleVAE.

save_model / load_model handle GMMModuleVAE and write a specs.json
that records all architecture hyperparameters needed to reconstruct
the model without any additional arguments.
"""

import json
import os
import re
from typing import Optional

import numpy as np
import torch

from bsvae.models import GMMModuleVAE

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def save_metadata(metadata: dict, directory: str, filename: str = META_FILENAME, **kwargs):
    """Persist metadata dict as a JSON file."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_metadata(directory: str, filename: str = META_FILENAME) -> dict:
    """Load metadata JSON from directory."""
    path = os.path.join(directory, filename)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GMMModuleVAE save / load
# ---------------------------------------------------------------------------

def save_model(
    model: GMMModuleVAE,
    directory: str,
    metadata: Optional[dict] = None,
    filename: str = MODEL_FILENAME,
):
    """
    Save a GMMModuleVAE (weights + specs.json).

    Parameters
    ----------
    model : GMMModuleVAE
    directory : str
    metadata : dict or None
        If None, inferred from model attributes.
    filename : str
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        metadata = _metadata_from_model(model)
    else:
        metadata = dict(metadata)
        metadata.setdefault("model_type", "GMMModuleVAE")
        metadata.setdefault("n_features", model.encoder.n_features)
        metadata.setdefault("n_latent", model.n_latent)
        metadata.setdefault("n_modules", model.n_modules)
        metadata.setdefault("hidden_dims", model.encoder.hidden_dims)
        metadata.setdefault("dropout", model.encoder.dropout)
        metadata.setdefault("use_batch_norm", model.encoder.use_batch_norm)
        metadata.setdefault("sigma_min", model.gmm_prior.sigma_min)

    save_metadata(metadata, directory)
    torch.save(model.state_dict(), os.path.join(directory, filename))
    model.to(device)


def load_model(
    directory: str,
    is_gpu: bool = True,
    filename: str = MODEL_FILENAME,
) -> GMMModuleVAE:
    """
    Load a trained GMMModuleVAE from a directory.

    Parameters
    ----------
    directory : str
    is_gpu : bool
    filename : str

    Returns
    -------
    model : GMMModuleVAE (eval mode)
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")
    metadata = load_metadata(directory)
    path_to_model = os.path.join(directory, filename)
    return _get_model(metadata, device, path_to_model)


def load_checkpoints(directory: str, is_gpu: bool = True) -> list:
    """Load all checkpointed models (model-<epoch>.pt)."""
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for fname in filenames:
            m = re.search(r"model-([0-9]+)\.pt$", fname)
            if m:
                epoch_idx = int(m.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=fname)
                checkpoints.append((epoch_idx, model))
    return checkpoints


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metadata_from_model(model: GMMModuleVAE) -> dict:
    return dict(
        model_type="GMMModuleVAE",
        n_features=model.encoder.n_features,
        n_latent=model.n_latent,
        n_modules=model.n_modules,
        hidden_dims=model.encoder.hidden_dims,
        dropout=model.encoder.dropout,
        use_batch_norm=model.encoder.use_batch_norm,
        sigma_min=model.gmm_prior.sigma_min,
    )


def _get_model(metadata: dict, device: torch.device, path_to_model: str) -> GMMModuleVAE:
    """Reconstruct a GMMModuleVAE from metadata and load its weights."""
    use_batch_norm = metadata.get("use_batch_norm", True)
    if use_batch_norm is None:
        # Infer from state dict
        state_dict = torch.load(path_to_model, map_location=device, weights_only=True)
        use_batch_norm = _infer_encoder_batch_norm(state_dict)
    else:
        state_dict = None

    model = GMMModuleVAE(
        n_features=metadata["n_features"],
        n_latent=metadata["n_latent"],
        n_modules=metadata["n_modules"],
        hidden_dims=metadata.get("hidden_dims"),
        dropout=metadata.get("dropout", 0.1),
        use_batch_norm=use_batch_norm,
        sigma_min=metadata.get("sigma_min", 0.3),
    ).to(device)

    if state_dict is None:
        state_dict = torch.load(path_to_model, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _infer_encoder_batch_norm(state_dict: dict) -> bool:
    return any(
        k.startswith("encoder.encoder") and k.endswith("running_mean")
        for k in state_dict
    )


# ---------------------------------------------------------------------------
# Numpy array helpers (unchanged from original)
# ---------------------------------------------------------------------------

def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.item()
    raise TypeError(f"Unknown type: {type(obj)}")


def save_np_arrays(arrays: dict, directory: str, filename: str):
    save_metadata(arrays, directory, filename=filename, default=numpy_serialize)


def load_np_arrays(directory: str, filename: str) -> dict:
    arrays = load_metadata(directory, filename=filename)
    return {k: np.array(v) for k, v in arrays.items()}
