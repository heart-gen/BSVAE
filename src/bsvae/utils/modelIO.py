# bsvae/utils/modelIO.py
import os
import re
import json
import numpy as np
import torch

from bsvae.models import StructuredFactorVAE

MODEL_FILENAME = "model.pt"
META_FILENAME = "specs.json"

# -------------------------
# Save / Load Models
# -------------------------
def save_model(model, directory, metadata=None, filename=MODEL_FILENAME):
    """
    Save a StructuredFactorVAE and corresponding metadata.

    Parameters
    ----------
    model : nn.Module
        The trained model.
    directory : str
        Path to the directory where to save the model and metadata.
    metadata : dict or None
        Additional metadata (e.g. hyperparameters).
        If None, will infer from the model.
    filename : str
        Filename for the saved weights.
    """
    device = next(model.parameters()).device
    model.cpu()

    if metadata is None:
        metadata = dict(
            model_type="StructuredFactorVAE",
            n_genes=model.encoder.n_genes,
            latent_dim=model.encoder.n_latent,
            hidden_dims=model.encoder.hidden_dims,
            dropout=model.encoder.dropout,
            learn_var=model.decoder.learn_var,
            l1_strength=getattr(model, "l1_strength", 1e-3),
            lap_strength=getattr(model, "lap_strength", 1e-4),
        )
    else:
        metadata = dict(metadata)
        metadata["model_type"] = _resolve_model_name(metadata)
        metadata.setdefault("n_genes", model.encoder.n_genes)
        metadata.setdefault("latent_dim", model.encoder.n_latent)
        metadata.setdefault("hidden_dims", model.encoder.hidden_dims)
        metadata.setdefault("dropout", model.encoder.dropout)
        metadata.setdefault("learn_var", model.decoder.learn_var)
        metadata.setdefault("l1_strength", getattr(model, "l1_strength", 1e-3))
        metadata.setdefault("lap_strength", getattr(model, "lap_strength", 1e-4))

    save_metadata(metadata, directory)

    path_to_model = os.path.join(directory, filename)
    torch.save(model.state_dict(), path_to_model)

    model.to(device)


def load_metadata(directory, filename=META_FILENAME):
    """Load the metadata of a training directory."""
    path_to_metadata = os.path.join(directory, filename)
    with open(path_to_metadata) as metadata_file:
        return json.load(metadata_file)


def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Save metadata dictionary as JSON file."""
    path_to_metadata = os.path.join(directory, filename)
    os.makedirs(directory, exist_ok=True)
    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


def load_model(directory, is_gpu=True, filename=MODEL_FILENAME):
    """
    Load a trained StructuredFactorVAE.

    Parameters
    ----------
    directory : str
        Directory where model + metadata are stored.
    is_gpu : bool
        Whether to map to GPU if available.
    filename : str
        Model weights file (default "model.pt").
    """
    device = torch.device("cuda" if torch.cuda.is_available() and is_gpu else "cpu")

    metadata = load_metadata(directory)
    path_to_model = os.path.join(directory, filename)

    model = _get_model(metadata, device, path_to_model)
    return model


def _resolve_model_name(metadata):
    """Resolve the model identifier used in metadata to a canonical name."""
    model_type = metadata.get("model_type") or metadata.get("model")
    if model_type is None:
        return "StructuredFactorVAE"

    if not isinstance(model_type, str):
        raise ValueError(f"Invalid model_type: {model_type}")

    if model_type.lower() in {"structuredfactorvae", "sfvae"}:
        return "StructuredFactorVAE"

    raise ValueError(f"Unknown model_type: {model_type}")


def load_checkpoints(directory, is_gpu=True):
    """Load all checkpointed models (saved as model-<epoch>.pt)."""
    checkpoints = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            results = re.search(r'.*?-([0-9].*?).pt', filename)
            if results is not None:
                epoch_idx = int(results.group(1))
                model = load_model(root, is_gpu=is_gpu, filename=filename)
                checkpoints.append((epoch_idx, model))

    return checkpoints


def _get_model(metadata, device, path_to_model):
    """Instantiate a StructuredFactorVAE from metadata + load weights."""
    model_type = _resolve_model_name(metadata)

    state_dict = torch.load(path_to_model, map_location=device)

    model = StructuredFactorVAE(
        n_genes=metadata["n_genes"],
        n_latent=metadata["latent_dim"],
        hidden_dims=metadata.get("hidden_dims"),
        dropout=metadata.get("dropout", 0.1),
        learn_var=metadata.get("learn_var", False),
    ).to(device)

    # store reg strengths for reproducibility
    model.l1_strength = metadata.get("l1_strength", 1e-3)
    model.lap_strength = metadata.get("lap_strength", 1e-4)

    # Backwards compatibility: older checkpoints may contain a Laplacian
    # buffer that is not registered in newly instantiated models unless a
    # Laplacian matrix is provided at construction time.
    if "laplacian_matrix" in state_dict:
        laplacian = state_dict["laplacian_matrix"]
        if "laplacian_matrix" not in model._buffers:
            # Remove placeholder attribute from __init__ so the buffer can be registered
            if hasattr(model, "laplacian_matrix"):
                delattr(model, "laplacian_matrix")
            model.register_buffer("laplacian_matrix", laplacian)
        else:
            model.laplacian_matrix = laplacian

    model.load_state_dict(state_dict)
    model.eval()
    return model


# -------------------------
# Save / Load NumPy Arrays
# -------------------------
def numpy_serialize(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def save_np_arrays(arrays, directory, filename):
    """Save dictionary of arrays in JSON format."""
    save_metadata(arrays, directory, filename=filename, default=numpy_serialize)


def load_np_arrays(directory, filename):
    """Load dictionary of arrays from JSON format into numpy arrays."""
    arrays = load_metadata(directory, filename=filename)
    return {k: np.array(v) for k, v in arrays.items()}
