"""Network utility functions."""

from __future__ import annotations

import numpy as np


def transform_adjacency_for_clustering(adjacency: np.ndarray, mode: str = "wgcna-signed", **kwargs) -> np.ndarray:
    """Transform adjacency matrices for community detection.

    Parameters
    ----------
    adjacency : np.ndarray
        Square adjacency matrix (G × G), may contain negative values.
    mode : {"wgcna-signed", "signed", "soft-threshold"}
        - "wgcna-signed" (default): clip negative values to zero
          (WGCNA-style signed network).
        - "signed": preserve negative edges (NOT supported by Leiden).
        - "soft-threshold": clip negatives to zero and apply power transform.

    Returns
    -------
    np.ndarray
        Transformed adjacency matrix.

    Raises
    ------
    ValueError
        If mode is unknown.
    """

    if mode == "wgcna-signed":
        return np.maximum(adjacency, 0.0)

    if mode == "signed":
        return adjacency

    if mode == "soft-threshold":
        power = kwargs.get("power", 6.0)
        return np.maximum(adjacency, 0.0) ** power

    raise ValueError(f"Unknown adjacency_mode: {mode}")


__all__ = ["transform_adjacency_for_clustering"]
