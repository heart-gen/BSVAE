import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bsvae.models.structured import StructuredFactorVAE
from bsvae.networks.extract_networks import (
    compute_W_similarity_soft,
    compute_jacobian_similarity,
)
from bsvae.networks.utils import transform_adjacency_for_clustering


def test_W_similarity_soft():
    W = torch.randn(12, 4)
    adjacency = compute_W_similarity_soft(W, power=6.0)

    assert adjacency.shape == (12, 12)
    assert np.all(adjacency >= 0)
    assert np.all(adjacency <= 1.0 + 1e-5)
    assert np.allclose(adjacency, adjacency.T, atol=1e-6)


def test_jacobian_similarity():
    torch.manual_seed(0)
    model = StructuredFactorVAE(n_genes=10, n_latent=3)
    x = torch.randn(7, 10)
    loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=False)

    adjacency = compute_jacobian_similarity(model, loader, device=torch.device("cpu"))

    assert adjacency.shape == (10, 10)
    assert np.allclose(adjacency, adjacency.T, atol=1e-6)
    assert np.allclose(np.diag(adjacency), 1.0, atol=1e-4)


def test_soft_threshold_transform():
    adjacency = np.array([[1.0, -0.5], [-0.5, 0.25]], dtype=float)
    transformed = transform_adjacency_for_clustering(
        adjacency, mode="soft-threshold", power=2.0
    )

    expected = np.array([[1.0, 0.0], [0.0, 0.0625]])
    assert np.allclose(transformed, expected)
