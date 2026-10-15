import numpy as np
import pytest

from bsvae.networks.extract_networks import (
    method_a_cosine,
    save_adjacency_npz,
    load_adjacency_npz,
)
from bsvae.networks.utils import transform_adjacency_for_clustering


def _two_block_mu(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.05, size=(10, 4)) + np.array([1.0, 0.0, 0.0, 0.0])
    b = rng.normal(0, 0.05, size=(10, 4)) + np.array([0.0, 1.0, 0.0, 0.0])
    return np.vstack([a, b]).astype(np.float32)


def test_method_a_cosine_shape_symmetry_and_blocks():
    mu = _two_block_mu()
    adj = method_a_cosine(mu, top_k=5)
    assert adj.shape == (20, 20)
    dense = adj.toarray()
    assert np.allclose(dense, dense.T, atol=1e-6)
    assert np.all(np.diag(dense) == 0)
    # Within-block edges should dominate cross-block edges.
    within = dense[:10, :10].sum() + dense[10:, 10:].sum()
    across = dense[:10, 10:].sum() + dense[10:, :10].sum()
    assert within > across


def test_method_a_cosine_top_k_bounds():
    mu = _two_block_mu()
    adj = method_a_cosine(mu, top_k=3)
    # Before symmetrisation each row has at most k edges; after max(A, A.T)
    # degree can exceed k but must stay bounded by F-1.
    deg = (adj.toarray() > 0).sum(axis=1)
    assert deg.max() <= 19
    assert method_a_cosine(mu, top_k=0).nnz == 0
    assert method_a_cosine(mu[:1], top_k=5).shape == (1, 1)


def test_adjacency_npz_roundtrip(tmp_path):
    mu = _two_block_mu()
    adj = method_a_cosine(mu, top_k=4)
    ids = [f"f{i}" for i in range(20)]
    path = tmp_path / "adj.npz"
    save_adjacency_npz(adj, str(path), ids)
    loaded, loaded_ids = load_adjacency_npz(str(path))
    assert loaded_ids == ids
    assert np.allclose(loaded.toarray(), adj.toarray(), atol=1e-6)


def test_soft_threshold_transform():
    adjacency = np.array([[1.0, -0.5], [-0.5, 0.25]], dtype=float)
    transformed = transform_adjacency_for_clustering(
        adjacency, mode="soft-threshold", power=2.0
    )
    expected = np.array([[1.0, 0.0], [0.0, 0.0625]])
    assert np.allclose(transformed, expected)
