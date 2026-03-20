"""
Tests for isoform-level clustering machinery:
  - hierarchical_loss() correctness
  - _aggregate_gamma_to_gene() in networks CLI
"""

import numpy as np
import pandas as pd
import pytest
import torch

from bsvae.models.losses import hierarchical_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tx2gene(tmp_path, mapping):
    df = pd.DataFrame(
        [(tx, g) for tx, g in mapping.items()],
        columns=["transcript_id", "gene_id"],
    )
    path = str(tmp_path / "tx2gene.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


# ---------------------------------------------------------------------------
# hierarchical_loss — unit tests
# ---------------------------------------------------------------------------

def test_hierarchical_loss_identical_isoforms_zero():
    """Identical μ vectors for isoforms of the same gene → loss == 0."""
    mu = torch.zeros(4, 8)
    gene_groups = {"geneA": [0, 1, 2]}
    loss = hierarchical_loss(mu, gene_groups)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hierarchical_loss_positive_for_different_isoforms():
    """Different μ vectors → positive loss."""
    D = 4
    mu = torch.zeros(4, D)
    mu[0] = torch.ones(D)
    gene_groups = {"geneA": [0, 1]}
    loss = hierarchical_loss(mu, gene_groups)
    assert loss.item() > 0.0


def test_hierarchical_loss_scales_with_distance():
    """Larger μ separation → larger loss."""
    D = 4
    gene_groups = {"geneA": [0, 1]}

    mu_near = torch.zeros(2, D)
    mu_near[0] = torch.ones(D) * 0.1
    loss_near = hierarchical_loss(mu_near, gene_groups).item()

    mu_far = torch.zeros(2, D)
    mu_far[0] = torch.ones(D) * 10.0
    loss_far = hierarchical_loss(mu_far, gene_groups).item()

    assert loss_far > loss_near


def test_hierarchical_loss_no_batch_overlap_returns_zero():
    """Gene isoforms not present in the batch → loss == 0."""
    mu = torch.randn(6, 4)
    # gene_groups refers to dataset indices 0 and 1,
    # but the batch contains features at dataset indices 2–7
    feature_idx = torch.tensor([2, 3, 4, 5, 6, 7])
    gene_groups = {"geneA": [0, 1]}
    loss = hierarchical_loss(mu, gene_groups, feature_idx_in_batch=feature_idx)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hierarchical_loss_partial_overlap_skipped():
    """Only 1 of 2 isoforms present in batch → gene skipped, loss == 0."""
    mu = torch.randn(4, 4)
    # Dataset feature 0 is in batch (position 0); feature 1 is absent
    feature_idx = torch.tensor([0, 5, 6, 7])
    gene_groups = {"geneA": [0, 1]}
    loss = hierarchical_loss(mu, gene_groups, feature_idx_in_batch=feature_idx)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hierarchical_loss_with_feature_idx_mapping():
    """feature_idx_in_batch correctly identifies which batch rows are isoforms."""
    D = 4
    mu = torch.zeros(4, D)
    mu[0] = torch.ones(D)   # batch position 0 = dataset feature 10
    mu[1] = torch.zeros(D)  # batch position 1 = dataset feature 11
    feature_idx = torch.tensor([10, 11, 12, 13])
    gene_groups = {"geneA": [10, 11]}
    loss = hierarchical_loss(mu, gene_groups, feature_idx_in_batch=feature_idx)
    assert loss.item() > 0.0


def test_hierarchical_loss_deduplicates_repeated_feature_indices_fast_path():
    """Repeated dataset indices should not create synthetic isoform pairs (fast path)."""
    D = 2
    mu = torch.zeros(3, D)
    mu[0] = torch.tensor([3.0, 4.0])  # dataset feature 10 (duplicated in batch)
    feature_idx = torch.tensor([10, 10, 12])
    gene_groups = {"geneA": [10, 12]}
    idx_to_gene = {10: "geneA", 12: "geneA"}

    loss = hierarchical_loss(
        mu,
        gene_groups,
        feature_idx_in_batch=feature_idx,
        idx_to_gene=idx_to_gene,
    )

    # Distinct isoforms for geneA are dataset features 10 and 12 only.
    # Expected L2 between mu[0] and mu[2] = 5.
    assert loss.item() == pytest.approx(5.0, abs=1e-6)


def test_hierarchical_loss_deduplicates_repeated_feature_indices_fallback():
    """Repeated dataset indices should not create synthetic isoform pairs (fallback path)."""
    D = 2
    mu = torch.zeros(3, D)
    mu[0] = torch.tensor([3.0, 4.0])  # dataset feature 10 (duplicated in batch)
    feature_idx = torch.tensor([10, 10, 12])
    gene_groups = {"geneA": [10, 12]}

    loss = hierarchical_loss(mu, gene_groups, feature_idx_in_batch=feature_idx)

    # Distinct isoforms for geneA are dataset features 10 and 12 only.
    # Expected L2 between mu[0] and mu[2] = 5.
    assert loss.item() == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize("idx_to_gene", [None, {10: "geneA"}])
def test_hierarchical_loss_repeated_single_isoform_is_skipped(idx_to_gene):
    """Repeated copies of one isoform should not trigger hierarchical loss."""
    mu = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    feature_idx = torch.tensor([10, 10])
    gene_groups = {"geneA": [10, 12]}

    loss = hierarchical_loss(
        mu,
        gene_groups,
        feature_idx_in_batch=feature_idx,
        idx_to_gene=idx_to_gene,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hierarchical_loss_empty_groups_returns_zero():
    """Empty gene_groups → zero loss (no pairs to compare)."""
    mu = torch.randn(4, 8)
    loss = hierarchical_loss(mu, {})
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_hierarchical_loss_multiple_genes_averages():
    """Loss is the mean over genes: geneA contributes 0, geneB contributes sqrt(2)."""
    D = 2
    mu = torch.zeros(4, D)
    # geneA: isoforms 0,1 are identical → contribution = 0
    # geneB: isoforms 2,3 are orthogonal unit vectors → L2 = sqrt(2)
    mu[2] = torch.tensor([1.0, 0.0])
    mu[3] = torch.tensor([0.0, 1.0])
    gene_groups = {"geneA": [0, 1], "geneB": [2, 3]}
    loss = hierarchical_loss(mu, gene_groups)
    expected = (0.0 + 2 ** 0.5) / 2
    assert loss.item() == pytest.approx(expected, abs=1e-5)


def test_hierarchical_loss_is_differentiable():
    """Loss supports backprop through μ."""
    mu = torch.randn(4, 8, requires_grad=True)
    gene_groups = {"geneA": [0, 1], "geneB": [2, 3]}
    loss = hierarchical_loss(mu, gene_groups)
    loss.backward()
    assert mu.grad is not None
    assert mu.grad.shape == mu.shape


# ---------------------------------------------------------------------------
# _aggregate_gamma_to_gene
# ---------------------------------------------------------------------------

def test_aggregate_gamma_to_gene_multi_isoform(tmp_path):
    """Multi-isoform gene γ is averaged; rows renormalise to 1."""
    from bsvae.cli.networks import _aggregate_gamma_to_gene

    gamma = np.array([
        [0.8, 0.2],   # tx1 → geneA
        [0.4, 0.6],   # tx2 → geneA
        [0.1, 0.9],   # tx3 → geneB (singleton)
    ], dtype=np.float32)
    feature_ids = ["tx1", "tx2", "tx3"]
    tx_path = _write_tx2gene(tmp_path, {"tx1": "geneA", "tx2": "geneA", "tx3": "geneB"})

    gamma_gene, gene_ids = _aggregate_gamma_to_gene(gamma, feature_ids, tx_path)

    assert set(gene_ids) == {"geneA", "geneB"}
    gA_idx = gene_ids.index("geneA")
    # mean([0.8,0.2], [0.4,0.6]) = [0.6, 0.4]
    np.testing.assert_allclose(gamma_gene[gA_idx], [0.6, 0.4], atol=1e-6)
    # All rows sum to 1
    np.testing.assert_allclose(gamma_gene.sum(axis=1), np.ones(len(gene_ids)), atol=1e-6)


def test_aggregate_gamma_to_gene_singleton_passthrough(tmp_path):
    """Single-isoform gene passes through unchanged (after renormalisation)."""
    from bsvae.cli.networks import _aggregate_gamma_to_gene

    gamma = np.array([[0.3, 0.7]], dtype=np.float32)
    feature_ids = ["tx1"]
    tx_path = _write_tx2gene(tmp_path, {"tx1": "geneA"})

    gamma_gene, gene_ids = _aggregate_gamma_to_gene(gamma, feature_ids, tx_path)

    assert gene_ids == ["geneA"]
    np.testing.assert_allclose(gamma_gene[0], [0.3, 0.7], atol=1e-6)


def test_aggregate_gamma_to_gene_unmapped_feature(tmp_path):
    """Features absent from tx2gene fall back to their own ID."""
    from bsvae.cli.networks import _aggregate_gamma_to_gene

    gamma = np.array([
        [0.7, 0.3],   # tx1 → geneA (in tx2gene)
        [0.2, 0.8],   # orphan → not in tx2gene, maps to "orphan"
    ], dtype=np.float32)
    feature_ids = ["tx1", "orphan"]
    tx_path = _write_tx2gene(tmp_path, {"tx1": "geneA"})

    gamma_gene, gene_ids = _aggregate_gamma_to_gene(gamma, feature_ids, tx_path)

    assert "geneA" in gene_ids
    assert "orphan" in gene_ids
    orphan_idx = gene_ids.index("orphan")
    np.testing.assert_allclose(gamma_gene[orphan_idx], [0.2, 0.8], atol=1e-6)


def test_aggregate_gamma_to_gene_row_sums_to_one(tmp_path):
    """Rows always sum to 1 after aggregation, regardless of input values."""
    from bsvae.cli.networks import _aggregate_gamma_to_gene

    K = 5
    # Uniform γ across 3 isoforms — mean is still uniform, sums to 1
    gamma = np.ones((3, K), dtype=np.float32) / K
    feature_ids = ["tx1", "tx2", "tx3"]
    tx_path = _write_tx2gene(tmp_path, {"tx1": "geneA", "tx2": "geneA", "tx3": "geneA"})

    gamma_gene, gene_ids = _aggregate_gamma_to_gene(gamma, feature_ids, tx_path)

    assert len(gene_ids) == 1
    np.testing.assert_allclose(gamma_gene.sum(axis=1), [1.0], atol=1e-6)


def test_aggregate_gamma_output_shape(tmp_path):
    """Output shape is (n_unique_genes, K)."""
    from bsvae.cli.networks import _aggregate_gamma_to_gene

    K = 4
    # 6 transcripts → 3 genes (2 isoforms each)
    mapping = {
        "tx0": "gA", "tx1": "gA",
        "tx2": "gB", "tx3": "gB",
        "tx4": "gC", "tx5": "gC",
    }
    gamma = np.random.dirichlet(np.ones(K), size=6).astype(np.float32)
    feature_ids = list(mapping.keys())
    tx_path = _write_tx2gene(tmp_path, mapping)

    gamma_gene, gene_ids = _aggregate_gamma_to_gene(gamma, feature_ids, tx_path)

    assert gamma_gene.shape == (3, K)
    assert len(gene_ids) == 3
    assert set(gene_ids) == {"gA", "gB", "gC"}
