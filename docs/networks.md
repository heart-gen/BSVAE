# Network extraction and latent export

The `bsvae-networks` CLI exposes utilities for deriving gene–gene networks from
trained BSVAE checkpoints and for exporting encoder statistics for downstream
analysis.

## Methods

The extractor implements four complementary strategies:

1. **Decoder-loading similarity (`w_similarity`)** – cosine similarity between
   decoder loading vectors ``W`` (genes × latent factors).
2. **Latent covariance propagation (`latent_cov`)** – propagates the dataset
   average latent variance ``diag(exp(logvar_mean))`` through the decoder,
   producing covariance and correlation matrices.
3. **Conditional independence (`graphical_lasso`)** – reconstructs expression
   ``\hat{X} = Z W^T`` from latent means and fits a Graphical Lasso to estimate
   a sparse precision matrix.
4. **Laplacian-refined (`laplacian`)** – masks decoder similarity by the model's
   ``laplacian_matrix`` buffer when available, preserving prior graph support.

Each method returns an adjacency matrix (genes × genes). Covariance,
correlation, and precision matrices are additionally stored for downstream
visualization.

## CLI usage

### Extract networks
```bash
bsvae-networks extract-networks \
  --model-path results/my_run \
  --dataset data/expression.csv \
  --output-dir results/networks
# optional: --methods latent_cov graphical_lasso laplacian \
  --threshold 0.2
```

Outputs include adjacency matrices saved as **sparse NPZ** files and edge lists
saved as **Parquet** by default. Sparse outputs are enabled with
``--sparse`` (default) and can be disabled with ``--no-sparse`` to fall back to
legacy dense CSV/TSV/NPY outputs. Edge lists are compressed by default via
``--compress`` (disable with ``--no-compress``), with sparsity controlled by
``--threshold``. When ``--threshold 0`` and sparse output is enabled, the
extractor computes an adaptive threshold based on ``--target-sparsity``
(default ``0.01`` = top 1% of edges). Sparse adjacencies can also be quantized
with ``--quantize`` (default ``int8``). Optional heatmaps are generated when
``--heatmaps`` is supplied. By default, the extractor runs the decoder-loading
cosine similarity (``w_similarity``); add additional methods as needed via
``--methods``.

### Export latents
```bash
bsvae-networks export-latents \
  --model-path results/my_run \
  --dataset data/expression.csv \
  --output latents.h5ad
```

Exports per-sample ``mu`` and ``logvar`` either as a tidy CSV (multi-column
with separate ``mu``/``logvar`` blocks) or an ``.h5ad`` with embeddings stored
in ``obsm``.

### Extract modules

Cluster adjacency matrices into discrete gene modules using Leiden or spectral
clustering:

```bash
bsvae-networks extract-modules \
  --adjacency results/networks/w_similarity_adjacency.npz \
  --expr data/expression.csv \
  --output-dir results/modules \
  --cluster-method leiden \
  --resolution 1.0
```

**Automatic resolution selection** finds the optimal Leiden resolution by
maximizing modularity (no ground truth required):

```bash
bsvae-networks extract-modules \
  --adjacency results/networks/w_similarity_adjacency.npz \
  --expr data/expression.csv \
  --output-dir results/modules \
  --resolution-auto
```

**Resolution sweep** runs clustering at multiple resolutions for comparison:

```bash
bsvae-networks extract-modules \
  --adjacency results/networks/w_similarity_adjacency.npz \
  --expr data/expression.csv \
  --output-dir results/modules \
  --resolutions 0.5 0.75 1.0 1.25 1.5
```

See the [BSVAE Networks documentation](bsvae_networks.md#2-extract-gene-modules-wgcna-like)
for full options and output format details.

## Interpreting results

- **High cosine similarity** indicates genes sharing latent modules.
- **Covariance/correlation** highlight uncertainty-propagated co-expression.
- **Precision (Graphical Lasso)** emphasizes conditional dependencies after
  controlling for other genes.
- **Laplacian-refined networks** restrict edges to those supported by the prior
  graph, weighted by decoder similarity.

All functions are available from Python via ``bsvae.networks`` for programmatic
workflows and unit testing.

### GPU acceleration

Network extraction methods (`w_similarity`, `latent_cov`, `laplacian`) use GPU
automatically when available. Module extraction (Leiden, spectral clustering)
runs on CPU. See the [BSVAE Networks documentation](bsvae_networks.md#gpu-acceleration)
for details.

### Signed vs unsigned networks

BSVAE network extraction methods may produce negative edge weights (e.g., cosine
similarity of decoder loadings).

Community detection algorithms such as Leiden require non-negative edge weights.

By default, BSVAE uses a **WGCNA-style signed network** (``adjacency_mode="wgcna-signed"``),
where negative edges are clipped to zero prior to clustering.

This preserves co-activation structure while avoiding artificial antagonistic
modules.

A fully signed community detection mode is planned for a future release.
