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

Outputs include adjacency matrices (CSV/TSV or NPY), edge lists filtered by the
chosen threshold, and optional heatmaps when ``--heatmaps`` is supplied. By
default, the extractor runs the decoder-loading cosine similarity
(``w_similarity``); add additional methods as needed via ``--methods``.

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

## Interpreting results

- **High cosine similarity** indicates genes sharing latent modules.
- **Covariance/correlation** highlight uncertainty-propagated co-expression.
- **Precision (Graphical Lasso)** emphasizes conditional dependencies after
  controlling for other genes.
- **Laplacian-refined networks** restrict edges to those supported by the prior
  graph, weighted by decoder similarity.

All functions are available from Python via ``bsvae.networks`` for programmatic
workflows and unit testing.
