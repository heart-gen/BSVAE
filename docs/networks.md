# Network And Latent Workflows

This is the canonical reference for `bsvae-networks`.

## Prerequisites

- a trained model directory containing `model.pt` and `specs.json`
- the same expression matrix orientation used for training: `features x samples`

## Extract Networks

`extract-networks` builds sparse feature-feature adjacency matrices from learned latents.

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Methods:

- `mu_cosine`: cosine similarity among latent means, sparsified to top-k neighbors
- `gamma_knn`: FAISS-based kNN graph built from normalized GMM soft assignments

Outputs:

- `mu_cosine_adjacency.npz`
- `gamma_knn_adjacency.npz`

## Extract Modules

`extract-modules` writes soft and hard assignments from the fitted GMM.

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules
```

Primary outputs:

- `gamma.npz`
- `hard_assignments.npz`

Add eigengenes:

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Post-processing (on by default) mirrors WGCNA: a confidence cutoff (`--min-gamma`) leaves uncertain features unassigned (grey, label `-1`), undersized modules are dissolved (`--min-module-size`), and modules whose eigengenes correlate above `1 - --merge-cut-height` are merged. Pass `--no-postprocess` for raw `argmax(γ)`.

Outputs:

- `modules.csv` (`feature_id, module, module_label`), `eigengenes.csv`, `kME.csv`, `merge_log.json`, `gamma_merged.npz`
- `gamma.npz`, `hard_assignments.npz` (raw)
- `soft_eigengenes.csv` with `--soft-eigengenes`
- `leiden_modules.csv` with `--use-leiden`
- `gamma_gene.npz`, `hard_assignments_gene.npz` and `gene_divergent.csv` (genes whose isoforms land in two or more modules) with `--aggregate-to-gene --tx2gene`

## Export Latents

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents
```

This writes `<output>.npz` containing:

- `mu`
- `logvar`
- `gamma`
- `feature_ids`

## Latent Analysis

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --kmeans-k 10 \
  --umap
```

Possible outputs:

- `latent_mu.csv`
- `latent_logvar.csv`
- `latent_clusters.csv`
- `latent_embeddings.csv`
- `latent_covariate_correlations.csv`

## Performance Notes

- `mu_cosine` scales with the number of features and can be expensive on large matrices.
- `gamma_knn` requires `faiss-cpu`.
- For memory-constrained runs, reduce `--batch-size` and consider `--no-cuda`.
