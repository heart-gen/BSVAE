# Network and Latent Workflows

`bsvae-networks` provides post-training utilities for `GMMModuleVAE` checkpoints.

## Network extraction

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

### Supported methods

- `mu_cosine`: top-k cosine neighbors in latent mean (`mu`) space
- `gamma_knn`: FAISS HNSW kNN in soft-assignment (`gamma`) space

Outputs are sparse NPZ adjacency files named `<method>_adjacency.npz`.

## Module extraction

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules
```

Primary outputs:

- `gamma.npz`
- `hard_assignments.npz`
- optional `soft_eigengenes.csv` with `--soft-eigengenes --expr`
- optional `leiden_modules.csv` with `--use-leiden`

## Latent export

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents.npz
```

NPZ includes:

- `mu`: latent means
- `logvar`: latent log variances
- `gamma`: soft module assignments
- `feature_ids`

## Latent analysis

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --kmeans-k 10 --umap
```

Saved files may include:

- `latent_mu.csv`
- `latent_logvar.csv`
- `latent_clusters.csv`
- `latent_embeddings.csv`
- `latent_covariate_correlations.csv`
