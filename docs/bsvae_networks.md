# BSVAE Networks (Detailed)

This page is a detailed companion to `networks.md` for the current `bsvae-networks` CLI.

## Prerequisites

- Trained model directory containing `model.pt` and `specs.json`
- Input matrix in `features x samples` orientation

## Commands

### 1. `extract-networks`

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Notes:

- Default method is `mu_cosine`
- `gamma_knn` requires FAISS (`faiss-cpu`)
- Saved format is sparse NPZ

### 2. `extract-modules`

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules \
  --soft-eigengenes \
  --expr data/expression.csv
```

Notes:

- Extracts GMM `gamma` and hard assignments
- `--soft-eigengenes` requires `--expr`
- `--use-leiden` runs an additional Leiden-based fallback clustering

### 3. `export-latents`

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents.npz
```

### 4. `latent-analysis`

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --gmm-k 8 --tsne
```

## Expected file conventions

- Expression input: first column as feature ID index
- Covariates input: first column or `sample_id` used as index

## Performance notes

- `mu_cosine` is dense similarity followed by top-k sparsification; memory scales with number of features.
- `gamma_knn` is typically more scalable for large feature counts.
