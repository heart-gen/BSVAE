# Usage Guide

For a complete workflow, see the [Tutorial](tutorial.md).

## End-to-end workflow

1. Train model
2. Extract networks
3. Extract modules
4. Export/analyze latents

## Train

```bash
bsvae-train study1 \
  --dataset data/expression.csv \
  --epochs 120 \
  --n-modules 24 \
  --latent-dim 32
```

## Post-training outputs

`results/study1/` contains:

- `model.pt`
- `specs.json`
- `train_losses.csv`
- checkpoint files `model-<epoch>.pt` (if enabled)

## Network extraction

```bash
bsvae-networks extract-networks \
  --model-path results/study1 \
  --dataset data/expression.csv \
  --output-dir results/study1/networks \
  --methods mu_cosine
```

## Module extraction

```bash
bsvae-networks extract-modules \
  --model-path results/study1 \
  --dataset data/expression.csv \
  --output-dir results/study1/modules \
  --soft-eigengenes \
  --expr data/expression.csv
```

## Latent export and analysis

```bash
bsvae-networks export-latents \
  --model-path results/study1 \
  --dataset data/expression.csv \
  --output results/study1/latents.npz

bsvae-networks latent-analysis \
  --model-path results/study1 \
  --dataset data/expression.csv \
  --output-dir results/study1/latent_analysis \
  --kmeans-k 8 --umap
```
