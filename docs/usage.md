# Usage Guide

For a complete workflow, see the [Tutorial](tutorial.md).

## End-to-end workflow

1. Train model
2. Extract networks
3. Extract modules
4. Export/analyze latents
5. Simulate benchmark scenarios

## Train

```bash
bsvae-train study1 \
  --dataset data/expression.csv \
  --epochs 120 \
  --n-modules 24 \
  --latent-dim 32
```

### Tune number of modules (K)

`--n-modules` (K) sets the expected number of GMM components/modules.
This should be tuned per dataset.

Quick sweep:

```bash
for k in 8 12 16 24 32; do
  bsvae-train study1_k${k} \
    --dataset data/expression.csv \
    --epochs 60 \
    --n-modules ${k} \
    --latent-dim 32
done
```

Optional downstream check (cluster `mu`):

```bash
bsvae-networks latent-analysis \
  --model-path results/study1_k16 \
  --dataset data/expression.csv \
  --output-dir results/study1_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
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

## Scenario-grid simulation benchmark

Create starter config:

```bash
bsvae-simulate init-config --output sim.yaml
```

Generate full grid:

```bash
bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13
```

Validate grid structure:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Generate one scenario only:

```bash
bsvae-simulate generate-scenario \
  --config sim.yaml \
  --scenario-id S001__confounding-none__n_samples-100__nonlinear_mode-off__overlap_rate-0.0__signal_scale-0.4 \
  --rep 0 \
  --outdir results/sim_pub_v1
```

Method-ready files in each run directory:

- BSVAE input: `expr/features_x_samples.tsv.gz`
- WGCNA input: `expr/samples_x_features.tsv.gz`
- GNVAE input: `expr/features_x_samples.tsv.gz` or `gnvae/fold_*/X_train.tsv.gz`
- Ground truth labels: `truth/modules_hard.csv`
- Canonical path map: `method_inputs.json`
