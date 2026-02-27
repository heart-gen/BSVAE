# Quick Start

For a complete workflow, see the [Tutorial](tutorial.md).

## 1. Prepare data

Input must be a matrix with shape `features x samples`.

Example CSV layout:

- Row index: feature IDs (genes, transcripts, proteins, etc.)
- Columns: sample IDs

## 2. Train a model

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

## 3. Extract a network

```bash
bsvae-networks extract-networks \
  --model-path results/pilot_run \
  --dataset data/expression.csv \
  --output-dir results/pilot_run/networks
```

## 4. Extract modules

```bash
bsvae-networks extract-modules \
  --model-path results/pilot_run \
  --dataset data/expression.csv \
  --output-dir results/pilot_run/modules
```

## 5. Export latents

```bash
bsvae-networks export-latents \
  --model-path results/pilot_run \
  --dataset data/expression.csv \
  --output results/pilot_run/latents.npz
```

## 6. Build a simulation grid

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 5 \
  --base-seed 13
```

Validate outputs:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Per scenario replicate, use:

- BSVAE/GNVAE: `expr/features_x_samples.tsv.gz`
- WGCNA: `expr/samples_x_features.tsv.gz`
- Canonical method paths: `method_inputs.json`
