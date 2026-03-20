# Quick Start

For a longer guided workflow, see the [Tutorial](tutorial.md). For flag-by-flag reference, see the [CLI Reference](cli.md).

## 1. Prepare Data

Input must be an expression matrix in `features x samples` orientation.

- rows are features
- columns are samples
- the first column is the feature ID index for CSV or TSV files

## 2. Train A Small Model

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

Expected outputs in `results/pilot_run/`:

- `model.pt`
- `specs.json`
- `train_losses.csv`

## 3. Recommended: Tune The Number Of Modules

`--n-modules` sets the number of Gaussian-mixture components. For most real datasets, use `bsvae-sweep-k` before a production run.

```bash
bsvae-sweep-k sweep_pilot \
  --dataset data/expression.csv \
  --k-grid 6,8,12,16 \
  --sweep-epochs 30 \
  --stability-reps 3 \
  --val-frac 0.1
```

This writes sweep artifacts under `results/sweep_pilot/sweep_k/` and retrains the selected model under `results/sweep_pilot/final_k<K>/`.

## 4. Extract Networks

```bash
bsvae-networks extract-networks \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/networks \
  --methods mu_cosine
```

## 5. Extract Modules

`extract-modules` always needs the training dataset, and it needs `--expr` when you want eigengenes.

```bash
bsvae-networks extract-modules \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Expected outputs:

- `gamma.npz`
- `hard_assignments.npz`
- `soft_eigengenes.csv` when requested

## 6. Export Latents

```bash
bsvae-networks export-latents \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output results/sweep_pilot/final_k12/latents
```

This writes `latents.npz` with `mu`, `logvar`, `gamma`, and `feature_ids`.

## 7. Run Latent Analysis

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/latent_analysis \
  --kmeans-k 12 \
  --umap
```

## 8. Build A Simulation Grid

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
