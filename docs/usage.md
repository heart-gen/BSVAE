# Usage Guide

This page describes the current end-to-end workflow for typical CLI users.

## Recommended Workflow

1. Prepare a `features x samples` matrix.
2. Run `bsvae-sweep-k` to choose the number of modules.
3. Use the retrained `final_k<K>` model for downstream analysis.
4. Extract networks, module assignments, and latent outputs.
5. Use `bsvae-simulate` when you need synthetic benchmarking.

## Training

Direct training:

```bash
bsvae-train study1 \
  --dataset data/expression.csv \
  --epochs 120 \
  --n-modules 24 \
  --latent-dim 32
```

Recommended model-selection flow:

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

This creates:

- sweep metrics in `results/sweep1/sweep_k/`
- a final retrained model in `results/sweep1/final_k<K>/`

## Post-Training Outputs

Training directories contain:

- `model.pt`
- `specs.json`
- `train_losses.csv`
- `model-<epoch>.pt` when checkpointing is enabled

Sweep directories additionally contain:

- `sweep_results.csv`
- `sweep_summary.json`
- per-K replicate subdirectories

## Network Extraction

```bash
bsvae-networks extract-networks \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/networks \
  --methods mu_cosine gamma_knn
```

Use `mu_cosine` when you want a graph based on latent-mean similarity. Use `gamma_knn` when you want a graph based on GMM soft assignments and have `faiss-cpu` available.

## Module Extraction

```bash
bsvae-networks extract-modules \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Outputs:

- `gamma.npz`
- `hard_assignments.npz`
- `soft_eigengenes.csv` when requested

Optional extras:

- `--use-leiden` to write `leiden_modules.csv`
- `--aggregate-to-gene --tx2gene` to write gene-level assignment files

## Latent Export And Analysis

Export:

```bash
bsvae-networks export-latents \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output results/sweep1/final_k16/latents
```

Analyze:

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

## Simulation Workflow

Generate one synthetic dataset:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv
```

Create a scenario grid:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13
```

Validate the grid:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Each replicate contains method-ready files for BSVAE and comparator pipelines.
