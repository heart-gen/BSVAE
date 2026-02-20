# Tutorial

This tutorial is a complete workflow for BSVAE using the current CLI.

## 1. Minimal Run

Goal: confirm install, data format, and training loop.

Input matrix requirements:

- shape: `features x samples`
- row index: feature IDs
- columns: sample IDs

Run:

```bash
bsvae-train tutorial_min \
  --dataset data/expression.csv \
  --epochs 5 \
  --batch-size 64 \
  --n-modules 8 \
  --latent-dim 12
```

Sanity checks:

- `results/tutorial_min/model.pt` exists
- `results/tutorial_min/specs.json` exists
- `results/tutorial_min/train_losses.csv` exists

## 2. Production Run

Goal: run a longer job with checkpointing and explicit settings.

```bash
bsvae-train study_prod \
  --dataset data/expression.csv \
  --epochs 150 \
  --batch-size 256 \
  --lr 5e-4 \
  --checkpoint-every 10 \
  --n-modules 24 \
  --latent-dim 32 \
  --hidden-dims "[512, 256, 128]" \
  --beta 1.0 \
  --free-bits 0.5
```

Recommended checks during/after run:

- loss is finite in logs
- checkpoints appear as `model-<epoch>.pt`
- final artifacts exist in `results/study_prod/`

## 3. Post-Training Analysis

### 3.1 Extract networks

```bash
bsvae-networks extract-networks \
  --model-path results/study_prod \
  --dataset data/expression.csv \
  --output-dir results/study_prod/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Expected outputs:

- `results/study_prod/networks/mu_cosine_adjacency.npz`
- `results/study_prod/networks/gamma_knn_adjacency.npz` (if selected)

### 3.2 Extract modules

```bash
bsvae-networks extract-modules \
  --model-path results/study_prod \
  --dataset data/expression.csv \
  --output-dir results/study_prod/modules \
  --soft-eigengenes \
  --expr data/expression.csv
```

Expected outputs:

- `gamma.npz`
- `hard_assignments.npz`
- `soft_eigengenes.csv` (when requested)

### 3.3 Export and analyze latents

```bash
bsvae-networks export-latents \
  --model-path results/study_prod \
  --dataset data/expression.csv \
  --output results/study_prod/latents.npz

bsvae-networks latent-analysis \
  --model-path results/study_prod \
  --dataset data/expression.csv \
  --output-dir results/study_prod/latent_analysis \
  --kmeans-k 10 \
  --umap
```

## 4. Simulation Benchmark

Generate a synthetic dataset with known module labels:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --n-features 500 \
  --n-samples 200 \
  --n-modules 10 \
  --save-ground-truth data/sim_truth.csv
```

Train on simulated data:

```bash
bsvae-train sim_run \
  --dataset data/sim_expr.csv \
  --n-modules 10 \
  --epochs 50
```

Benchmark recovery:

```bash
bsvae-simulate benchmark \
  --dataset data/sim_expr.csv \
  --ground-truth data/sim_truth.csv \
  --model-path results/sim_run \
  --output results/sim_run/sim_metrics.json
```

Inspect `sim_metrics.json` for `ari` and `nmi`.

## 5. Troubleshooting

Common issues and fixes:

- Missing `--dataset`: always pass a matrix path to `bsvae-train`.
- OOM / memory pressure: reduce `--batch-size`, try `--no-cuda`.
- `gamma_knn` import error: install `faiss-cpu`.
- Hierarchical loss errors: ensure `--tx2gene` IDs match matrix row IDs.
- Missing eigengene output: `--soft-eigengenes` requires `--expr`.
- Data orientation mismatch: ensure row entities are features, not samples.

Quick debug run:

```bash
bsvae-train debug_run \
  --dataset data/expression.csv \
  --epochs 2 \
  --batch-size 16 \
  --no-cuda \
  --log-level debug
```

## 6. Migration From Legacy Configs

BSVAE no longer uses `hyperparam.ini` in the active CLI path.

Deprecated files:

- `src/bsvae/hyperparam.ini`
- `docs/hyperparam.ini`

Migration pattern:

1. Take values from old INI sections.
2. Convert them to explicit CLI flags in your run scripts.
3. Keep a single shell command per experiment for reproducibility.

Example conversion:

- old INI: `latent_dim=32`, `batch_size=128`, `beta=1.0`
- new CLI: `--latent-dim 32 --batch-size 128 --beta 1.0`

For long pipelines, keep parameterized shell scripts or workflow files (Snakemake/Nextflow) instead of INI presets.
