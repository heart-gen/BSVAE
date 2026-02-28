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

Goal: select K robustly, then train the final production model.

### 2.1 Tuning the number of modules (K)

`--n-modules` (K) sets the expected number of latent modules/clusters in the GMM prior.
This is a key hyperparameter and should be tuned for each dataset.

Recommended approach: use `bsvae-sweep-k` with stability mode to avoid overfitting K.
It holds out a feature split and, when `--stability-reps > 1`, selects K by mean
pairwise ARI across replicate runs.

Example sweep (recommended):

```bash
bsvae-sweep-k sweep_prod \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

The selected model is retrained on the full dataset at:
`results/sweep_prod/final_k<K>/`.

Optional downstream check (latent clustering on `mu`):

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

Recommended checks during/after run:

- loss is finite in logs
- checkpoints appear as `model-<epoch>.pt`
- final artifacts exist in `results/sweep_prod/final_k<K>/`

## 3. Post-Training Analysis

### 3.1 Extract networks

```bash
bsvae-networks extract-networks \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Expected outputs:

- `results/sweep_prod/final_k16/networks/mu_cosine_adjacency.npz`
- `results/sweep_prod/final_k16/networks/gamma_knn_adjacency.npz` (if selected)

### 3.2 Extract modules

```bash
bsvae-networks extract-modules \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/modules \
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
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output results/sweep_prod/final_k16/latents.npz

bsvae-networks latent-analysis \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

## 4. Simulation Benchmark

Generate a publication-style scenario grid:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13

bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Run one scenario replicate through BSVAE:

```bash
bsvae-train sim_run \
  --dataset results/sim_pub_v1/scenarios/S001__confounding-none__n_samples-100__nonlinear_mode-off__overlap_rate-0.0__signal_scale-0.4/rep_000/expr/features_x_samples.tsv.gz \
  --n-modules 20 \
  --epochs 50
```

Benchmark hard-module recovery:

```bash
bsvae-simulate benchmark \
  --dataset results/sim_pub_v1/scenarios/S001__confounding-none__n_samples-100__nonlinear_mode-off__overlap_rate-0.0__signal_scale-0.4/rep_000/expr/features_x_samples.tsv.gz \
  --ground-truth results/sim_pub_v1/scenarios/S001__confounding-none__n_samples-100__nonlinear_mode-off__overlap_rate-0.0__signal_scale-0.4/rep_000/truth/modules_hard.csv \
  --model-path results/sim_run \
  --output results/sim_run/sim_metrics.json
```

For WGCNA, use `expr/samples_x_features.tsv.gz`.
For GNVAE, use `expr/features_x_samples.tsv.gz` (or `gnvae/fold_*/X_train.tsv.gz`).
`method_inputs.json` provides canonical paths for all three methods.

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
