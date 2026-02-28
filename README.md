# BSVAE: Gaussian Mixture VAE for Module Discovery

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/bsvae/badge/?version=latest)](https://bsvae.readthedocs.io/)

BSVAE is a PyTorch package centered on `GMMModuleVAE`, a Gaussian-mixture variational autoencoder for feature-level module discovery in omics data.

## What It Does

- Trains a two-phase GMM-VAE (`bsvae-train`)
- Extracts feature-feature networks from trained models (`bsvae-networks`)
- Extracts module assignments and optional eigengenes (`bsvae-networks`)
- Exports latents (`mu`, `logvar`, `gamma`) as `.npz`
- Runs latent analysis (UMAP/t-SNE, clustering, covariate correlation) (`bsvae-networks latent-analysis`)
- Simulates synthetic datasets and benchmarks module recovery (`bsvae-simulate`)

## Installation

From PyPI:

```bash
pip install bsvae
```

From source:

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

## CLI Entry Points

- `bsvae-train`
- `bsvae-networks`
- `bsvae-simulate`
- `bsvae-sweep-k`

## Quickstart

For a full walkthrough (minimal run, production run, post-training analysis, simulation benchmark, troubleshooting, and migration), see `docs/tutorial.md`.

### Tune K (Recommended)

`--n-modules` (K) sets the expected number of modules. Recommended approach is
`bsvae-sweep-k` with stability replicates:

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

The selected model is retrained on the full dataset at:
`results/sweep1/final_k<K>/`.

### 1. Select K and train final model

Input matrix must be `features x samples` with feature IDs in row index and sample IDs in columns.

```bash
bsvae-sweep-k exp1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

Use the selected model directory for downstream steps:
`results/exp1/final_k<K>/` (example below uses `final_k16`).

### 2. Extract networks

```bash
bsvae-networks extract-networks \
  --model-path results/exp1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/exp1/final_k16/networks \
  --methods mu_cosine gamma_knn
```

### 3. Extract modules

```bash
bsvae-networks extract-modules \
  --model-path results/exp1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/exp1/final_k16/modules
```

### 4. Export latents

```bash
bsvae-networks export-latents \
  --model-path results/exp1/final_k16 \
  --dataset data/expression.csv \
  --output results/exp1/final_k16/latents.npz
```

### 5. Simulate and benchmark

Single dataset:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv

bsvae-simulate benchmark \
  --dataset data/sim_expr.csv \
  --ground-truth data/sim_truth.csv \
  --model-path results/exp1/final_k16 \
  --output results/exp1/final_k16/sim_metrics.json
```

Scenario grid for publication-style benchmarking:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13

bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Each scenario replicate writes method-ready files under
`results/sim_pub_v1/scenarios/<scenario_id>/rep_<rep>/`, including:

- `expr/features_x_samples.tsv.gz` (BSVAE, GNVAE)
- `expr/samples_x_features.tsv.gz` (WGCNA)
- `truth/modules_hard.csv` (hard labels for ARI/NMI)
- `method_inputs.json` (canonical paths for each method)

## Training Outputs

`bsvae-train` writes to `results/<experiment>/`:

- `model.pt` (weights)
- `specs.json` (metadata and run args)
- `train_losses.csv` (epoch/component losses)
- `model-<epoch>.pt` checkpoints when `--checkpoint-every` is set

## Data Formats

The loader supports:

- `.csv` / `.csv.gz`
- `.tsv` / `.tsv.gz`
- `.h5` / `.hdf5`
- `.h5ad` (optional `anndata` dependency)

## Python API (Minimal)

```python
from bsvae.utils.modelIO import load_model
from bsvae.networks.extract_networks import create_dataloader_from_expression, run_extraction

model = load_model("results/exp1/final_k16", is_gpu=False)
loader, feature_ids, _ = create_dataloader_from_expression("data/expression.csv", batch_size=128)
results = run_extraction(model, loader, feature_ids=feature_ids, methods=["mu_cosine"], top_k=50)
print(results[0].method, results[0].adjacency.shape)
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
