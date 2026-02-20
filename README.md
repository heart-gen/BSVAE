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

## Quickstart

For a full walkthrough (minimal run, production run, post-training analysis, simulation benchmark, troubleshooting, and migration), see `docs/tutorial.md`.

### 1. Train

Input matrix must be `features x samples` with feature IDs in row index and sample IDs in columns.

```bash
bsvae-train exp1 \
  --dataset data/expression.csv \
  --epochs 100 \
  --n-modules 20 \
  --latent-dim 32
```

### 2. Extract networks

```bash
bsvae-networks extract-networks \
  --model-path results/exp1 \
  --dataset data/expression.csv \
  --output-dir results/exp1/networks \
  --methods mu_cosine gamma_knn
```

### 3. Extract modules

```bash
bsvae-networks extract-modules \
  --model-path results/exp1 \
  --dataset data/expression.csv \
  --output-dir results/exp1/modules
```

### 4. Export latents

```bash
bsvae-networks export-latents \
  --model-path results/exp1 \
  --dataset data/expression.csv \
  --output results/exp1/latents.npz
```

### 5. Simulate and benchmark

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv

bsvae-simulate benchmark \
  --dataset data/sim_expr.csv \
  --ground-truth data/sim_truth.csv \
  --model-path results/exp1 \
  --output results/exp1/sim_metrics.json
```

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

model = load_model("results/exp1", is_gpu=False)
loader, feature_ids, _ = create_dataloader_from_expression("data/expression.csv", batch_size=128)
results = run_extraction(model, loader, feature_ids=feature_ids, methods=["mu_cosine"], top_k=50)
print(results[0].method, results[0].adjacency.shape)
```

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
