# BSVAE: Gaussian Mixture VAE for Module Discovery

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/bsvae/badge/?version=latest)](https://bsvae.readthedocs.io/)

BSVAE is a PyTorch package for feature-level module discovery in omics data. The main model, `GMMModuleVAE`, learns latent structure across features such as genes, transcripts, or proteins, then supports downstream network extraction, module assignment, latent analysis, and simulation-based benchmarking.

## What The Package Provides

- `bsvae-train` for training a two-phase GMM-VAE
- `bsvae-sweep-k` for selecting the number of modules (`K`) with held-out validation and optional stability replicates
- `bsvae-networks` for post-training network extraction, module extraction, latent export, and latent analysis
- `bsvae-simulate` for synthetic data generation, scenario-grid simulation, and module-recovery benchmarking

## Installation

Install from PyPI:

```bash
pip install bsvae
```

Install from source:

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

Optional dependency:

- `anndata` is only needed for `.h5ad` input or `.h5ad` latent export helpers

## Input Data Contract

Training and most downstream commands expect an expression matrix in `features x samples` orientation:

- rows are features such as genes, transcripts, or proteins
- columns are sample IDs
- the first column is the feature index when using CSV/TSV

Supported loaders:

- `.csv` / `.csv.gz`
- `.tsv` / `.tsv.gz`
- `.h5` / `.hdf5`
- `.h5ad` with optional `anndata`

## CLI Entry Points

- `bsvae-train`
- `bsvae-sweep-k`
- `bsvae-networks`
- `bsvae-simulate`

## Quickstart

Train a small model:

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

Recommended production flow: select `K` first, then use the retrained final model.

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

This writes sweep outputs under `results/sweep1/sweep_k/` and retrains the selected model under `results/sweep1/final_k<K>/`.

Extract module assignments from the final model:

```bash
bsvae-networks extract-modules \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Export latents:

```bash
bsvae-networks export-latents \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output results/sweep1/final_k16/latents
```

This command writes `results/sweep1/final_k16/latents.npz` containing `mu`, `logvar`, `gamma`, and `feature_ids`.

Extract feature-feature networks:

```bash
bsvae-networks extract-networks \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Run latent-space analysis:

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

## Training Outputs

`bsvae-train` writes to `results/<experiment>/` by default:

- `model.pt`
- `specs.json`
- `train_losses.csv`
- `model-<epoch>.pt` checkpoints when `--checkpoint-every` is greater than zero

`bsvae-sweep-k` writes:

- `results/<name>/sweep_k/sweep_results.csv`
- `results/<name>/sweep_k/sweep_summary.json`
- `results/<name>/sweep_k/k<K>/rep_<rep>/...` per-sweep run outputs
- `results/<name>/final_k<K>/...` for the final retrained model when `--train-final` is enabled

## Simulation and Benchmarking

Generate one synthetic dataset:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv
```

Generate a scenario grid:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13
```

Validate the generated grid:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Each scenario replicate writes method-ready files under `results/sim_pub_v1/scenarios/<scenario_id>/rep_<rep>/`, including:

- `expr/features_x_samples.tsv.gz`
- `expr/samples_x_features.tsv.gz`
- `truth/modules_hard.csv`
- `method_inputs.json`

## Documentation

See the docs for the full workflow and command reference:

- `docs/index.md`
- `docs/quickstart.md`
- `docs/tutorial.md`
- `docs/cli.md`
- `docs/networks.md`
- `docs/hyperparameters.md`

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
