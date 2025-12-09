# BSVAE: Biologically Structured Variational Autoencoder

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1059699525.svg)](https://doi.org/10.5281/zenodo.17871790)

**BSVAE** is a PyTorch package for **Structured Factor Variational Autoencoders** (StructuredFactorVAE).  
It is designed for **gene expression modeling with biological priors**, integrating **protein–protein interaction (PPI) networks** and **sparsity constraints** for interpretable latent representations.

---

## Features

- **Structured VAE architecture** (`StructuredFactorVAE`)
  - Factorized encoder/decoder with group sparsity
  - Optional Laplacian regularization from PPI networks
- **Dataset utilities**
  - Load gene expression matrices (`GeneExpression`)
  - Support for CSV full-matrix mode or pre-split train/test mode
- **Biological priors**
  - Fetch and cache STRING PPI networks by NCBI TaxID
  - Map gene symbols / Ensembl IDs to protein IDs using MyGene.info or BioMart
- **Training and evaluation**
  - Unified training loop (`Trainer`)
  - Evaluation with reconstruction, KL, sparsity, and Laplacian penalties (`Evaluator`)
- **Reproducibility**
  - Save/load models + metadata (`modelIO`)
  - Configurable hyperparameters via `hyperparam.ini`

---

## Installation

Clone the repo and install with [Poetry](https://python-poetry.org/) (recommended):

```bash
git clone https://github.com/YOUR-LAB/BSVAE.git
cd BSVAE
poetry install
````

Or using pip:

```bash
pip install -e .
```

Dependencies:

* Python 3.9+
* PyTorch ≥ 2.0
* pandas, numpy, scikit-learn
* networkx, scipy
* mygene (for gene annotation)

---

## Quickstart

### 1. Prepare gene expression data

BSVAE expects **genes × samples** CSVs.

* **Full-matrix mode**:
  Provide `expr.csv` with all samples → 10-fold CV split is created.
* **Pre-split mode**:
  Provide directory with `X_train.csv` and `X_test.csv`.

### 2. Train a model

```bash
poetry run python -m bsvae.main exp1 \
    --gene-expression-filename data/expr.csv \
    --epochs 50 \
    --latent-dim 10 \
    --ppi-taxid 9606
```

* Results (checkpoints, logs, metadata) saved under `results/exp1/`.

### 3. Evaluate a trained model

```bash
poetry run python -m bsvae.main exp1 \
    --gene-expression-filename data/expr.csv \
    --is-eval-only
```

---

## ⚙Configuration

Hyperparameters can be set via `hyperparam.ini`:

```ini
[Custom]
seed = 42
no_cuda = False
epochs = 100
batch_size = 64
latent_dim = 10
hidden_dims = [128, 64]
dropout = 0.1
l1_strength = 1e-3
lap_strength = 1e-4
```

Override from CLI if needed:

```bash
--epochs 50 --latent-dim 20
```

---

## PPI Priors

BSVAE supports automatic download & caching of **STRING v12.0 PPI networks**.

* Supported species (via NCBI TaxID):

  * Human (`9606`)
  * Mouse (`10090`)
  * Rat (`10116`)
  * Fly (`7227`)
* Cache location defaults to `~/.bsvae/ppi` (override via `--ppi-cache`).

### Prefetch PPI cache from the CLI

Use the lightweight downloader to cache a STRING network ahead of training:

```bash
poetry run bsvae-download-ppi --taxid 9606 --cache-dir ~/.bsvae/ppi
```

---

## Citation

If you use **BSVAE** in your research, please cite:

```
@article{Benjamin2025bsvae,
  title={Structured Factor Variational Autoencoder with Biological Priors},
  author={Kynon J. M. Benjamin},
  year={2025},
  journal={N/A}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).

