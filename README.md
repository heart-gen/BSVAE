# BSVAE: Biologically Structured Variational Autoencoder

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
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
- **Post-training analysis**
  - Gene–gene network extraction via decoder similarity, propagated covariance, Graphical Lasso, and Laplacian refinement
  - Latent export (`mu`, `logvar`) to CSV or AnnData for downstream workflows

---

## Installation

Install from PyPI:

```bash
pip install bsvae
```

Or install from source with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/YOUR-LAB/BSVAE.git
cd BSVAE
poetry install
```

Dependencies:

* Python 3.11+
* PyTorch ≥ 2.8
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
bsvae-train exp1 \
    --gene-expression-filename data/expr.csv \
    --epochs 50 \
    --latent-dim 10 \
    --ppi-taxid 9606
```

* Results (checkpoints, logs, metadata) saved under `results/exp1/`.

### 3. Evaluate a trained model

```bash
bsvae-train exp1 \
    --gene-expression-filename data/expr.csv \
    --is-eval-only
```

### 4. Extract networks and export latents

```bash
bsvae-networks extract-networks \
    --model-path results/exp1 \
    --dataset data/expr.csv \
    --output-dir results/exp1/networks
# optional: --methods latent_cov graphical_lasso laplacian

bsvae-networks export-latents \
    --model-path results/exp1 \
    --dataset data/expr.csv \
    --output results/exp1/latents.h5ad
```

The extractor writes **sparse NPZ** adjacency matrices and **Parquet** edge
lists by default (``--sparse`` and ``--compress`` are on unless disabled with
``--no-sparse``/``--no-compress``). When ``--threshold 0`` with sparse output,
an adaptive threshold is computed from ``--target-sparsity`` (default 0.01 =
top 1% of edges). Use ``--quantize`` (default ``int8``) to reduce adjacency
size. Legacy dense CSV/TSV/NPY outputs are available only with ``--no-sparse``
or explicit legacy formats. By default the decoder-loading cosine similarity
(``w_similarity``) is computed; add other methods with ``--methods``. Latent
exports include per-sample ``mu`` and ``logvar`` as tidy CSV or AnnData files.

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
bsvae-train my_experiment --epochs 50 --latent-dim 20
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
bsvae-download-ppi --taxid 9606 --cache-dir ~/.bsvae/ppi
```

### Troubleshooting PPI downloads on HPC systems

Some clusters block HTTPS certificate resolution for outbound downloads. If `bsvae-download-ppi` cannot reach STRING, manually
cache the file with `wget` (or `curl`) using `--no-check-certificate` and point `--ppi-cache` to the same directory:

```bash
OUTDIR="$HOME/.bsvae/ppi"
mkdir -p "${OUTDIR}"
wget --no-check-certificate \
  "https://stringdb-static.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz" \
  -O "${OUTDIR}/9606_string.txt.gz"
```

Use `curl -k -L "<url>" -o "${OUTDIR}/9606_string.txt.gz"` if `wget` is unavailable.

---

## Integration notes

- The `bsvae-networks` workflows reuse the same gene ordering as training. When
  loading a standalone expression file, ensure columns correspond to the genes
  seen by the checkpoint.
- The CLI automatically handles CPU/GPU placement based on availability; models
  are loaded in evaluation mode without modifying training metadata.
- Network extraction functions are written to be test-friendly: they accept
  PyTorch `DataLoader` instances, operate without global state, and persist
  outputs as CSV/TSV/NPY for interoperability with graph toolchains.

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

This project is licensed under the [GNU General Public License v3.0](LICENSE).

