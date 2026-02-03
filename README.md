# BSVAE: Biologically Structured Variational Autoencoder

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/bsvae/badge/?version=latest)](https://bsvae.readthedocs.io/)
[![DOI](https://zenodo.org/badge/1059699525.svg)](https://doi.org/10.5281/zenodo.17871790)

**BSVAE** is a PyTorch package for **Structured Factor Variational Autoencoders** (StructuredFactorVAE).
It is designed for **gene expression modeling with biological priors**, integrating **protein–protein interaction (PPI) networks** and **sparsity constraints** for interpretable latent representations.

---

## Features

- **Structured VAE architecture** (`StructuredFactorVAE`)
  - Factorized encoder/decoder with group sparsity (L1 regularization)
  - Optional Laplacian regularization from PPI networks
  - Per-gene variance learning
  - GPU/CPU compatible
- **Dataset utilities**
  - Load gene expression matrices (`GeneExpression`)
  - Support for CSV/TSV (including gzip-compressed) in full-matrix or pre-split mode
  - Automatic correction of dataset orientation (genes × samples)
- **Biological priors**
  - Fetch and cache STRING v12.0 PPI networks by NCBI TaxID
  - Map gene symbols / Ensembl IDs to protein IDs using MyGene.info or BioMart
- **Training and evaluation**
  - Unified training loop (`Trainer`) with checkpointing and loss logging
  - Evaluation with reconstruction, KL, sparsity, and Laplacian penalties (`Evaluator`)
- **Reproducibility**
  - Save/load models + metadata (`modelIO`)
  - Configurable hyperparameters via `hyperparam.ini`
- **Post-training network analysis** (`bsvae-networks`)
  - Gene–gene network extraction via decoder similarity, latent covariance, Graphical Lasso, and Laplacian refinement
  - Gene module clustering (Leiden with auto-resolution, spectral) with eigengene computation
  - Latent export (`mu`, `logvar`) to CSV or AnnData for downstream workflows
  - Sample-level latent analysis (UMAP, t-SNE, K-means, GMM)

---

## Installation

Install from PyPI:

```bash
pip install bsvae
```

For the latest development version:

```bash
pip install git+https://github.com/heart-gen/BSVAE.git
```

**Requirements:**

* Python 3.11+
* PyTorch ≥ 2.8

Core dependencies (automatically installed): pandas, numpy, scipy, scikit-learn,
networkx, igraph, leidenalg, mygene, anndata.

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

### 4. Extract networks

```bash
bsvae-networks extract-networks \
    --model-path results/exp1 \
    --dataset data/expr.csv \
    --output-dir results/exp1/networks
# optional: --methods latent_cov graphical_lasso laplacian
```

Outputs **sparse NPZ** adjacency matrices and **Parquet** edge lists by default.
Use `--quantize int8` (default) to reduce file size. The decoder-loading cosine
similarity (`w_similarity`) is computed by default; add other methods with `--methods`.

### 5. Extract gene modules (WGCNA-like)

```bash
bsvae-networks extract-modules \
    --adjacency results/exp1/networks/w_similarity_adjacency.npz \
    --expr data/expr.csv \
    --output-dir results/exp1/modules \
    --cluster-method leiden \
    --resolution-auto \
    --n-jobs 4
```

Clusters the gene network into modules using Leiden community detection with
automatic resolution optimization. Outputs module assignments and eigengenes.

### 6. Export latents for downstream analysis

```bash
bsvae-networks export-latents \
    --model-path results/exp1 \
    --dataset data/expr.csv \
    --output results/exp1/latents.h5ad
```

Exports per-sample `mu` and `logvar` as AnnData (.h5ad) or CSV files.

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

## CLI Entry Points

BSVAE provides three command-line tools:

| Command | Description |
|---------|-------------|
| `bsvae-train` | Train and evaluate models |
| `bsvae-networks` | Post-training network/module extraction and latent analysis |
| `bsvae-download-ppi` | Pre-cache STRING PPI networks |

### `bsvae-networks` subcommands

| Subcommand | Description |
|------------|-------------|
| `extract-networks` | Compute gene–gene adjacency matrices |
| `extract-modules` | Cluster networks into gene modules (Leiden/spectral) |
| `export-latents` | Export encoder μ and log σ² to CSV or AnnData |
| `latent-analysis` | Sample-level clustering, UMAP, t-SNE, covariate correlation |

---

## Integration Notes

- The `bsvae-networks` workflows reuse the same gene ordering as training. When
  loading a standalone expression file, ensure columns correspond to the genes
  seen by the checkpoint.
- The CLI automatically handles CPU/GPU placement based on availability; models
  are loaded in evaluation mode without modifying training metadata.
- Network extraction outputs sparse NPZ adjacency matrices and Parquet edge lists
  by default for efficient storage and interoperability with graph toolchains.
- Module extraction supports parallel execution (`--n-jobs`) for resolution sweeps
  and auto-optimization on HPC systems.

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

