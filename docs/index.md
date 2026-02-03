# BSVAE Documentation

Biologically Structured Variational Autoencoder (**BSVAE**) learns interpretable latent factors from bulk or single-cell gene-expression profiles by combining a β-VAE objective with sparsity and protein-protein interaction (PPI) smoothness priors. The command-line interface automates end-to-end training, evaluation, and post-training analysis workflows for reproducible biology-aware representation learning.

## Key Features

- **Structured latent space** with configurable sparsity (L1) and Laplacian regularization
- **End-to-end CLI pipeline** for training, validation, and evaluation in a single command
- **Reproducible experiments** via `.ini` configuration sections and deterministic seeds
- **PPI-aware priors** that encourage biologically coherent factors (STRING v12.0)
- **Post-training network analysis** including gene–gene network extraction, WGCNA-like module clustering (Leiden/spectral), and latent space analysis
- **GPU acceleration** for model inference and network extraction operations

## CLI Entry Points

BSVAE provides three command-line tools:

| Command | Description |
|---------|-------------|
| `bsvae-train` | Train and evaluate StructuredFactorVAE models |
| `bsvae-networks` | Post-training network extraction, module clustering, and latent analysis |
| `bsvae-download-ppi` | Pre-cache STRING PPI networks for offline/HPC use |

## What's New

### Network Analysis Pipeline

* **Gene–gene network extraction** with four complementary methods: decoder similarity (`w_similarity`), latent covariance propagation (`latent_cov`), Graphical Lasso (`graphical_lasso`), and Laplacian refinement (`laplacian`)
* **Sparse output formats**: NPZ adjacency matrices and Parquet edge lists with optional int8/float16 quantization for efficient storage
* **GPU acceleration** for network extraction operations (except Graphical Lasso fitting)

### Module Extraction (WGCNA-like)

* **Leiden community detection** with automatic resolution optimization (modularity-based)
* **Resolution sweep** functionality with parallel execution support (`--n-jobs`)
* **Eigengene computation** for module-level expression summaries
* **Spectral clustering** alternative for different network types

### Latent Space Analysis

* **Export latents** to CSV or AnnData (.h5ad) format
* **Sample-level clustering** with K-means and Gaussian Mixture Models
* **Dimensionality reduction** with UMAP and t-SNE
* **Covariate correlation** analysis

### Robustness Improvements

* Unified model metadata and checkpoint handling
* CSV, TSV, and gzip-compressed input support with automatic orientation detection
* STRING v12.0 PPI integration with improved caching
* Configurable logging levels and standardized loss output formats

## Quick Install & Run
```bash
pip install bsvae
bsvae-train my_experiment --gene-expression-filename data/expression.csv
```

Explore the rest of the documentation to learn how to configure experiments and interpret results:

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [CLI Reference](cli.md)
- [Network extraction](networks.md)
- [BSVAE network utilities](bsvae_networks.md)
- [API Docs](api/models.md)

```{toctree}
:maxdepth: 2

installation
quickstart
cli
networks
bsvae_networks
usage
hyperparameters
hpc
faq
contributing
api/models
api/utils
```

---

*BSVAE was developed by the Systems Biology Lab. Please cite the associated preprint when publishing results generated with this software.*
