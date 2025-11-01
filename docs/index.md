# BSVAE Documentation

Biologically Structured Variational Autoencoder (**BSVAE**) learns interpretable latent factors from bulk or single-cell gene-expression profiles by combining a β-VAE objective with sparsity and protein-protein interaction (PPI) smoothness priors. The command-line interface automates end-to-end training and evaluation workflows for reproducible biology-aware representation learning.

## Key Features
- **Structured latent space** with configurable sparsity and Laplacian regularization.
- **End-to-end CLI pipeline** for training, validation, and evaluation in a single command.
- **Reproducible experiments** via `.ini` configuration sections and deterministic seeds.
- **PPI-aware priors** that encourage biologically coherent factors.

## Quick Install & Run
```bash
pip install bsvae
bsvae-train my_experiment --gene-expression-filename data/expression.csv
```

Explore the rest of the documentation to learn how to configure experiments and interpret results:

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [CLI Reference](cli.md)
- [API Docs](api/models.md)

```{toctree}
:maxdepth: 2

installation
quickstart
cli
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
