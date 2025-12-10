# BSVAE Documentation

Biologically Structured Variational Autoencoder (**BSVAE**) learns interpretable latent factors from bulk or single-cell gene-expression profiles by combining a β-VAE objective with sparsity and protein-protein interaction (PPI) smoothness priors. The command-line interface automates end-to-end training and evaluation workflows for reproducible biology-aware representation learning.

## Key Features
- **Structured latent space** with configurable sparsity and Laplacian regularization.
- **End-to-end CLI pipeline** for training, validation, and evaluation in a single command.
- **Reproducible experiments** via `.ini` configuration sections and deterministic seeds.
- **PPI-aware priors** that encourage biologically coherent factors.

## What’s New in This Release

This release introduces major improvements across configuration handling, dataset flexibility, metadata stability, PPI integration, and training/evaluation robustness. Users should experience smoother training runs, more reproducible evaluation results, and expanded compatibility with common transcriptomics formats.

### Improved Model Metadata & Checkpointing

* Unified and normalized model metadata for StructuredFactorVAE, including latent dimensionality, input gene count, and regularization settings.
* Checkpoints now reliably store and reload Laplacian buffers and maintain device consistency.
* Evaluation now validates gene dimensionality against the training dataset to prevent silent mismatches.

### Enhanced Input & Dataset Support

* Gene expression files can now be supplied as CSV, TSV, or their compressed `.gz` variants.
* Automatic correction of dataset orientation (genes × samples).
* More robust parsing of dataset paths and safe handling of missing files.

### Better PPI / STRING Integration

* New CLI tool for downloading STRING protein–protein interaction networks.
* Laplacian matrices now use safer Tensor conversions and track device placement correctly.

### Logging & Configuration Improvements

* Added a user-configurable `--log-level` flag to control verbosity.
* All loss logging respects the selected logging level.
* Standardization of training loss logs into `.csv` format for easier downstream analysis.
* Safer creation of logging directories, even for runtimes with bare filenames.

### Reliability Improvements for Training & Evaluation

* Default batch size behavior fixed for evaluation dataloaders.
* Prevented evaluation failures due to empty batches by enforcing `drop_last=False`.
* More stable model initialization and metadata resolution.

These changes collectively enhance reproducibility, dataset compatibility, and robustness across the entire BSVAE training and evaluation workflow.

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
- [API Docs](api/models.md)

```{toctree}
:maxdepth: 2

installation
quickstart
cli
networks
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
