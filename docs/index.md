# BSVAE Documentation

BSVAE is a CLI-first workflow for feature-level module discovery in omics data. It combines a Gaussian-mixture variational autoencoder with tools for model selection, network extraction, module assignment, latent analysis, and synthetic benchmarking.

## Main Commands

| Command | Purpose |
| --- | --- |
| `bsvae-train` | Train a `GMMModuleVAE` on a `features x samples` matrix |
| `bsvae-sweep-k` | Select the number of modules with held-out validation and optional stability replicates |
| `bsvae-networks` | Extract networks, modules, latents, and latent-analysis outputs from a trained model |
| `bsvae-simulate` | Generate synthetic datasets, build scenario grids, and benchmark recovery |

## Data Orientation

Most commands expect an expression matrix in `features x samples` orientation:

- rows are feature IDs
- columns are sample IDs
- CSV and TSV files use the first column as the feature index

Supported inputs:

- `.csv` / `.csv.gz`
- `.tsv` / `.tsv.gz`
- `.h5` / `.hdf5`
- `.h5ad` with optional `anndata`

## Start Here

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Tutorial](tutorial.md)
- [CLI Reference](cli.md)
- [Network And Latent Workflows](networks.md)
- [Usage Guide](usage.md)
- [Hyperparameters](hyperparameters.md)
- [HPC](hpc.md)
- [FAQ](faq.md)
- [Contributing](contributing.md)
- [API: Models](api/models.md)
- [API: Utils](api/utils.md)

```{toctree}
:maxdepth: 2

installation
quickstart
tutorial
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
