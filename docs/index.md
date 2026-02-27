# BSVAE Documentation

BSVAE provides a GMM-based variational autoencoder workflow for omics module discovery.

## Core Components

- `bsvae-train`: train and optionally evaluate `GMMModuleVAE`
- `bsvae-networks`: network extraction, module extraction, latent export/analysis
- `bsvae-simulate`: synthetic data generation, scenario-grid simulation, and benchmarking

## Current CLI Entry Points

| Command | Purpose |
| --- | --- |
| `bsvae-train` | Train a GMMModuleVAE model |
| `bsvae-networks` | Post-training networks/modules/latents |
| `bsvae-simulate` | Single-dataset simulation, scenario-grid generation, ARI/NMI benchmarking |

## Quick Start

```bash
pip install bsvae
bsvae-train my_run --dataset data/expression.csv
```

## Docs Map

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Tutorial](tutorial.md)
- [CLI Reference](cli.md)
- [Usage Guide](usage.md)
- [Network Workflows](networks.md)
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
bsvae_networks
usage
hyperparameters
hpc
faq
contributing
api/models
api/utils
```
