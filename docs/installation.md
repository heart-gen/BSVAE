# Installation

## Requirements

- Python `>=3.11,<3.14`
- PyTorch `>=2.8,<3.0`

## Install From PyPI

```bash
pip install bsvae
```

## Install From Source

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

## Optional Dependencies

- `anndata` is optional and only required for `.h5ad` input or `.h5ad` latent export helpers

## Verify The Install

```bash
bsvae-train --help
bsvae-sweep-k --help
bsvae-networks --help
bsvae-simulate --help
```

## GPU Behavior

The CLI uses CUDA when available. Use `--no-cuda` to force CPU mode.

```bash
bsvae-train run_cpu --dataset data/expression.csv --no-cuda
```
