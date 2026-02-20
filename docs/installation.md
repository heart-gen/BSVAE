# Installation

## Requirements

- Python `>=3.11,<3.14`
- PyTorch `>=2.8`

## Install from PyPI

```bash
pip install bsvae
```

## Install from source

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

## Verify install

```bash
bsvae-train --help
bsvae-networks --help
bsvae-simulate --help
```

## GPU behavior

CLI commands use CUDA when available. Use `--no-cuda` to force CPU mode.

```bash
bsvae-train run_cpu --dataset data/expression.csv --no-cuda
```
