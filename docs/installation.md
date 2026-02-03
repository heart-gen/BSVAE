# Installation

## Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.8 with optional CUDA support
- (Optional) CUDA toolkit if training on GPU

## Install from PyPI

```bash
pip install bsvae
```

This installs BSVAE and all required dependencies.

## Install Development Version

For the latest features and bug fixes:

```bash
pip install git+https://github.com/heart-gen/BSVAE.git
```

Or for local development:

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

The installation exposes three CLI entry points: `bsvae-train`, `bsvae-networks`, and `bsvae-download-ppi`.

## Verifying the Installation
```bash
bsvae-train --help
pytest -q
```

The first command prints the CLI usage summary. Running the test suite ensures PyTorch and data utilities are configured correctly.

## GPU Detection
BSVAE automatically selects `cuda` when `torch.cuda.is_available()` returns `True`. Pass `--no-cuda` to force CPU execution:

```bash
bsvae-train my_experiment --gene-expression-filename data/expression.csv --no-cuda
```

💡 **Tip:** When running in environments without GPU drivers, set `CUDA_VISIBLE_DEVICES=""` or use the flag above to avoid runtime warnings.
