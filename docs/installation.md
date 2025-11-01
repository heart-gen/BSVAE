# Installation

## Requirements
- Python ≥ 3.11
- PyTorch ≥ 2.0 with optional CUDA support (2.8 recommended)
- GCC toolchain and build essentials for compiling native extensions
- (Optional) CUDA toolkit if training on GPU

## Install from Source
```bash
git clone https://github.com/.../bsvae.git
cd bsvae
pip install -e .
```

The editable installation exposes the `bsvae-train` command-line entry point defined in `pyproject.toml`.

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
