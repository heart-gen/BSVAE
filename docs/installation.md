# Installation

## Requirements

- Python `>=3.11,<3.14`
- PyTorch `>=2.8,<3.0` (installed with the package; install a CUDA build first if you want the GPU)
- R with `WGCNA` (which brings `dynamicTreeCut`), plus `glasso` for the TWN method, only if you run `bsvae-benchmark` against WGCNA, acorde or TWN

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

To develop with Poetry instead (dev tools and all extras):

```bash
poetry install --with dev --extras "anndata yaml"
poetry run pytest -q
```

## Optional Dependencies

- `anndata` is only required for `.h5ad` input or `.h5ad` latent export helpers
- `pyyaml` is only required to read YAML scenario configs (`bsvae-simulate`, `bsvae-benchmark`); JSON configs need nothing

Install both extras with:

```bash
pip install "bsvae[anndata,yaml]"
```

`bsvae-benchmark check-env` reports whether `Rscript`, `WGCNA` and `torch` can be found.

## Verify The Install

```bash
bsvae-train --help
bsvae-sweep-k --help
bsvae-networks --help
bsvae-simulate --help
bsvae-benchmark --help
bsvae-benchmark check-env   # optional: R, WGCNA and torch
python -c "import bsvae; print(bsvae.__version__)"
```

## GPU Behavior

The CLI uses CUDA when available. Use `--no-cuda` to force CPU mode.

```bash
bsvae-train run_cpu --dataset data/expression.csv --no-cuda
```
