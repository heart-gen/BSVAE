# Command-Line Interface

BSVAE exposes a single entry point, `bsvae-train`, for launching experiments, validating models, and exporting evaluation metrics. Arguments are loaded from an `.ini` configuration file and can be overridden directly on the command line.

## Basic Usage
```bash
bsvae-train EXPERIMENT_NAME [options]
```

`EXPERIMENT_NAME` determines the output directory under `results/` (for example, `results/my_experiment/`).

## Configuration Hierarchy
1. Defaults are read from the section specified by `--config` and `--section` (defaults: bundled `hyperparam.ini` and `[Custom]`).
2. CLI flags override values from the configuration file.
3. Resolved arguments are saved to `metadata.json` inside the experiment directory.

💡 **Tip:** Store common presets as separate sections (e.g., `[beta_genenet]`) and override only the fields that change between runs.

## Argument Reference

### General
- `--config`, `-c` – Path to the hyperparameter `.ini` file.
- `--section` – Section within the `.ini` file to load.
- `--seed` – Random seed applied via `set_seed()`.
- `--no-cuda` – Force CPU execution.

### Training
- `--epochs` – Number of optimization epochs.
- `--batch-size` – Training batch size.
- `--lr` – Learning rate for Adam.
- `--checkpoint-every` – Save checkpoints after this many epochs.
- `--is-eval-only` – Skip training and run evaluation using an existing checkpoint.
- `--no-test` – Disable evaluation after training.
- `--eval-batchsize` – Batch size used during evaluation.

### Model
- `--latent-dim`, `-z` – Size of the latent factor space.
- `--hidden-dims`, `-Z` – Encoder hidden-layer widths (Python literal list).
- `--dropout` – Dropout rate applied to encoder layers.
- `--learn-var` – Learn per-gene decoder variance.
- `--init-sd` – Standard deviation for decoder weight initialization.

### Loss
- `--loss` – Choice of reconstruction loss (`VAE` or `beta`).
- `--beta` – KL divergence weight (β-VAE).
- `--l1-strength` – L1 sparsity penalty on decoder loadings.
- `--lap-strength` – Laplacian smoothness penalty using PPI priors.

### Dataset
- `--dataset` – Registered dataset loader key (defaults to `genenet`).
- `--gene-expression-filename` – CSV containing a gene-expression matrix (genes × samples). *Mutually exclusive with* `--gene-expression-dir`.
- `--gene-expression-dir` – Directory containing `X_train.csv`, `X_test.csv`, and optional metadata files. *Mutually exclusive with* `--gene-expression-filename`.

### PPI Priors
- `--ppi-taxid` – NCBI taxonomy identifier for the PPI network (default `9606` for human).
- `--ppi-cache` – Directory where downloaded STRING/IntAct PPI graphs are cached.

## PPI cache downloader

Download a STRING network into the configured cache without launching training:

```bash
bsvae-download-ppi --taxid 9606 --cache-dir ~/.bsvae/ppi
```

This command respects the same defaults as `bsvae-train` (taxid `9606`, cache `~/.bsvae/ppi`).

## Common Workflows

Train with defaults using a CSV:
```bash
bsvae-train my_experiment --gene-expression-filename data/expression.csv
```

Run evaluation only on a completed experiment:
```bash
bsvae-train my_experiment --is-eval-only --no-test --gene-expression-filename data/expression.csv
```

Use a curated preset section:
```bash
bsvae-train beta_genenet_run \
  --section beta_genenet \
  --gene-expression-dir data/splits/
```

💡 **Tip:** If both input flags are provided, the parser raises `Specify exactly one of --gene-expression-filename or --gene-expression-dir.` Fix the invocation by choosing a single data source.
