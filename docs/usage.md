# Usage Guide

This guide walks through a complete BSVAE workflow, from data preparation to factor interpretation.

## 1. Prepare Data
BSVAE expects gene-expression measurements where rows correspond to samples and columns to genes. Use either a single CSV (`--gene-expression-filename`) or a directory with explicit splits (`--gene-expression-dir`). When a directory is used, `get_dataloaders()` loads the following files:

```
data/
├── X_train.csv
├── X_test.csv
├── gene_names.txt
```

Additional files such as `y_train.csv` are ignored unless referenced by custom dataset code.

## 2. Configure Hyperparameters
Select or create a section in `hyperparam.ini` that specifies model architecture, training schedule, and loss weights. Override particular flags on the command line when exploring nearby settings.

## 3. Launch Training
```bash
bsvae-train my_experiment \
  --section beta_genenet \
  --gene-expression-dir data/splits/
```

The command creates `results/my_experiment/` and prints progress logs to stdout.

## 4. Evaluate Results
Unless `--no-test` is provided, the CLI loads the best checkpoint and runs `Evaluator` on the test split. Metrics, reconstruction losses, and β-VAE terms are appended to `logs.txt` and serialized to `metadata.json`.

💡 **Tip:** Re-run with `--is-eval-only` to recompute metrics without additional training steps.

## 5. Interpret Latent Factors
Checkpoints are saved as `model.pt`. Programmatic APIs enable downstream analysis:

```python
from bsvae.utils.modelIO import load_model
from bsvae.utils.mapping import rank_genes_for_latent

model = load_model("results/my_experiment", is_gpu=False)
top_genes = rank_genes_for_latent(model, dim=0)
print(top_genes[:10])
```

This snippet ranks genes contributing to the first latent dimension, helping interpret biological processes captured by the model.

## Checkpointing and Experiment Structure
Each experiment directory contains:
- `model.pt` – Serialized `StructuredFactorVAE` weights.
- `metadata.json` – Full CLI argument snapshot.
- `logs.txt` – Training and evaluation log stream.
- `checkpoints/` (optional) – Periodic saves based on `--checkpoint-every`.

Optional scripts located in `bin/` provide plotting and metric aggregation helpers, such as `bin/metrics_all.sh` and `bin/plot_all.sh`. Adapt these wrappers to your HPC environment for large-scale sweeps.
