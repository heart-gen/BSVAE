# Quick Start

The `bsvae-train` command trains a StructuredFactorVAE and evaluates it on held-out data in a single run. This section demonstrates the minimal steps required to launch an experiment.

## Prepare Input Data
Provide a gene-expression matrix either as:
- A CSV file containing genes × samples (`--gene-expression-filename`). The loader infers train/test splits internally.
- A directory with explicit splits (`--gene-expression-dir`) containing `X_train.csv` and `X_test.csv` files plus optional metadata.

## Minimal Configuration
Copy or extend the `[Custom]` section from `src/bsvae/hyperparam.ini`:

```ini
[Custom]
latent_dim = 10
hidden_dims = [512, 256]
beta = 4.0
l1_strength = 1e-3
lap_strength = 1e-4
epochs = 100
batch_size = 64
```

Save this file or reference the default config bundled with the package.

## Launch Training
```bash
bsvae-train pilot_run \
  --config src/bsvae/hyperparam.ini \
  --section Custom \
  --gene-expression-filename data/expression.csv
```

## What Happens Next?
- A directory `results/pilot_run/` is created automatically.
- `StructuredFactorVAE` is instantiated with the supplied latent dimensionality and hyperparameters.
- Training proceeds for the configured number of epochs, followed by evaluation on the test split.
- Metrics, training logs, and the serialized model checkpoint are stored under the experiment directory.

Resulting layout:
```
results/pilot_run/
├── model.pt
├── metadata.json
└── logs.txt
```

💡 **Tip:** Resume or inspect an existing experiment by re-running `bsvae-train` with `--is-eval-only` to skip training and recompute metrics.
