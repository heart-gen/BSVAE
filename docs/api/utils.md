# Utility API

BSVAE bundles helper classes and functions to streamline training, evaluation, and graph-aware regularization. Below is an overview of the most commonly used utilities.

## Data Loading
### `get_dataloaders`
```python
from bsvae.utils.datasets import get_dataloaders
train_loader = get_dataloaders(
    dataset="genenet",
    batch_size=64,
    train=True,
    gene_expression_filename="data/expression.csv",
    logger=logger,
)
```
Loads a registered dataset (default `GeneExpression`) and returns a `torch.utils.data.DataLoader`. Accepts `gene_expression_filename` or `gene_expression_dir` along with flags such as `batch_size`, `shuffle`, and `drop_last`.

## Training and Evaluation
### `Trainer`
Wraps a `StructuredFactorVAE`, optimizer, and loss function. Calling the trainer iterates over epochs, writes logs, and emits periodic checkpoints.

```python
from bsvae.utils import Trainer
trainer = Trainer(model, optimizer, loss_f, device=device, save_dir="results/my_experiment")
trainer(train_loader, epochs=100, checkpoint_every=10)
```

### `Evaluator`
Runs evaluation loops with shared logging semantics. The evaluator saves averaged losses to `test_losses.log` and returns a dictionary of metrics.

```python
from bsvae.utils import Evaluator
evaluator = Evaluator(model, loss_f, device=device, save_dir="results/my_experiment")
metrics = evaluator(test_loader)
```

## PPI Priors
### `load_ppi_laplacian`
```python
from bsvae.utils.ppi import load_ppi_laplacian
L, G = load_ppi_laplacian(gene_list, taxid="9606", min_score=700, cache_dir="~/.bsvae/ppi")
```
Downloads STRING-DB edges, builds a NetworkX graph, and converts it to a Laplacian aligned with the provided gene list. Pass the Laplacian to `StructuredFactorVAE` for Laplacian regularization.

## Model I/O
### `save_model` / `load_model`
Use these helpers from `bsvae.utils.modelIO` to persist and restore experiments. `save_model(model, directory, metadata)` exports weights and metadata; `load_model(directory, is_gpu=True)` rebuilds the model on the desired device. Companion functions `save_metadata`, `load_metadata`, and `load_checkpoints` manage JSON metadata and intermediate checkpoints.

## Reproducibility Helpers
- `set_seed(seed)` – Sets NumPy, Python, and PyTorch seeds.
- `get_device(use_gpu=True)` – Returns `cuda` or `cpu` based on availability.
- `create_safe_directory(path)` – Archives existing experiment directories before creating new ones.
- `get_n_params(model)` – Counts trainable parameters.
- `update_namespace_(namespace, dictionary)` – Applies configuration overrides to argparse namespaces.

💡 **Tip:** Combine `set_seed`, `create_safe_directory`, and `load_ppi_laplacian` within custom scripts to reproduce the CLI workflow programmatically.
