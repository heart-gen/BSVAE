# Utility API

This page summarizes the Python helpers most likely to be useful outside the CLI.

## Data Loading

From `src/bsvae/utils/datasets.py`:

### `OmicsDataset`

- one item per feature
- returns `(profile_tensor, feature_id)`
- expects a `features x samples` matrix
- supports `.csv`, `.tsv`, `.h5`, `.hdf5`, and `.h5ad`

### `get_omics_dataloader`

```python
from bsvae.utils.datasets import get_omics_dataloader

loader = get_omics_dataloader("data/expression.csv", batch_size=128)
```

## Training And Evaluation

From `src/bsvae/utils/training.py`:

- `Trainer` for warmup-plus-GMM training
- `Evaluator` for no-grad evaluation

The training loop writes `train_losses.csv` and optional checkpoint files.

## Model I/O

From `src/bsvae/utils/modelIO.py`:

- `save_model(model, directory, metadata=None)`
- `load_model(directory, is_gpu=True)`
- `load_metadata(directory)`
- `load_checkpoints(directory, is_gpu=True)`

Persisted model artifacts are `model.pt` and `specs.json`.

## Network Extraction Helpers

From `src/bsvae/networks/extract_networks.py`:

- `create_dataloader_from_expression(path, batch_size=128)`
- `extract_mu_gamma(model, dataloader)`
- `method_a_cosine(mu, top_k=50)`
- `method_b_gamma_knn(gamma, top_k=50)`
- `run_extraction(...)`

These helpers back the current `bsvae-networks extract-networks` workflow.

## Module Extraction Helpers

From `src/bsvae/networks/module_extraction.py`:

- `extract_gmm_modules(...)`
- `compute_module_eigengenes_from_soft(...)`
- `leiden_modules(...)`
- `save_modules(...)`

`extract_gmm_modules(...)` is the source of the `gamma.npz` and `hard_assignments.npz` outputs used by the CLI.

## Latent Utilities

From `src/bsvae/latent/latent_analysis.py` and `src/bsvae/latent/latent_export.py`:

- latent extraction helpers
- K-means and Gaussian-mixture clustering
- UMAP and t-SNE embeddings
- covariate-correlation utilities
- CSV and `.h5ad` export helpers for Python workflows
