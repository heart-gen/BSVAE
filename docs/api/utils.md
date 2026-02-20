# Utility API

## Data loading

### `OmicsDataset`

From `src/bsvae/utils/datasets.py`.

- One row entity per item (feature-level training)
- Returns `(profile_tensor, feature_id)`
- Accepts `.csv`, `.tsv`, `.h5/.hdf5`, `.h5ad`

### `get_omics_dataloader`

```python
from bsvae.utils.datasets import get_omics_dataloader
loader = get_omics_dataloader("data/expression.csv", batch_size=128)
```

## Training/evaluation

From `src/bsvae/utils/training.py`:

- `Trainer`: two-phase warmup + GMM transition training loop
- `Evaluator`: no-grad loss evaluation loop

Trainer writes `train_losses.csv` and checkpoint files.

## Model I/O

From `src/bsvae/utils/modelIO.py`:

- `save_model(model, directory, metadata=None)`
- `load_model(directory, is_gpu=True)`
- `load_metadata(directory)`
- `load_checkpoints(directory, is_gpu=True)`

Persisted files are `model.pt` and `specs.json`.

## Network extraction helpers

From `src/bsvae/networks/extract_networks.py`:

- `create_dataloader_from_expression(path, batch_size=128)`
- `extract_mu_gamma(model, dataloader)`
- `method_a_cosine(mu, top_k=50)`
- `method_b_gamma_knn(gamma, top_k=50)`
- `run_extraction(...)`

## Module extraction helpers

From `src/bsvae/networks/module_extraction.py`:

- `extract_gmm_modules(...)`
- `compute_module_eigengenes_from_soft(...)`
- `leiden_modules(...)`
- `save_modules(...)`

## Latent utilities

From `src/bsvae/latent/latent_analysis.py` and `src/bsvae/latent/latent_export.py`:

- latent extraction
- k-means / GMM clustering
- UMAP / t-SNE embeddings
- covariate correlation tables
- CSV/H5AD export helpers (API-level)
