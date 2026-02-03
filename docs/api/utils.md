# Utility API

BSVAE bundles helper classes and functions to streamline training, evaluation, graph-aware regularization, and post-training analysis. Below is an overview of the most commonly used utilities.

---

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

### `GeneExpression`

The underlying dataset class that handles gene expression matrices:

- Supports CSV, TSV, and gzip-compressed variants (`.csv.gz`, `.tsv.gz`)
- Automatically detects and corrects orientation to genes × samples
- Full-matrix mode: auto 10-fold CV split
- Pre-split mode: expects `X_train.csv` and `X_test.csv` in directory

---

## Training and Evaluation
### `Trainer`
Wraps a `StructuredFactorVAE`, optimizer, and loss function. Calling the trainer iterates over epochs, writes logs, and emits periodic checkpoints.

```python
from bsvae.utils import Trainer
trainer = Trainer(model, optimizer, loss_f, device=device, save_dir="results/my_experiment")
trainer(train_loader, epochs=100, checkpoint_every=10)
```

### `Evaluator`
Runs evaluation loops with shared logging semantics. The evaluator saves averaged losses to `test_losses.pt` (a Torch-serialized dictionary) and returns a dictionary of metrics.

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

---

## Network Analysis Utilities

### Gene Mapping

Located in `bsvae.utils.mapping`:

```python
from bsvae.utils.mapping import map_genes_to_proteins, resolve_gene_symbols

# Map gene symbols to STRING protein IDs
protein_map = map_genes_to_proteins(gene_list, taxid=9606)

# Resolve gene symbols using MyGene.info or BioMart
resolved = resolve_gene_symbols(gene_list, taxid=9606, method="mygene")
```

### Module Extraction

Located in `bsvae.networks.module_extraction`:

```python
from bsvae.networks.module_extraction import (
    leiden_modules,
    optimize_resolution_modularity,
    compute_module_eigengenes,
    PartitionResult,
)

# Cluster genes into modules
modules = leiden_modules(adjacency_matrix, resolution=1.0)

# Auto-optimize resolution
best_res, best_qual, modules = optimize_resolution_modularity(
    adjacency_matrix,
    resolution_min=0.5,
    resolution_max=1.5,
    n_steps=10,
    return_modules=True,
    n_jobs=4,  # Parallel execution
)

# Compute module eigengenes
eigengenes = compute_module_eigengenes(expression_df, modules)
```

### `PartitionResult`

A serializable container for Leiden partition results, used internally for
parallel execution with joblib:

```python
@dataclass
class PartitionResult:
    membership: List[int]  # Module assignments per gene
    quality: float         # Modularity score
```

### Network Extraction

Located in `bsvae.networks.extract_networks`:

```python
from bsvae.networks.extract_networks import (
    extract_network_from_model,
    compute_w_similarity,
    compute_latent_covariance,
)

# Extract gene-gene network using decoder weights
adjacency = compute_w_similarity(model.decoder.W)

# Full extraction with multiple methods
results = extract_network_from_model(
    model, dataloader,
    methods=["w_similarity", "latent_cov"],
    output_dir="networks/",
)
```

---

## Latent Analysis

Located in `bsvae.latent`:

```python
from bsvae.latent import export_latents, run_latent_analysis

# Export encoder outputs
export_latents(model, dataloader, output_path="latents.h5ad")

# Run sample-level analysis (clustering, UMAP, etc.)
run_latent_analysis(
    model, dataloader,
    output_dir="analysis/",
    kmeans_k=6,
    umap=True,
    covariates_path="covariates.csv",
)
```
