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

- `Trainer` for warmup-plus-GMM training, with optional mixed precision (`--amp`) and the neighbour-contrastive latent term (`--contrastive-strength`)
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

## Transcript-Level Helpers

From `src/bsvae/utils/isoform.py` and `src/bsvae/utils/hierarchy.py`:

- `load_tx2gene(path, ...)` and `group_isoforms_by_gene(...)`: read the transcript-to-gene map (headers auto-detected, Ensembl version suffixes stripped) and group isoforms
- `isoform_residuals(expr, tx2gene, ...)`: the gene-programme-removed residual matrix behind `isoform-residuals`
- `isoform_specific_calls(expr, tx2gene, labels, feature_ids, ...)`: the residual scan behind `isoform-specific`; returns a table with `kme_residual`, `p_value`, `q_value`, `label_discordant` and `called`. Options mirror the CLI: `min_kme`, `alpha`, `covariates`, `max_target_corr`, `loo_eigengene` (`"approx"`, `"exact"`, `"off"`)
- `cross_fit_calls(expr, tx2gene, labels_by_fold, fold_of_sample, feature_ids, ...)`: the cross-fitted version (per-fold z-scores combined with Stouffer's method before the best module is chosen; `require_module_agreement` mirrors the CLI's default)
- `IsoformStratifiedSampler`, `GeneGroupedBatchSampler`: batch samplers for the hierarchical loss

```python
from bsvae.utils.isoform import isoform_specific_calls

calls = isoform_specific_calls(expr, tx2gene, labels, expr.index, loo_eigengene="approx")
calls.loc[calls["called"], ["gene_id", "gene_module", "best_other_module", "q_value"]]
```

## Preprocessing

From `src/bsvae/utils/preprocess.py`:

- `size_factors(counts, method="median_ratio")`: DESeq-style size factors
- `pearson_residuals(...)`: negative-binomial Pearson residuals of raw counts (evaluated as an alternative input; log-normalised input remains the default)
- `design_matrix(covariates, columns=None)` and `regress_covariates(...)`: the least-squares covariate removal behind `regress-covariates`

## Post-Processing, Quality And Targeted Tests

From `src/bsvae/networks/postprocess.py`:

- `postprocess_assignments(gamma, feature_ids, expr, ...)`: grey class (`min_gamma`), minimum module size, eigengene merging and optional kME reassignment; returns a `PostprocessResult`
- `enforce_gene_coherence(...)`: the gene-coherence policy applied to those labels at transcript level
- `consensus_assignments(...)` and `align_labels(...)`: majority vote over seeds

From `src/bsvae/networks/quality.py`: `module_quality(...)`, `module_preservation(...)`, `module_enrichment(...)`.

From `src/bsvae/networks/targets.py`: `read_bed(...)`, `read_gene_annotation(...)`, `loci_to_genes(...)`, `resolve_target_transcripts(...)`, `target_gene_tests(...)` and `combine_target_tests(...)`, the engines of `target-genes` and `target-genes-combine`.

## Simulation And Benchmarking

From `src/bsvae/simulation/`: `generate.py` (gene-level simulator), `isoform.py` and `scenario.py` (transcript-level scenarios and `generate_scenario`), `metrics.py` (`evaluate_labels`, `coverage_metrics`, `gene_level_metrics`, `switched_isoform_metrics`).

From `src/bsvae/benchmark/`: `harness.py` (the grid runner behind `bsvae-benchmark`), `runners.py` (`run_bsvae`, WGCNA and acorde runners), `methods.py` (usage network, ICA, NMF), `edges.py` (edge metrics) and the packaged R scripts `wgcna.R`, `acorde.R`, `twn.R`. See [Benchmarking](../benchmarking.md).

## Latent Utilities

From `src/bsvae/latent/latent_analysis.py` and `src/bsvae/latent/latent_export.py`:

- latent extraction helpers
- K-means and Gaussian-mixture clustering
- UMAP and t-SNE embeddings
- covariate-correlation utilities
- CSV and `.h5ad` export helpers for Python workflows
