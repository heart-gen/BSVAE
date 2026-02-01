# BSVAE Networks & Latent Utilities

This page documents the `bsvae-networks` command-line interface, which provides **post-training utilities** for extracting gene–gene networks, gene modules, and sample-level latent representations from a trained **BSVAE** model.

The tool is designed to be used **after model training** and does not retrain the model.

---

## Overview

`bsvae-networks` supports four main workflows:

1. **Extract gene–gene networks** from a trained model
2. **Cluster networks into gene modules** (WGCNA-like)
3. **Export latent variables** (μ, log σ², z)
4. **Analyze latent space** at the sample level

```text
Train BSVAE
   ↓
bsvae-networks extract-networks
   ↓
bsvae-networks extract-modules
```

or

```text
Train BSVAE
   ↓
bsvae-networks export-latents
   ↓
bsvae-networks latent-analysis
```

---

## Prerequisites

You must have:

* A trained BSVAE model directory containing:

  * `model.pt` (or checkpoint)
  * `specs.json`
* A gene expression matrix with shape **genes × samples**

Example:

```text
results/
└── simulation_final/
    ├── model.pt
    └── specs.json
```

---

## Command Summary

```bash
bsvae-networks <command> [options]
```

Available commands:

* `extract-networks`
* `extract-modules`
* `export-latents`
* `latent-analysis`

---

## 1. Extract Gene–Gene Networks

Compute gene–gene adjacency matrices using decoder weights, latent covariance, or Laplacian structure.

### Example

```bash
bsvae-networks extract-networks \
  --model-path results/simulation_final \
  --dataset data/log2rpkm.tsv.gz \
  --output-dir networks/ \
  --methods w_similarity laplacian \
  --threshold 0.1 \
  --heatmaps
```

### What this does

* Loads a trained BSVAE
* Computes adjacency matrices using selected methods:

  * `w_similarity` (decoder weight similarity)
  * `latent_cov`
  * `graphical_lasso`
  * `laplacian`
* Saves adjacency matrices and optional edge lists

### Outputs

```text
networks/
├── w_similarity_adjacency.csv
├── w_similarity_edges.tsv
├── laplacian_adjacency.csv
└── heatmaps/
```

---

### ⚠️ Memory Notes (Network Extraction)

**Potentially high memory usage** occurs when:

* Number of genes > ~15,000
* Dense adjacency matrices are computed (`N_genes × N_genes`)
* Multiple methods are run simultaneously

**Recommendations:**

* Run **one method at a time** for large datasets
* Avoid `heatmaps` for large gene sets
* Prefer `w_similarity` or `laplacian` over `graphical_lasso`
* Run on nodes with ≥32–64 GB RAM for whole-transcriptome data

---

## 2. Extract Gene Modules (WGCNA-like)

Adjacency matrices are **continuous** and must be clustered to obtain discrete modules.

### Option A: Use a precomputed adjacency

```bash
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --cluster-method leiden \
  --resolution 1.0
```

### Option B: Compute adjacency on the fly

```bash
bsvae-networks extract-modules \
  --model-path results/simulation_final \
  --dataset data/log2rpkm.tsv.gz \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --cluster-method spectral \
  --n-clusters 20
```

### Option C: Automatic resolution selection

Let BSVAE find the optimal Leiden resolution by maximizing modularity:

```bash
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --cluster-method leiden \
  --resolution-auto
```

This searches resolutions in [0.5, 1.5] (configurable) and selects the one
with highest modularity score. No ground truth labels are required.

**Tuning the search:**

```bash
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --resolution-auto \
  --resolution-min 0.1 \
  --resolution-max 2.0 \
  --resolution-steps 15
```

**Parallel execution for HPC:**

```bash
# Use all available CPUs
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --resolution-auto \
  --n-jobs -1

# Or specify core count (match your SLURM --cpus-per-task)
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --resolution-auto \
  --n-jobs 16
```

### Option D: Resolution sweep

Run clustering at multiple resolutions to compare results:

```bash
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --resolutions 0.5 0.75 1.0 1.25 1.5
```

This creates separate subdirectories for each resolution. You can combine
`--resolution-auto` with `--resolutions` to include both auto-selected and
fixed resolutions in one run.

**Parallel resolution sweep:**

```bash
bsvae-networks extract-modules \
  --adjacency networks/w_similarity_adjacency.csv \
  --expr data/log2rpkm.tsv.gz \
  --output-dir modules/ \
  --resolutions 0.5 0.75 1.0 1.25 1.5 \
  --n-jobs -1
```

### Clustering Options

| Option | Description |
| ------ | ----------- |
| `--cluster-method` | `leiden` (default) or `spectral` |
| `--resolution` | Leiden resolution parameter (default: 1.0) |
| `--resolution-auto` | Auto-select resolution by maximizing modularity |
| `--resolutions` | Run multiple resolutions (e.g., `1.0 2.0 5.0`) |
| `--resolution-min` | Min resolution for auto search (default: 0.5) |
| `--resolution-max` | Max resolution for auto search (default: 1.5) |
| `--resolution-steps` | Number of steps in auto search (default: 10) |
| `--n-jobs` | Parallel jobs for resolution sweeps (-1 = all CPUs, default: 1) |
| `--n-clusters` | Number of clusters for spectral method |
| `--n-components` | Number of eigenvectors for spectral method |
| `--adjacency-mode` | `wgcna-signed` (default) or `signed` |

### Handling Negative Edge Weights

BSVAE networks may contain negative edge weights (anti-correlated genes).
Leiden clustering requires non-negative weights.

* **`wgcna-signed`** (default): Clips negative weights to zero before clustering.
  This matches WGCNA's signed network behavior.
* **`signed`**: Preserves negative weights. Not supported for Leiden; use with
  spectral clustering (may cause numerical issues).

### Outputs

**Single resolution:**

```text
modules/
├── modules.csv              # gene → module
├── eigengenes.csv           # samples × modules
└── clustering_metadata.json # resolution, n_modules, etc.
```

**Resolution sweep or auto:**

```text
modules/
├── res_auto/                # auto-selected resolution
│   ├── modules.csv
│   ├── eigengenes.csv
│   └── clustering_metadata.json
├── res_1_0/                 # resolution 1.0
│   └── ...
├── res_5_0/                 # resolution 5.0
│   └── ...
├── resolution_sweep_summary.json
└── resolution_sweep_summary.tsv
```

---

### ⚠️ Memory Notes (Module Extraction)

High memory usage may occur during:

* **Leiden clustering** on dense graphs
* **Spectral clustering** (eigen decomposition)

**Recommendations:**

* Prefer **Leiden** for large graphs
* For spectral clustering:

  * Explicitly set `--n-clusters`
  * Use smaller `--n-components`
* Consider thresholding adjacency before clustering

---

## 3. Export Latent Variables

Extract encoder outputs for samples.

```bash
bsvae-networks export-latents \
  --model-path results/simulation_final \
  --dataset data/log2rpkm.tsv.gz \
  --output latents.h5ad
```

### Output

* `mu`
* `logvar`
* sample IDs

---

### ⚠️ Memory Notes (Latent Export)

* Latent export is generally **low memory**
* Memory scales with `n_samples × latent_dim`
* Safe for large datasets

---

## 4. Latent Space Analysis

Perform clustering, dimensionality reduction, and covariate association.

### Example

```bash
bsvae-networks latent-analysis \
  --model-path results/simulation_final \
  --dataset data/log2rpkm.tsv.gz \
  --covariates data/covariates.tsv \
  --output-dir latent_analysis/ \
  --kmeans-k 6 \
  --umap
```

### Outputs

```text
latent_analysis/
├── mu.csv
├── logvar.csv
├── clusters.csv
├── umap.csv
└── covariate_correlations.csv
```

---

### ⚠️ Memory Notes (Latent Analysis)

Potential memory hotspots:

* **UMAP / t-SNE** with many samples
* **Covariate correlation** with many covariates

**Recommendations:**

* Use UMAP over t-SNE for large `n_samples`
* Subset samples if exploratory
* Avoid dense covariate matrices

---

## Recommended Workflows

### WGCNA-like Gene Module Discovery

```text
Train BSVAE
   ↓
extract-networks (w_similarity)
   ↓
extract-modules (leiden)
   ↓
eigengene × phenotype analysis
```

### Sample-Level Latent Analysis

```text
Train BSVAE
   ↓
export-latents
   ↓
latent-analysis (UMAP / clustering)
```

---

## GPU Acceleration

BSVAE network utilities use GPU acceleration where possible via PyTorch.

| Operation | GPU | Notes |
| --------- | --- | ----- |
| Model inference (encoder) | ✅ | Automatic if CUDA available |
| W similarity (`w_similarity`) | ✅ | PyTorch matrix operations |
| Latent covariance (`latent_cov`) | ✅ | PyTorch matrix operations |
| Laplacian refinement (`laplacian`) | ✅ | PyTorch matrix operations |
| Graphical Lasso (`graphical_lasso`) | ⚠️ Partial | Reconstruction on GPU, fitting on CPU (sklearn) |
| Leiden clustering | ❌ | CPU only (leidenalg library) |
| Spectral clustering | ❌ | CPU only (sklearn library) |
| Eigengene computation (PCA) | ❌ | CPU only (sklearn library) |

GPU is used automatically when available. No configuration is required.

---

## Key Design Notes

* **Adjacency ≠ modules**
  Clustering is always required.
* **Leiden is the default recommendation**
  More stable than hierarchical clustering for noisy graphs.
* **Latent μ describes samples, not genes**
* **Architecture tuning should precede network extraction**

---

## Summary Table

| Task             | Command            | Memory Risk      |
| ---------------- | ------------------ | ---------------- |
| Extract networks | `extract-networks` | ⚠️ High          |
| Cluster modules  | `extract-modules`  | ⚠️ Moderate–High |
| Export latents   | `export-latents`   | ✅ Low            |
| Latent analysis  | `latent-analysis`  | ⚠️ Moderate      |

---
