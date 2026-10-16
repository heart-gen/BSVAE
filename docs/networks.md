# Network And Latent Workflows

This is the canonical reference for `bsvae-networks`.

## Prerequisites

- a trained model directory containing `model.pt` and `specs.json`
- the same expression matrix orientation used for training: `features x samples`

## Extract Networks

`extract-networks` builds sparse feature-feature adjacency matrices from learned latents.

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Methods:

- `mu_cosine`: cosine similarity among latent means, sparsified to top-k neighbors
- `gamma_knn`: FAISS-based kNN graph built from normalized GMM soft assignments

Outputs:

- `mu_cosine_adjacency.npz`
- `gamma_knn_adjacency.npz`

## Extract Modules

`extract-modules` writes soft and hard assignments from the fitted GMM.

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules
```

Primary outputs:

- `gamma.npz`
- `hard_assignments.npz`

Add eigengenes:

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Post-processing (on by default) mirrors WGCNA: a confidence cutoff (`--min-gamma`) leaves uncertain features unassigned (grey, label `-1`), undersized modules are dissolved (`--min-module-size`), and modules whose eigengenes correlate above `1 - --merge-cut-height` are merged. Pass `--no-postprocess` for raw `argmax(γ)`.

Outputs:

- `modules.csv` (`feature_id, module, module_label`), `eigengenes.csv`, `kME.csv`, `merge_log.json`, `gamma_merged.npz`
- `gamma.npz`, `hard_assignments.npz` (raw)
- `soft_eigengenes.csv` with `--soft-eigengenes`
- `leiden_modules.csv` with `--use-leiden`
- `gamma_gene.npz`, `hard_assignments_gene.npz` and `gene_divergent.csv` (genes whose isoforms land in two or more modules) with `--aggregate-to-gene --tx2gene`

## Consensus Over Seeds

```bash
bsvae-networks consensus-modules --modules run_s0/modules.csv run_s1/modules.csv run_s2/modules.csv --output-dir consensus
```

Writes `consensus_modules.csv` with a per-feature `agreement` stability score; features below `--min-agreement` are unassigned.

## Isoform-Specific Membership

```bash
bsvae-networks isoform-specific --dataset transcripts.tsv.gz --tx2gene tx2gene.tsv \
  --modules results/run/modules/modules.csv --output results/run/isoform_specific.csv
```

For each multi-isoform transcript, the part of its profile that its sibling isoforms do not explain is correlated with every module eigengene (each residualised on the gene's own module eigengene, which removes leakage between correlated modules); a transcript whose residual tracks another module (kME ≥ `--min-kme`, BH FDR < `--alpha`) is called isoform-specific, with the target module. On the isoform simulations this enriches truly switched isoforms about 8× over the base rate at the default `--min-kme 0.3`. Low-abundance isoforms have attenuated correlations and are the hardest to call; see `docs/cli.md` for what helps. `isoform-residuals` writes the residual matrix for custom analyses.

## Export Latents

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents
```

This writes `<output>.npz` containing:

- `mu`
- `logvar`
- `gamma`
- `feature_ids`

## Latent Analysis

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --kmeans-k 10 \
  --umap
```

Possible outputs:

- `latent_mu.csv`
- `latent_logvar.csv`
- `latent_clusters.csv`
- `latent_embeddings.csv`
- `latent_covariate_correlations.csv`

## Performance Notes

- `mu_cosine` scales with the number of features and can be expensive on large matrices.
- `gamma_knn` requires `faiss-cpu`.
- For memory-constrained runs, reduce `--batch-size` and consider `--no-cuda`.
