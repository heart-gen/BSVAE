# Tutorial

This tutorial walks through the current CLI workflow from data loading to post-training analysis and simulation benchmarking.

## 1. Confirm The Data Layout

BSVAE trains on feature profiles, not sample profiles. The main matrix must be `features x samples`.

- rows are feature IDs
- columns are sample IDs
- CSV and TSV files use the first column as the row index

Supported formats:

- `.csv` / `.csv.gz`
- `.tsv` / `.tsv.gz`
- `.h5` / `.hdf5`
- `.h5ad` with optional `anndata`

## 2. Run A Minimal Training Job

Use a short run first to confirm the install and data format.

```bash
bsvae-train tutorial_min \
  --dataset data/expression.csv \
  --epochs 5 \
  --batch-size 64 \
  --n-modules 8 \
  --latent-dim 12
```

Sanity checks:

- `results/tutorial_min/model.pt` exists
- `results/tutorial_min/specs.json` exists
- `results/tutorial_min/train_losses.csv` exists

## 3. Select The Number Of Modules

`--n-modules` controls the number of Gaussian-mixture components in the prior. For real analyses, the recommended path is `bsvae-sweep-k`.

```bash
bsvae-sweep-k sweep_prod \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

Key outputs:

- `results/sweep_prod/sweep_k/sweep_results.csv`
- `results/sweep_prod/sweep_k/sweep_summary.json`
- per-K replicate directories under `results/sweep_prod/sweep_k/k<K>/rep_<rep>/`
- final retrained model under `results/sweep_prod/final_k<K>/`

Selection behavior:

- with `--stability-reps 1`, the best `K` is chosen by validation loss
- with `--stability-reps > 1`, the best `K` is chosen by mean pairwise ARI on held-out features

## 4. Train A Final Model Directly

If you already know `K`, train directly with `bsvae-train`.

```bash
bsvae-train study1 \
  --dataset data/expression.csv \
  --epochs 120 \
  --n-modules 24 \
  --latent-dim 32
```

Useful flags to review for production runs:

- `--normalize-input`
- `--warmup-epochs`
- `--transition-epochs`
- `--free-bits`
- `--sep-strength`
- `--bal-strength`
- `--checkpoint-every`

## 5. Extract Feature Networks

`bsvae-networks extract-networks` builds sparse feature-feature graphs from trained latents.

```bash
bsvae-networks extract-networks \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Methods:

- `mu_cosine`: top-k cosine neighbors in latent mean space
- `gamma_knn`: FAISS-based kNN graph in soft-assignment space

Outputs are sparse adjacency files such as:

- `mu_cosine_adjacency.npz`
- `gamma_knn_adjacency.npz`

## 6. Extract Modules

`extract-modules` saves soft and hard GMM assignments. Add `--expr` and `--soft-eigengenes` to compute eigengenes.

```bash
bsvae-networks extract-modules \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Primary outputs:

- `gamma.npz`
- `hard_assignments.npz`
- `soft_eigengenes.csv` when requested

Optional comparison outputs:

- `leiden_modules.csv` with `--use-leiden`
- `gamma_gene.npz` and `hard_assignments_gene.npz` with `--aggregate-to-gene --tx2gene`

## 7. Export Latents

`export-latents` writes a compressed NumPy archive.

```bash
bsvae-networks export-latents \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output results/sweep_prod/final_k16/latents
```

Saved arrays:

- `mu`
- `logvar`
- `gamma`
- `feature_ids`

## 8. Analyze The Latent Space

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep_prod/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep_prod/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

Typical outputs:

- `latent_mu.csv`
- `latent_logvar.csv`
- `latent_clusters.csv` when clustering is requested
- `latent_embeddings.csv` when `--umap` or `--tsne` is used
- `latent_covariate_correlations.csv` when `--covariates` is provided

## 9. Generate Synthetic Data And Benchmark Recovery

Generate one dataset:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv
```

Benchmark a trained model against the ground truth:

```bash
bsvae-simulate benchmark \
  --dataset data/sim_expr.csv \
  --ground-truth data/sim_truth.csv \
  --model-path results/sweep_prod/final_k16 \
  --output results/sweep_prod/final_k16/sim_metrics.json
```

Generate a publication-style scenario grid:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13

bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Each replicate directory includes method-ready files such as:

- `expr/features_x_samples.tsv.gz`
- `expr/samples_x_features.tsv.gz`
- `truth/modules_hard.csv`
- `method_inputs.json`

## 10. Common Problems

- Data orientation is wrong: transpose sample-by-feature matrices before training.
- CUDA memory is tight: reduce `--batch-size` or use `--no-cuda`.
- `gamma_knn` fails: verify `faiss-cpu` is installed (required dependency, but may be missing in some custom envs).
- Hierarchical options fail: make sure `--tx2gene` matches the matrix row IDs.
- No eigengene file appears: `--soft-eigengenes` only writes output when `--expr` is supplied.

## 11. Legacy Configuration Note

The active CLI does not use `hyperparam.ini`. The files in `src/bsvae/hyperparam.ini` and `docs/hyperparam.ini` are retained for compatibility context only.

For reproducible runs, prefer shell scripts or workflow files with explicit CLI flags.
