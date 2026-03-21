# Command-Line Interface

This reference documents the installed CLI entry points defined in `pyproject.toml`.

## `bsvae-train`

Train a `GMMModuleVAE` on a `features x samples` matrix.

```bash
bsvae-train NAME --dataset PATH [options]
```

Required:

- `NAME`: experiment name
- `--dataset`: expression matrix path

Common options:

- `--outdir` default `results`
- `--seed` default `13`
- `--epochs` default `100`
- `--batch-size` default `128`
- `--lr` default `5e-4`
- `--checkpoint-every` default `10`
- `--warmup-epochs` default `20`
- `--transition-epochs` default `10`
- `--freeze-gmm-epochs` default `0`
- `--n-modules` default `20`
- `--latent-dim` default `32`
- `--hidden-dims` default `[512, 256, 128]`
- `--dropout` default `0.1`
- `--use-batch-norm` / `--no-batch-norm`
- `--sigma-min` default `0.3`
- `--normalize-input`
- `--beta` default `1.0`
- `--free-bits` default `0.0`
- `--kl-warmup-epochs` default `0`
- `--kl-anneal-mode` `linear` or `cyclical`
- `--kl-cycle-length` default `50`
- `--sep-strength` default `0.1`
- `--sep-alpha` default `2.0`
- `--bal-strength` default `0.1`
- `--bal-ema-blend` default `0.5`
- `--pi-entropy-strength` default `0.0`
- `--hier-strength` default `0.0`
- `--corr-strength` default `0.0`
- `--latent-corr-strength` default `0.0`
- `--masked-recon`
- `--tx2gene`
- `--isoform-stratified`
- `--p-multi` default `0.5`
- `--collapse-threshold` default `0.5`
- `--collapse-noise-scale` default `0.5`
- `--no-eval`
- `--eval-batch-size`
- `--no-cuda`
- `--log-level`
- `--no-progress-bar`

Outputs:

- `model.pt`
- `specs.json`
- `train_losses.csv`
- `model-<epoch>.pt` when checkpointing is enabled

## `bsvae-sweep-k`

Run a held-out validation sweep over candidate `K` values and optionally retrain the selected model on the full dataset.

```bash
bsvae-sweep-k NAME --dataset PATH [options]
```

Key options:

- `--k-grid` comma list or `start:end:step`
- `--sweep-epochs` default `60`
- `--stability-reps` default `1`
- `--stability-seed` default `13`
- `--val-frac` default `0.1`
- `--val-seed` default `13`
- `--train-final` enabled by default
- `--no-train-final`
- `--final-epochs`

Training-related flags mirror `bsvae-train` for architecture and optimization.

Selection behavior:

- `--stability-reps 1`: select the best `K` by validation loss
- `--stability-reps > 1`: select the best `K` by mean pairwise ARI across held-out-feature assignments

Outputs:

- `results/<name>/sweep_k/sweep_results.csv`
- `results/<name>/sweep_k/sweep_summary.json`
- `results/<name>/sweep_k/k<K>/rep_<rep>/...`
- `results/<name>/final_k<K>/...` when final retraining is enabled

## `bsvae-networks`

Post-training utilities for trained models.

Subcommands:

- `extract-networks`
- `extract-modules`
- `export-latents`
- `latent-analysis`

### `extract-networks`

Build sparse feature-feature graphs from latent outputs.

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Options:

- `--methods` choices: `mu_cosine`, `gamma_knn`
- `--top-k` default `50`
- `--batch-size` default `128`
- `--no-cuda`

Outputs:

- `<method>_adjacency.npz`

### `extract-modules`

Extract GMM assignments and optional eigengenes or comparison clusters.

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Options:

- `--expr` expression matrix for eigengene computation
- `--soft-eigengenes`
- `--use-leiden`
- `--leiden-resolution` default `1.0`
- `--tx2gene`
- `--aggregate-to-gene`
- `--batch-size` default `128`
- `--no-cuda`

Outputs:

- `gamma.npz`
- `hard_assignments.npz`
- `soft_eigengenes.csv` when requested
- `leiden_modules.csv` when requested
- `gamma_gene.npz` and `hard_assignments_gene.npz` when gene aggregation is requested

### `export-latents`

Export latent arrays for all features.

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents
```

Options:

- `--batch-size` default `128`
- `--no-cuda`

Output:

- `latents.npz` or `<output>.npz` with arrays `mu`, `logvar`, `gamma`, and `feature_ids`

### `latent-analysis`

Run clustering, embeddings, and optional covariate correlations on latent outputs.

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --kmeans-k 10 \
  --umap
```

Options:

- `--kmeans-k`
- `--gmm-k`
- `--umap`
- `--tsne`
- `--tsne-perplexity` default `30.0`
- `--covariates`
- `--batch-size` default `128`
- `--no-cuda`

Outputs may include:

- `latent_mu.csv`
- `latent_logvar.csv`
- `latent_clusters.csv`
- `latent_embeddings.csv`
- `latent_covariate_correlations.csv`

## `bsvae-simulate`

Simulation and benchmarking utilities.

Subcommands:

- `generate`
- `benchmark`
- `init-config`
- `generate-grid`
- `generate-scenario`
- `validate-grid`

### `generate`

```bash
bsvae-simulate generate \
  --output data/sim.csv \
  --save-ground-truth data/gt.csv
```

Important options:

- `--n-features` default `500`
- `--n-samples` default `200`
- `--n-modules` default `10`
- `--within-corr` default `0.8`
- `--between-corr` default `0.0`
- `--noise-std` default `0.2`
- `--seed` default `13`

### `benchmark`

```bash
bsvae-simulate benchmark \
  --dataset data/sim.csv \
  --ground-truth data/gt.csv \
  --model-path results/run \
  --output results/run/sim_metrics.json
```

Outputs JSON metrics including `ari`, `nmi`, and `n_features`.

### Scenario-grid commands

```bash
bsvae-simulate init-config --output sim.yaml
bsvae-simulate generate-grid --config sim.yaml --outdir results/sim_pub_v1 --reps 30 --base-seed 13
bsvae-simulate generate-scenario --config sim.yaml --scenario-id S001__... --rep 0 --outdir results/sim_pub_v1
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Per-replicate outputs are written under:

`<outdir>/scenarios/<scenario_id>/rep_<rep>/`

Common files:

- `expr/features_x_samples.tsv.gz`
- `expr/samples_x_features.tsv.gz`
- `truth/modules_hard.csv`
- `method_inputs.json`
