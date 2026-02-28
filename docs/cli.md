# Command-Line Interface

## `bsvae-train`

Train a `GMMModuleVAE` model.

```bash
bsvae-train NAME --dataset PATH [options]
```

### Required arguments

- `NAME`: experiment name (output under `results/NAME` by default)
- `--dataset`: expression matrix path (`features x samples`)

### Common options

- `--outdir` (default: `results`)
- `--seed` (default: `13`)
- `--epochs` (default: `100`)
- `--batch-size` (default: `128`)
- `--lr` (default: `5e-4`)
- `--warmup-epochs` (default: `20`)
- `--transition-epochs` (default: `10`)
- `--freeze-gmm-epochs` (default: `0`)
- `--n-modules` (default: `20`)
- `--latent-dim` (default: `32`)
- `--hidden-dims` (default: `[512, 256, 128]`)
- `--sigma-min` (default: `0.3`)
- `--beta` (default: `1.0`)
- `--free-bits` (default: `0.5`)
- `--kl-warmup-epochs` (default: `0`)
- `--kl-anneal-mode` (`linear` or `cyclical`)
- `--kl-cycle-length` (default: `50`)
- `--sep-strength` / `--bal-strength` / `--hier-strength`
- `--bal-ema-blend` (default: `0.5`)
- `--pi-entropy-strength` (default: `0.0`)
- `--collapse-threshold` / `--collapse-noise-scale`
- `--tx2gene` (used with `--hier-strength > 0`)
- `--checkpoint-every` (default: `10`)
- `--no-eval` (skip evaluation pass)
- `--eval-batch-size`
- `--no-cuda`
- `--log-level` (`debug`, `info`, `warning`, `error`)
- `--no-progress-bar`

## `bsvae-networks`

Subcommands:

- `extract-networks`
- `extract-modules`
- `export-latents`
- `latent-analysis`

### `extract-networks`

```bash
bsvae-networks extract-networks \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/networks \
  --methods mu_cosine gamma_knn
```

Supported `--methods` values: `mu_cosine`, `gamma_knn`.
Other options:

- `--top-k` (default: `50`)
- `--batch-size` (default: `128`)
- `--no-cuda`

### `extract-modules`

```bash
bsvae-networks extract-modules \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/modules
```

Optional:

- `--soft-eigengenes --expr <features x samples csv/tsv>`
- `--use-leiden --leiden-resolution <float>`
- `--batch-size` (default: `128`)
- `--no-cuda`

### `export-latents`

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents.npz
```

Writes compressed NPZ with keys: `mu`, `logvar`, `gamma`, `feature_ids`.
Other options:

- `--batch-size` (default: `128`)
- `--no-cuda`

### `latent-analysis`

```bash
bsvae-networks latent-analysis \
  --model-path results/run \
  --dataset data/expression.csv \
  --output-dir results/run/latent_analysis \
  --kmeans-k 10 --umap
```

Optional:

- `--gmm-k`
- `--tsne --tsne-perplexity`
- `--covariates` (CSV/TSV indexed by row IDs)
- `--batch-size` (default: `128`)
- `--no-cuda`

## `bsvae-simulate`

Subcommands:

- `generate`: create synthetic expression + optional ground-truth labels
- `benchmark`: compute ARI/NMI from predicted module assignments
- `init-config`: write a starter scenario-grid config file
- `generate-grid`: generate all scenario/replicate outputs from config
- `generate-scenario`: generate one scenario/replicate
- `validate-grid`: validate generated grid structure

```bash
bsvae-simulate generate --output data/sim.csv --save-ground-truth data/gt.csv
bsvae-simulate benchmark --dataset data/sim.csv --ground-truth data/gt.csv --model-path results/run --output results/run/sim.json

bsvae-simulate init-config --output sim.yaml
bsvae-simulate generate-grid --config sim.yaml --outdir results/sim_pub_v1 --reps 30 --base-seed 13
bsvae-simulate generate-scenario --config sim.yaml --scenario-id S001__confounding-none__n_samples-100__nonlinear_mode-off__overlap_rate-0.0__signal_scale-0.4 --rep 0 --outdir results/sim_pub_v1
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Note: `--config` currently expects JSON-compatible YAML (the generated `sim.yaml` uses this format).

### `generate` options

- `--n-features` (default: `500`)
- `--n-samples` (default: `200`)
- `--n-modules` (default: `10`)
- `--within-corr` (default: `0.8`)
- `--between-corr` (default: `0.0`)
- `--noise-std` (default: `0.2`)
- `--seed` (default: `13`)
- `--save-ground-truth` (CSV with module labels)

### `benchmark` options

- `--batch-size` (default: `128`)
- `--no-cuda`

Scenario-grid outputs are written to:
`<outdir>/scenarios/<scenario_id>/rep_<rep>/`

- `expr/features_x_samples.tsv.gz` (BSVAE/GNVAE input)
- `expr/samples_x_features.tsv.gz` (WGCNA input)
- `covariates.tsv.gz`
- `truth/modules_hard.csv`
- `truth/modules_long.csv`
- `truth/module_latents.tsv.gz`
- `truth/edge_list.tsv.gz`
- `truth/gene_metadata.tsv.gz`
- `metadata.json`
- `method_inputs.json`

## `bsvae-sweep-k`

K sweep with a held-out validation split. Optional stability mode runs multiple
replicates per K and selects the most stable module assignments.

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --val-frac 0.1
```

Stability mode (recommended for module discovery):

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

Key options:

- `--k-grid` (comma list or `start:end:step`)
- `--sweep-epochs`
- `--stability-reps` / `--stability-seed`
- `--val-frac` / `--val-seed`
- `--train-final` (default: on; retrains best K on full data)

Outputs are written under: `results/<name>/sweep_k/`
