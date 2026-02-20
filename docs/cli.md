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
- `--epochs` (default: `100`)
- `--batch-size` (default: `128`)
- `--lr` (default: `5e-4`)
- `--warmup-epochs` (default: `20`)
- `--transition-epochs` (default: `10`)
- `--n-modules` (default: `20`)
- `--latent-dim` (default: `32`)
- `--hidden-dims` (default: `[512, 256, 128]`)
- `--sigma-min` (default: `0.3`)
- `--beta` (default: `1.0`)
- `--free-bits` (default: `0.5`)
- `--sep-strength` / `--bal-strength` / `--hier-strength`
- `--tx2gene` (used with `--hier-strength > 0`)
- `--checkpoint-every` (default: `10`)
- `--no-eval` (skip evaluation pass)
- `--no-cuda`

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

### `export-latents`

```bash
bsvae-networks export-latents \
  --model-path results/run \
  --dataset data/expression.csv \
  --output results/run/latents.npz
```

Writes compressed NPZ with keys: `mu`, `logvar`, `gamma`, `feature_ids`.

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

## `bsvae-simulate`

Subcommands:

- `generate`: create synthetic expression + optional ground-truth labels
- `benchmark`: compute ARI/NMI from predicted module assignments

```bash
bsvae-simulate generate --output data/sim.csv --save-ground-truth data/gt.csv
bsvae-simulate benchmark --dataset data/sim.csv --ground-truth data/gt.csv --model-path results/run --output results/run/sim.json
```
