# BSVAE: Gaussian Mixture VAE for Module Discovery

[![PyPI](https://img.shields.io/pypi/v/bsvae.svg)](https://pypi.org/project/bsvae/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/bsvae/badge/?version=latest)](https://bsvae.readthedocs.io/)

BSVAE is a PyTorch package for feature-level module discovery in omics data. The main model, `GMMModuleVAE`, learns latent structure across features such as genes, transcripts, or proteins, then supports downstream network extraction, module assignment, latent analysis, and simulation-based benchmarking.

## What The Package Provides

- `bsvae-train` for training a two-phase GMM-VAE
- `bsvae-sweep-k` for selecting the number of modules (`K`) with held-out validation and optional stability replicates
- `bsvae-networks` for post-training network extraction, module extraction (with gene coherence and an isoform-specific residual scan at transcript level), targeted-gene tests, latent export, and latent analysis
- `bsvae-simulate` for synthetic data generation, scenario-grid simulation, and module-recovery benchmarking
- `bsvae-benchmark` for head-to-head runs of BSVAE, WGCNA, acorde, usage networks, ICA/NMF and TWN on simulated scenario grids

## Installation

Install from PyPI:

```bash
pip install bsvae
```

Install from source:

```bash
git clone https://github.com/heart-gen/BSVAE.git
cd BSVAE
pip install -e .
```

Optional dependencies:

- `anndata` is only needed for `.h5ad` input or `.h5ad` latent export helpers
- `pyyaml` is only needed to read YAML scenario configs (JSON configs need nothing)
- R with `WGCNA` (plus `glasso` for TWN) is only needed for the WGCNA, acorde and TWN methods of `bsvae-benchmark`; check with `bsvae-benchmark check-env`

```bash
pip install "bsvae[anndata,yaml]"
```

For development, use Poetry: `poetry install --with dev --extras "anndata yaml"`.

## Input Data Contract

Training and most downstream commands expect an expression matrix in `features x samples` orientation:

- rows are features such as genes, transcripts, or proteins
- columns are sample IDs
- the first column is the feature index when using CSV/TSV

Supported loaders:

- `.csv` / `.csv.gz`
- `.tsv` / `.tsv.gz`
- `.h5` / `.hdf5`
- `.h5ad` with optional `anndata`

## CLI Entry Points

- `bsvae-train`
- `bsvae-sweep-k`
- `bsvae-networks`
- `bsvae-simulate`
- `bsvae-benchmark` (BSVAE vs WGCNA, transcript-usage networks, ICA/NMF, acorde and TWN on simulated data; needs R with `WGCNA`, plus `glasso` for TWN)

At transcript level, `extract-modules` returns two independent outputs: coherent module assignments (the gene-coherence policy is the default whenever `--expr` and `--tx2gene` are given) and the isoform-specific residual scan over every multi-isoform transcript, so module membership and isoform-specific divergence are reported and validated separately. Further transcript-level commands in `bsvae-networks`: `consensus-modules` (majority vote over seeds with a per-feature stability score), `isoform-specific` (isoforms whose expression tracks a module other than their gene's), `target-genes` (calibrated membership tests for candidate genes or GWAS loci from a gene list, transcript list or BED file, with a case/control interaction test), `target-genes-combine` (pool those tests across cohorts or brain regions), `module-quality` (kME, enrichment, preservation in a second dataset) and `regress-covariates` (remove batch and other known covariates before training).

### `bsvae-networks` subcommands

| Subcommand | Purpose |
| --- | --- |
| `extract-networks` | feature-feature networks (`mu_cosine`, `gamma_knn`) |
| `extract-modules` | post-processed modules, eigengenes, kME; at transcript level also gene coherence and the isoform-specific scan |
| `consensus-modules` | majority vote over seeds with a per-feature stability score |
| `isoform-specific` | isoforms whose expression tracks a module other than their gene's, optionally cross-fitted (`--folds`) |
| `isoform-residuals` | the gene-programme-removed residual matrix |
| `target-genes`, `target-genes-combine` | calibrated tests for candidate genes or GWAS loci, and pooling across cohorts |
| `module-quality` | kME, GMT enrichment, preservation in a second dataset |
| `regress-covariates` | remove batch and other known covariates before training |
| `export-latents`, `latent-analysis` | latent arrays, clustering and embeddings |

Flags worth knowing at transcript level: `--tx2gene`, `--covariates`, `--loo-eigengene {approx,exact,off}`, `--max-target-corr`, `--samples` / `--exclude-samples` (train fold models for cross-fitting), `--aggregate-to-gene`, and `--amp` on `bsvae-train` for mixed precision. See the [CLI reference](https://bsvae.readthedocs.io/en/latest/cli.html) for every flag and default.

## Quickstart

Train a small model:

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

Recommended production flow: train with a generous number of GMM components, about three times the number of modules you expect (at least 20), and let `extract-modules` merge components whose eigengenes correlate (see [Choosing K](https://bsvae.readthedocs.io/en/latest/hyperparameters.html#choosing-k)). `bsvae-sweep-k` is an optional check of that component count by held-out ELBO and, with `--stability-reps`, by seed stability; it retrains the selected model so the steps below can start from it.

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

This writes sweep outputs under `results/sweep1/sweep_k/` and retrains the selected model under `results/sweep1/final_k<K>/`.

Extract module assignments from the final model (post-processing adds a grey class, a minimum module size and eigengene merging):

```bash
bsvae-networks extract-modules \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

With transcript-level data, pass `--tx2gene` as well. `extract-modules` then produces two independent outputs: module assignments annotated by the gene-coherence policy (default; `--no-gene-coherence` to disable) and the isoform-specific residual scan (`--no-isoform-calls` to skip, `--loo-eigengene` to change the leave-one-gene-out eigengene correction). A transcript's module label never decides whether it is tested. For calibrated calls, pass technical covariates (`--covariates`) and, where modules were learned on the same samples, cross-fit over sample folds (`isoform-specific --folds`).

```bash
bsvae-networks extract-modules \
  --model-path results/iso1 \
  --dataset data/transcripts.tsv.gz \
  --output-dir results/iso1/modules \
  --expr data/transcripts.tsv.gz \
  --tx2gene data/tx2gene.tsv
```

This adds `modules_raw.csv`, `gene_coherence.csv` and `isoform_specific.csv` next to `modules.csv`; see the [CLI reference](docs/cli.md) for the columns.

Export latents:

```bash
bsvae-networks export-latents \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output results/sweep1/final_k16/latents
```

This command writes `results/sweep1/final_k16/latents.npz` containing `mu`, `logvar`, `gamma`, and `feature_ids`.

Extract feature-feature networks:

```bash
bsvae-networks extract-networks \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/networks \
  --methods mu_cosine gamma_knn \
  --top-k 50
```

Run latent-space analysis:

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/latent_analysis \
  --kmeans-k 16 \
  --umap
```

## Worked Example: Transcript-Level Workflow On Simulated Data

This end-to-end example uses the same simulator as the [benchmark study](https://github.com/heart-gen/bsvae-benchmarking), so every file below has a known truth to compare against. It runs in minutes on a CPU. The real-data version of the same chain (residualise, `bsvae-sweep-k`, `extract-modules`, `extract-networks`, `export-latents`) is in the [tutorial](https://bsvae.readthedocs.io/en/latest/tutorial.html).

```bash
# 1. simulate one transcript-level scenario (500 genes, 6 modules, 150 samples)
bsvae-simulate init-config --output iso.json --isoform      # edit n_genes, n_modules, grid as needed
bsvae-simulate generate-grid --config iso.json --outdir sim --reps 1 --base-seed 13
SCEN=$(ls -d sim/scenarios/*/rep_000 | head -1)

# 2. choose K, then retrain the selected model (transcript-to-gene map enables the hierarchical loss)
bsvae-sweep-k iso1 --dataset $SCEN/expr/features_x_samples.tsv.gz \
  --k-grid 12,18,24 --sweep-epochs 60 --stability-reps 2 --val-frac 0.1 \
  --tx2gene $SCEN/truth/tx2gene.tsv --hier-strength 0.2 --isoform-stratified

# 3. two outputs: gene-coherent modules and the isoform-specific residual scan
bsvae-networks extract-modules --model-path results/iso1/final_k<K> \
  --dataset $SCEN/expr/features_x_samples.tsv.gz --expr $SCEN/expr/features_x_samples.tsv.gz \
  --tx2gene $SCEN/truth/tx2gene.tsv --covariates $SCEN/covariates.tsv.gz \
  --loo-eigengene approx --max-target-corr 0.4 --soft-eigengenes \
  --output-dir results/iso1/modules
```

`results/iso1/modules/` now holds `modules.csv` (gene-coherent assignments, grey is `-1`), `modules_raw.csv`, `gene_coherence.csv`, `isoform_specific.csv`, `eigengenes.csv`, `kME.csv` and `gamma.npz`. Score the modules against the simulator's truth:

```python
import pandas as pd
from sklearn.metrics import adjusted_rand_score

pred = pd.read_csv("results/iso1/modules/modules.csv", index_col="feature_id")["module"]
truth = pd.read_csv(f"{scen}/truth/modules_hard.csv", index_col="feature_id")["module"]
print(adjusted_rand_score(truth.loc[pred.index], pred))          # grey (-1) counted as its own label

calls = pd.read_csv("results/iso1/modules/isoform_specific.csv")
sw = pd.read_csv(f"{scen}/truth/switch_table.tsv.gz", sep="\t", index_col=0)
print(calls["called"].sum(), "calls;", sw["is_switched"].sum(), "truly switched isoforms")
```

For calibrated calls when modules were learned on the same samples, train one fold model per sample split and cross-fit:

```bash
bsvae-networks isoform-specific --dataset $SCEN/expr/features_x_samples.tsv.gz \
  --tx2gene $SCEN/truth/tx2gene.tsv --covariates $SCEN/covariates.tsv.gz \
  --modules fold0/modules/modules_raw.csv fold1/modules/modules_raw.csv --folds folds.tsv \
  --max-target-corr 0.4 --output results/iso1/isoform_specific_crossfit.tsv
```

Then test candidate genes against the same modules, and pool several cohorts:

```bash
bsvae-networks target-genes --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --modules results/iso1/modules/modules.csv --genes candidates.txt --output-dir results/iso1/targets

bsvae-networks target-genes-combine \
  --inputs DLPFC=results/dlpfc/targets/target_isoforms.tsv HIPP=results/hipp/targets/target_isoforms.tsv \
  --n-samples DLPFC=160 HIPP=150 --output results/targets_combined.tsv
```

## Training Outputs

`bsvae-train` writes to `results/<experiment>/` by default:

- `model.pt`
- `specs.json`
- `train_losses.csv`
- `model-<epoch>.pt` checkpoints when `--checkpoint-every` is greater than zero

`bsvae-sweep-k` writes:

- `results/<name>/sweep_k/sweep_results.csv`
- `results/<name>/sweep_k/sweep_summary.json`
- `results/<name>/sweep_k/k<K>/rep_<rep>/...` per-sweep run outputs
- `results/<name>/final_k<K>/...` for the final retrained model when `--train-final` is enabled

## Simulation and Benchmarking

Generate one synthetic dataset:

```bash
bsvae-simulate generate \
  --output data/sim_expr.csv \
  --save-ground-truth data/sim_truth.csv
```

Generate a scenario grid:

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 30 \
  --base-seed 13
```

Validate the generated grid:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Each scenario replicate writes method-ready files under `results/sim_pub_v1/scenarios/<scenario_id>/rep_<rep>/`, including:

- `expr/features_x_samples.tsv.gz`
- `expr/samples_x_features.tsv.gz`
- `truth/modules_hard.csv` (null features are `-1`)
- `truth/tx2gene.tsv` and `truth/modules_gene.csv` for the transcript-level family (`bsvae-simulate init-config --isoform`)
- `method_inputs.json`

Run BSVAE and WGCNA head-to-head on a grid and summarise grey-aware ARI/NMI, best-match Jaccard, and gene-level recovery:

```bash
bsvae-benchmark check-env                      # torch, Rscript, WGCNA
bsvae-benchmark run --config iso.json --outdir results/bench --methods bsvae,wgcna --seeds 0,1,2
bsvae-benchmark summarize --results results/bench/results.tsv --by scenario_id,method
```

`run` is resumable and can be split across a cluster with `--rep-ids` and `--scenario-ids`. Comparison methods are `bsvae_raw`, `bsvae_hier`, `bsvae_isoform`, `wgcna_raw`, `acorde`, `twn`, `usage_network`, `ica` and `nmf`. In the companion study (10 replicates, transcript-level scenarios, 150 samples), mean all-feature ARI was:

| Method | ARI (all features) | Unassigned |
| --- | --- | --- |
| `bsvae` (gene coherence) | 0.78 | 10% |
| `bsvae_raw` | 0.70 | 10% |
| `wgcna` (gene coherence) | 0.65 | 0.3% |
| `wgcna_raw` | 0.54 | 0.3% |
| `acorde` | 0.44 | 35% |

These are simulation results; see [Benchmarking](https://bsvae.readthedocs.io/en/latest/benchmarking.html) for the pipeline, output files, matched-coverage metrics and caveats, including that the residual-scan defaults were not validated in every simulated cell.

## Documentation

Full documentation is at [bsvae.readthedocs.io](https://bsvae.readthedocs.io/):

- [Installation](https://bsvae.readthedocs.io/en/latest/installation.html) and [Quick Start](https://bsvae.readthedocs.io/en/latest/quickstart.html)
- [Tutorial](https://bsvae.readthedocs.io/en/latest/tutorial.html) (real-data workflow and the simulated example)
- [CLI reference](https://bsvae.readthedocs.io/en/latest/cli.html), [network workflows](https://bsvae.readthedocs.io/en/latest/networks.html) and [hyperparameters](https://bsvae.readthedocs.io/en/latest/hyperparameters.html)
- [Benchmarking](https://bsvae.readthedocs.io/en/latest/benchmarking.html) and [HPC](https://bsvae.readthedocs.io/en/latest/hpc.html)

The same pages are in `docs/` in this repository. Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
