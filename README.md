# BSVAE: Gaussian Mixture VAE for Module Discovery

[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-orange)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue)](https://python.org/)
[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Documentation](https://readthedocs.org/projects/bsvae/badge/?version=latest)](https://bsvae.readthedocs.io/)

BSVAE is a PyTorch package for feature-level module discovery in omics data. The main model, `GMMModuleVAE`, learns latent structure across features such as genes, transcripts, or proteins, then supports downstream network extraction, module assignment, latent analysis, and simulation-based benchmarking.

## What The Package Provides

- `bsvae-train` for training a two-phase GMM-VAE
- `bsvae-sweep-k` for selecting the number of modules (`K`) with held-out validation and optional stability replicates
- `bsvae-networks` for post-training network extraction, module extraction, latent export, and latent analysis
- `bsvae-simulate` for synthetic data generation, scenario-grid simulation, and module-recovery benchmarking

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

Optional dependency:

- `anndata` is only needed for `.h5ad` input or `.h5ad` latent export helpers

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

## Quickstart

Train a small model:

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

Recommended production flow: select `K` first, then use the retrained final model.

```bash
bsvae-sweep-k sweep1 \
  --dataset data/expression.csv \
  --k-grid 8,12,16,24,32 \
  --sweep-epochs 60 \
  --stability-reps 5 \
  --val-frac 0.1
```

This writes sweep outputs under `results/sweep1/sweep_k/` and retrains the selected model under `results/sweep1/final_k<K>/`.

Extract module assignments from the final model (post-processing adds a grey class, a minimum module size and eigengene merging, so train with a generous `--n-modules`, about three times the expected module count):

```bash
bsvae-networks extract-modules \
  --model-path results/sweep1/final_k16 \
  --dataset data/expression.csv \
  --output-dir results/sweep1/final_k16/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

With transcript-level data, pass `--tx2gene` as well. `extract-modules` then produces two independent outputs: module assignments annotated by the gene-coherence policy (default; `--no-gene-coherence` to disable) and the isoform-specific residual scan (`--no-isoform-calls` to skip, `--loo-eigengene` to change the leave-one-gene-out eigengene correction). A transcript's module label never decides whether it is tested.

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
bsvae-benchmark run --config iso.yaml --outdir results/bench --methods bsvae,wgcna --seeds 0,1,2
bsvae-benchmark summarize --results results/bench/results.tsv
```

## Documentation

See the docs for the full workflow and command reference:

- `docs/index.md`
- `docs/quickstart.md`
- `docs/tutorial.md`
- `docs/cli.md`
- `docs/networks.md`
- `docs/hyperparameters.md`

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
