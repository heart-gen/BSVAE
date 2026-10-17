# Quick Start

For a longer guided workflow, see the [Tutorial](tutorial.md). For flag-by-flag reference, see the [CLI Reference](cli.md).

## 1. Prepare Data

Input must be an expression matrix in `features x samples` orientation.

- rows are features
- columns are samples
- the first column is the feature ID index for CSV or TSV files

## 2. Train A Small Model

```bash
bsvae-train pilot_run \
  --dataset data/expression.csv \
  --epochs 50 \
  --n-modules 12 \
  --latent-dim 16
```

Expected outputs in `results/pilot_run/`:

- `model.pt`
- `specs.json`
- `train_losses.csv`

## 3. Recommended: Tune The Number Of Modules

`--n-modules` sets the number of Gaussian-mixture components, not the final module count: post-processing in `extract-modules` merges components whose eigengenes correlate, so use about three times the number of modules you expect (at least 20). `bsvae-sweep-k` can pick the component count by held-out ELBO before a production run.

```bash
bsvae-sweep-k sweep_pilot \
  --dataset data/expression.csv \
  --k-grid 6,8,12,16 \
  --sweep-epochs 30 \
  --stability-reps 3 \
  --val-frac 0.1
```

This writes sweep artifacts under `results/sweep_pilot/sweep_k/` and retrains the selected model under `results/sweep_pilot/final_k<K>/`.

## 4. Extract Networks

```bash
bsvae-networks extract-networks \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/networks \
  --methods mu_cosine
```

## 5. Extract Modules

`extract-modules` always needs the training dataset, and it needs `--expr` when you want eigengenes.

```bash
bsvae-networks extract-modules \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/modules \
  --expr data/expression.csv \
  --soft-eigengenes
```

Expected outputs:

- `gamma.npz`
- `hard_assignments.npz` (raw `argmax(γ)`)
- `modules.csv`, `eigengenes.csv`, `kME.csv`, `merge_log.json` (post-processed: grey class, minimum size, eigengene merging)
- `soft_eigengenes.csv` when requested

### Transcript-level data

With transcript-level data, pass `--tx2gene` as well. `extract-modules` then produces two independent outputs: module assignments annotated by the gene-coherence policy (default; `--no-gene-coherence` to disable) and the isoform-specific residual scan (`--no-isoform-calls` to skip, `--loo-eigengene` to change the leave-one-gene-out eigengene correction). A transcript's module label never decides whether it is tested.

```bash
bsvae-networks extract-modules \
  --model-path results/iso1 \
  --dataset data/transcripts.tsv.gz \
  --output-dir results/iso1/modules \
  --expr data/transcripts.tsv.gz \
  --tx2gene data/tx2gene.tsv
```

Additional outputs:

- `modules.csv`: module assignments after the gene-coherence policy (an isoform placed apart from its gene's majority module is returned to it unless its sibling-residual kME supports the other membership at BH FDR 0.05)
- `modules_raw.csv`: the labels before the policy
- `gene_coherence.csv`: every divergent isoform tested, with its residual kME, q-value and whether it was kept
- `isoform_specific.csv`: the residual scan over every multi-isoform transcript on the raw labels (`called`, `best_other_module`, `kme_residual`, `q_value`, `label_discordant`), the primary evidence for isoform-specific divergence

## 6. Export Latents

```bash
bsvae-networks export-latents \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output results/sweep_pilot/final_k12/latents
```

This writes `latents.npz` with `mu`, `logvar`, `gamma`, and `feature_ids`.

## 7. Run Latent Analysis

```bash
bsvae-networks latent-analysis \
  --model-path results/sweep_pilot/final_k12 \
  --dataset data/expression.csv \
  --output-dir results/sweep_pilot/final_k12/latent_analysis \
  --kmeans-k 12 \
  --umap
```

## 8. Build A Simulation Grid

```bash
bsvae-simulate init-config --output sim.yaml

bsvae-simulate generate-grid \
  --config sim.yaml \
  --outdir results/sim_pub_v1 \
  --reps 5 \
  --base-seed 13
```

Validate outputs:

```bash
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

## 9. Benchmark Methods On The Simulated Grid

```bash
bsvae-benchmark check-env
bsvae-benchmark run --config sim.yaml --outdir results/bench --methods bsvae,wgcna --reps 3
bsvae-benchmark summarize --results results/bench/results.tsv
```

See [Benchmarking](benchmarking.md) for the full pipeline, the output files and the headline results, and the [Tutorial](tutorial.md) for an end-to-end transcript-level example.
