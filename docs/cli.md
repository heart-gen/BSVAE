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
- `--diag-covariance` (one variance per latent dimension per component instead of isotropic)
- `--beta` default `1.0` (a true per-element ELBO weight; see `--no-kl-per-element`)
- `--no-kl-per-element` legacy KL scaling (sum over latent dims), whose effective β grows with the number of samples
- `--free-bits` default `0.0`
- `--kl-warmup-epochs` default `0` (ramp counted from the start of the GMM phase)
- `--kl-anneal-mode` `linear` or `cyclical`
- `--kl-cycle-length` default `50`
- `--sep-strength` default `0.1`
- `--sep-alpha` default `2.0`
- `--bal-strength` default `0.1`
- `--bal-type` default `entropy` (`uniform` pushes equal module sizes, `none` disables)
- `--bal-scale-gmm`
- `--bal-ema-blend` default `0.5`
- `--pi-entropy-strength` default `0.0`
- `--hier-strength` default `0.0`
- `--hier-margin` default `1.0` (isoform pairs closer than this in latent space are left alone)
- `--no-hier-corr-weight` disable weighting isoform pairs by their positive expression correlation
- `--corr-strength` default `0.0`
- `--latent-corr-strength` default `0.0`
- `--contrastive-strength` default `0.0` (size-invariant neighbour-contrastive latent alignment)
- `--contrastive-k` default `5`, `--contrastive-tau` default `0.1`, `--contrastive-min-corr` default `0.3`
- `--masked-recon`
- `--tx2gene` (header optional; Ensembl version suffixes are stripped on both sides)
- `--keep-tx-version`
- `--isoform-stratified` (gene-grouped batches so all isoforms of a gene co-occur)
- `--p-multi` default `0.5`
- `--collapse-min-usage` default `0.002` (absolute usage floor for dead-component revival)
- `--collapse-threshold` deprecated and ignored
- `--collapse-noise-scale` default `0.5`
- `--num-workers` default `0`
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

All model, loss and schedule flags are shared with `bsvae-train` (same parser groups, same defaults), so the selected `K` is selected with the model you will train.

Selection behavior:

- `--stability-reps 1`: select the best `K` by held-out ELBO (reconstruction + β·KL, without the regularisers, which are not comparable across `K`)
- `--stability-reps > 1`: select the best `K` by mean pairwise ARI across held-out-feature assignments; ties are broken by held-out ELBO
- with `--tx2gene`, the held-out split is made by gene so sibling isoforms never straddle the split

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

By default the GMM assignments are post-processed the way WGCNA post-processes a tree cut: modules smaller than `--min-module-size` are dissolved, modules whose eigengenes correlate above `1 - --merge-cut-height` are merged, and features whose summed responsibility for their merged module is below `--min-gamma` become unassigned ("grey", label `-1`). The cutoff is applied after merging so that a member whose responsibility is spread over several components of one over-clustered module is judged by their total (merging and kME need `--expr`).

Because merging consolidates split modules, train with a generous `--n-modules` (about three times the number of modules you expect, at least 20): K-means initialisation gives centroids to the largest modules first, so small modules only get a component of their own when K is larger than the module count. The benchmark harness does this automatically (`--over-cluster-factor`, default 3).

Options:

- `--expr` expression matrix for eigengenes, merging and kME
- `--min-gamma` default `0.5`
- `--min-module-size` default `10`
- `--merge-cut-height` default `0.4`
- `--no-merge`
- `--kme-reassign`
- `--min-kme` default `0.3`
- `--kme-margin` default `0.1`
- `--no-postprocess` write raw `argmax(γ)` only
- `--soft-eigengenes`
- `--use-leiden` (sparse graph; safe at transcript scale)
- `--leiden-resolution` default `1.0`
- `--tx2gene`
- `--aggregate-to-gene`
- `--batch-size` default `128`
- `--no-cuda`

Outputs:

- `gamma.npz`, `hard_assignments.npz` (raw)
- `modules.csv` (`feature_id, module, module_label`; grey is `-1` / `grey`)
- `eigengenes.csv`, `kME.csv`, `merge_log.json`, `gamma_merged.npz`
- `soft_eigengenes.csv` when requested
- `leiden_modules.csv` when requested
- `gamma_gene.npz`, `hard_assignments_gene.npz` and `gene_divergent.csv` when gene aggregation is requested

### `consensus-modules`

Majority-vote consensus over several `extract-modules` runs (for example different seeds). Runs are aligned by module overlap, each feature takes the label most runs agree on, and the agreement fraction is reported as a stability score.

```bash
bsvae-networks consensus-modules \
  --modules results/seed0/modules.csv results/seed1/modules.csv results/seed2/modules.csv \
  --output-dir results/consensus
```

Options:

- `--min-agreement` default `0.5` (features below this agreement are unassigned)
- `--min-module-size` default `10`

Outputs: `consensus_modules.csv` (`feature_id, module, module_label, agreement`) and `consensus_summary.json` (per-run ARI to the consensus).

### `isoform-specific`

Call isoforms whose expression belongs to a module other than their gene's. Each multi-isoform transcript's profile is regressed on its gene aggregate (isoforms summed on the linear scale); the residual is correlated with every module eigengene from `extract-modules`, and a transcript is called when its residual kME to a module other than its gene's majority module reaches `--min-kme`.

```bash
bsvae-networks isoform-specific \
  --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --modules results/run/modules/modules.csv --output results/run/isoform_specific.csv
```

Options:

- `--min-kme` default `0.4` (0.4–0.5 gives 6–9× enrichment of truly switched isoforms over the base rate on the isoform simulations; lower values trade precision for recall)
- `--keep-tx-version`

Output columns: `gene_id, gene_module, best_other_module, kme_residual, r2_gene, n_isoforms, called`.

### `isoform-residuals`

Write the gene-programme-removed residual matrix itself (what `isoform-specific` correlates with the eigengenes), for custom downstream analysis. Clustering the residuals alone does **not** recover isoform-specific modules: switched isoforms are too few per module to form clusters of their own, so the membership test above is the recommended route.

```bash
bsvae-networks isoform-residuals \
  --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --output results/residuals.tsv.gz --stats results/residual_stats.tsv
```

Options:

- `--min-isoforms` default `2`
- `--keep-tx-version`

### `module-quality`

Per-module quality (size, mean within-module correlation, eigengene variance explained, kME distribution), optional Fisher enrichment of GMT annotation sets with Benjamini–Hochberg q-values, and optional preservation of the modules in a second dataset (permutation Z-score of within-module density, `Z_density`, with WGCNA's conventions: > 2 weak, > 10 strong; plus `cor_kME`, the correlation of members' kME between datasets).

```bash
bsvae-networks module-quality \
  --modules results/run/modules/modules.csv --expr data/expression.csv \
  --test-expr data/replication.csv --annotations go.gmt --output-dir results/run/quality
```

Options:

- `--test-expr`, `--n-perm` default `200`
- `--annotations` (GMT), `--min-overlap` default `3`

Outputs: `module_quality.csv`, `module_preservation.csv`, `module_enrichment.csv`.

### `regress-covariates`

Remove known sample covariates (batch, sex, age, cell-type proportions, ...) from every feature profile by least squares before training, the way `limma::removeBatchEffect` does; numeric covariates are centred, categorical ones one-hot encoded.

```bash
bsvae-networks regress-covariates --dataset data/expression.csv --covariates data/samples.tsv --columns batch,sex --output data/expression_corrected.tsv.gz
```

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

Assignments are post-processed first (same flags as `extract-modules`; `--no-postprocess` scores raw `argmax(γ)`). The JSON holds the grey-aware metrics: `ari_all`/`nmi_all` (grey as its own label), `ari_assigned`/`nmi_assigned` (grey dropped on either side), grey precision/recall, best-match Jaccard/F1, module counts and `n_features`.

### Scenario-grid commands

```bash
bsvae-simulate init-config --output sim.yaml            # gene-level nb_latent_modules
bsvae-simulate init-config --output iso.yaml --isoform  # transcript-level nb_isoform_modules
bsvae-simulate generate-grid --config sim.yaml --outdir results/sim_pub_v1 --reps 30 --base-seed 13
bsvae-simulate generate-scenario --config sim.yaml --scenario-id S001__... --rep 0 --outdir results/sim_pub_v1
bsvae-simulate validate-grid --grid-dir results/sim_pub_v1
```

Per-replicate outputs are written under:

`<outdir>/scenarios/<scenario_id>/rep_<rep>/`

Common files:

- `expr/features_x_samples.tsv.gz`
- `expr/samples_x_features.tsv.gz` (when `outputs.emit_wgcna_matrix` is true)
- `truth/modules_hard.csv` (`module` is `-1` for null features), `truth/modules_overlap.csv`
- `truth/tx2gene.tsv`, `truth/modules_gene.csv`, `truth/switch_table.tsv.gz` for the isoform family
- `method_inputs.json`

Configs are JSON by default; YAML is accepted when PyYAML is installed (`pip install bsvae[yaml]`).

Counts are library-size normalised with DESeq-style median-of-ratios size factors (`generator.normalization: "median_ratio"`, the default) before the log2 transform. Plain CPM (`"cpm"`) is available but makes every feature without module signal co-vary against the large modules, so "null" features are no longer null and no method can leave them unassigned.

## `bsvae-benchmark`

Head-to-head benchmark of BSVAE and WGCNA on a scenario grid. WGCNA runs through `Rscript` with the packaged `wgcna.R` (R with the `WGCNA` package must be installed; otherwise its rows are recorded as skipped).

```bash
bsvae-benchmark check-env
bsvae-benchmark run --config iso.yaml --outdir results/bench --methods bsvae,wgcna --seeds 0,1,2 --reps 3
bsvae-benchmark summarize --results results/bench/results.tsv
```

BSVAE is trained with `--over-cluster-factor` (default 3) times the true module count and merged back by post-processing; pass `--over-cluster-factor 1` to train with the true K. Methods: `bsvae`, `bsvae_consensus` (majority vote over `--consensus-runs` seeds, default 3), `bsvae_isoform` (isoform-specific membership calls, scored by switched-isoform recall/precision/ARI), `bsvae_regressed` / `wgcna_regressed` (batch regressed out of the simulated matrix first), `wgcna` (signed network, the WGCNA authors' recommendation), `wgcna_unsigned`, `wgcna_signed`.

`run` writes one row per (scenario, replicate, method, seed) to `results.tsv` as soon as it finishes (resumable with the same command), per-run outputs under `scenarios/<id>/rep_<r>/methods/<method>/seed_<s>/`, and `summary.tsv` (mean ± sd per scenario and method). BSVAE options (`--epochs`, `--latent-dim`, `--hier-strength`, `--isoform-stratified`, ...), post-processing options (`--min-gamma`, `--min-module-size`, `--merge-cut-height`, `--no-postprocess`) and WGCNA options (`--wgcna-network-type`, `--wgcna-soft-power`, `--wgcna-min-module-size`, `--wgcna-merge-cut-height`, `--wgcna-threads`) can be overridden. `scripts/benchmark_sims.py` drives the same harness over seven laptop-sized reference scenarios.
