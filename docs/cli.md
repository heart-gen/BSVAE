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
- `--amp` mixed-precision (bfloat16) forward pass on CUDA
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
- `--gene-coherence` (requires `--expr` and `--tx2gene`): an isoform assigned to a module other than its gene's majority module is moved back to the gene's module unless its sibling-residual kME to that other module reaches `--min-kme` at BH FDR 0.05. This removes divergence driven by abundance noise alone; on the abundance-confounding simulations it is the difference between BSVAE's false-split rate and WGCNA's (see `scripts/benchmark_confounding.py`). Writes `gene_coherence.csv`.
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

Call isoforms whose expression belongs to a module other than their gene's. Each multi-isoform transcript's profile is regressed on the sum of its *sibling* isoforms (linear scale, leave-one-out, so an isoform carrying a large share of its gene does not regress out its own signal); the residual is correlated with every module eigengene from `extract-modules`, after each eigengene has been residualised on the gene's own module eigengene (module eigengenes are correlated, and without this step gene-programme signal that the regression did not remove leaks into neighbouring modules and doubles the false-call rate among unswitched isoforms). A transcript is called when its residual kME to a module other than its gene's majority module reaches `--min-kme` and the correlation passes a Benjamini–Hochberg test at `--alpha`.

```bash
bsvae-networks isoform-specific \
  --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --modules results/run/modules/modules.csv --output results/run/isoform_specific.csv
```

Options:

- `--min-kme` default `0.3` (effect-size floor; with the residualised eigengenes this keeps precision at or above the previous `0.4` default while catching about 10 % more switched isoforms on the isoform simulations)
- `--alpha` default `0.05` (BH false-discovery rate for the residual correlation, Bonferroni-corrected over modules per transcript; `0` disables it). At 150 samples the kME floor dominates; with several hundred samples the test is what keeps weak-but-significant correlations from flooding the calls.
- `--no-residualize-eigengenes`, `--no-leave-one-out` restore the previous behaviour
- `--keep-tx-version`

Output columns: `gene_id, gene_module, best_other_module, kme_residual, p_value, q_value, r2_gene, n_isoforms, mean_expr, called`.

Rare transcripts: a low-abundance isoform has an attenuated correlation with every eigengene, so its achievable kME is bounded by its sampling noise rather than by biology. On simulations where minor isoforms carry about a tenth of their gene, the lowest-expressed third of switched isoforms is called about a third of the time at 150 samples and the true target module is ranked first about 70 % of the time even with the true module factors, so more samples (or deeper quantification) is the lever, not a lower threshold. Removing zeros from the reconstruction loss (`--masked-recon`), filtering low-expression features before training, and training on negative-binomial Pearson residuals of the raw counts (`bsvae.utils.preprocess.pearson_residuals`) were all evaluated: the first two reduce recovery of switched isoforms, and Pearson residuals leave rare-isoform recovery unchanged (caller recall +0.04) while lowering transcript-level ARI by 0.05 on the isoform scenarios (unchanged on gene-level data), so log-normalised input remains the default. Simulated scenarios now also write `expr/counts_features_x_samples.tsv.gz` for count-aware methods.

### `target-genes`

Targeted-gene mode: test the isoforms of candidate genes or GWAS loci for module membership against a transcriptome-wide module solution, without re-training. Only the target isoforms are tested, so multiplicity is controlled within the target set and rare isoforms whose correlations are too weak for a transcriptome-wide call become reportable. The modules, eigengenes and neighbours all come from the full run; a target gene is never clustered on its own.

```bash
# candidate genes (Ensembl IDs, or symbols with --annotation)
bsvae-networks target-genes --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --modules results/run/modules/modules.csv --genes candidates.txt --output-dir results/run/targets

# GWAS loci: index SNPs or credible-set intervals in BED, genes within 500 kb
bsvae-networks target-genes --dataset data/transcripts.tsv.gz --tx2gene data/tx2gene.tsv \
  --modules results/run/modules/modules.csv --bed scz_loci.bed --annotation gencode.v44.annotation.gtf.gz \
  --rule window --window 500000 --counts data/transcript_counts.tsv.gz \
  --covariates data/samples.tsv --condition dx --output-dir results/run/scz_loci
```

Targets (combine freely):

- `--genes` one Ensembl gene ID or symbol per line (symbols need `--annotation`)
- `--transcripts` one transcript ID per line
- `--bed` loci (`chrom start end [name]`, 0-based) with `--annotation` (GTF/GFF `gene` records, or a BED-like `chrom start end gene_id [gene_name] [strand]` table); genes are chosen per locus by `--rule overlap` (gene body overlaps the locus), `window` (default; locus extended by `--window` bp on each side, so a one-base BED of index SNPs means "within 500 kb of the SNP") or `closest` (`--n-closest` genes by TSS distance). The mapping is written to `loci_genes.tsv` with distances.

Test, per target isoform:

1. the residual after regressing the isoform on its sibling isoforms (single-isoform genes use their own centred profile);
2. kME to every module eigengene, each residualised on the gene's own module eigengene;
3. the best module other than the gene's own, against a null: `--null permutation` (default; the residual is permuted over samples `--n-perm` times and the maximum |kME| over the other modules forms the null, which handles best-module selection and sparse, non-Gaussian profiles), `matched` (observed best-other |kME| of the `--n-matched` non-target isoforms that sit in their gene's own module, closest in mean expression and zero fraction, which also absorbs residual leakage) or `analytic` (t-test);
4. BH within the target set (`--alpha`, default 0.05) plus the effect-size floor `--min-kme` (default 0.2, lower than the transcriptome-wide caller's 0.3 because the calibrated test should decide once samples are plentiful: on simulations with rare switched isoforms and 500 samples, the floor at 0.2 raises recall of switched isoforms from 0.58 to 0.71 for hard switches and 0.33 to 0.57 for partial switches, with under 10 % false calls among unswitched isoforms; at 150 samples the test itself is the binding constraint and results match the transcriptome-wide caller);
5. with `--counts`, a reliability estimate from the negative-binomial sampling noise of each profile and a disattenuated kME (`kme_residual / sqrt(reliability)`), so rare isoforms are ranked by the correlation they would have without sampling noise;
6. with `--covariates` and a two-level `--condition`, the best-other kME per group and a Fisher-z interaction test (`p_interaction`): does membership differ between cases and controls.

Outputs in `--output-dir`:

- `target_isoforms.tsv`: `gene_id, n_isoforms, single_isoform, mean_expr, zero_fraction, gene_module, isoform_module, kme_gene_module, best_other_module, kme_residual, p_value, q_value, r2_gene, called` plus `reliability, kme_disattenuated` and the condition columns when requested
- `target_genes.tsv`: per gene, the modules its isoforms occupy, `divergent` (isoforms in two or more modules, or any isoform called), best isoform and minimum q
- `target_neighbours.tsv`: for each called isoform, the `--n-neighbours` members of its target module most correlated with the residual, with their kME (hub status)
- `target_loci.tsv` (with `--bed`): per locus, genes and isoforms tested, isoforms called, modules hit and a binomial p-value for concentration in the most frequent module

What it cannot do: it needs a transcriptome-wide `extract-modules` run first, it cannot rescue transcripts that quantification dropped, and it does not move the noise ceiling of a rare isoform (see `isoform-specific`); it lowers the multiplicity burden and reports calibrated evidence for exactly the genes you care about.

### `target-genes-combine`

Combine `target-genes` results from several cohorts, tissues or brain regions into one evidence table per isoform. Each cohort's test asks whether the isoform's residual tracks a module other than its gene's against that cohort's own module solution, so the p-values combine with Stouffer's method (weights √n with `--n-samples`, otherwise equal) without aligning modules across cohorts; module identity is reported per cohort. A combined call needs BH FDR < `--alpha` (0.05), mean |kME| ≥ `--min-kme` (0.2) and at least `--min-cohorts` (2) cohorts. This is the lever for 150-sample regions: a rare isoform at correlation 0.2 is undetectable in one GTEx brain region and clear across ten.

```bash
bsvae-networks target-genes-combine \
  --inputs DLPFC=results/dlpfc/targets/target_isoforms.tsv HIPP=results/hipp/targets/target_isoforms.tsv \
  --n-samples DLPFC=160 HIPP=150 --output results/brain_targets_combined.tsv
```

Output columns: `gene_id, n_cohorts, n_called, mean_abs_kme, max_abs_kme, z_combined, p_combined, modules, q_combined, called`.

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
- `truth/tx2gene.tsv`, `truth/modules_gene.csv`, `truth/switch_table.tsv.gz` for the isoform family (the switch table's `mechanism` column is `switch` for an uncompensated abundance switch, `usage` for a compensated usage shift that leaves the gene total unchanged — generator keys `usage_rate`, `usage_strength`, `usage_from`; `sample_seed` draws a second cohort of the same truth with new samples, and `tissue_shift` moves that fraction of signal genes to another module in the second cohort, flagged in `modules_gene.csv` as `tissue_shifted`)
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

BSVAE is trained with `--over-cluster-factor` (default 3) times the true module count and merged back by post-processing; pass `--over-cluster-factor 1` to train with the true K. Methods: `bsvae`, `bsvae_consensus` (majority vote over `--consensus-runs` seeds, default 3), `bsvae_isoform` (isoform-specific membership calls, scored by switched-isoform recall/precision/ARI), `bsvae_regressed` / `wgcna_regressed` (batch regressed out of the simulated matrix first), `wgcna` (signed network, the WGCNA authors' recommendation), `wgcna_unsigned`, `wgcna_signed`, and the comparison methods below.

Comparison methods (`bsvae.benchmark.methods`):

- `usage_network`: transcript-usage network — logit isoform fractions within each gene (single-isoform genes grey) clustered with WGCNA, so modules follow co-varying usage rather than abundance.
- `ica` / `nmf`: FastICA and NMF on the transcript matrix with the same component count as BSVAE; a feature joins the component with its largest standardised loading (|z| ≥ 2) or the basis column carrying at least half of its reconstruction, otherwise grey; components below the minimum module size are dissolved.
- `acorde`: the acorde isoform co-expression pipeline (Arzalluz-Luque et al. 2022) re-implemented in R with `dynamicTreeCut`: hierarchical clustering on 1 − |cor|, dynamic tree cut (deepSplit 3, minimum cluster 20), metatranscript merging at height 0.2. Its within-cluster correlation filter and forced expansion were tuned for percentile correlations on single-cell data and empty every cluster on bulk Pearson correlations, so they are off by default (`run_acorde(lowcor_threshold=3, expand=True)` restores them). Needs `Rscript` with `dynamicTreeCut` (installed with WGCNA).
- `twn`: transcriptome-wide network (Saha et al. 2017) — a sparse Gaussian graphical model over gene total expression and isoform ratios with the paper's penalty structure (λ_tt 0.5, λ_ti 0.4, λ_ii 0.4, within-gene λ_s 0.05), solved with the R `glasso` package instead of the original MATLAB QUIC; within-gene edges are removed as in the original. It yields edges, not modules, so it is scored with edge metrics: `edge_tt_precision` (gene–gene edges within one true module), `edge_tt_recall` (against the simulator's top within-module pairs), `edge_ti_switched_precision/recall` (isoform-ratio edges of switched or usage-regulated isoforms to genes of their target module) and `edge_ti_unswitched_false_rate` (unswitched isoforms with a cross-module gene edge). Module methods get the same gene-level edge metrics from their labels (`edges_from_labels`: top-10 correlated gene pairs within each module) so TWN can be compared. Needs `Rscript` with `glasso`.

`run` writes one row per (scenario, replicate, method, seed) to `results.tsv` as soon as it finishes (resumable with the same command), per-run outputs under `scenarios/<id>/rep_<r>/methods/<method>/seed_<s>/`, and `summary.tsv` (mean ± sd per scenario and method). BSVAE options (`--epochs`, `--latent-dim`, `--hier-strength`, `--isoform-stratified`, ...), post-processing options (`--min-gamma`, `--min-module-size`, `--merge-cut-height`, `--no-postprocess`) and WGCNA options (`--wgcna-network-type`, `--wgcna-soft-power`, `--wgcna-min-module-size`, `--wgcna-merge-cut-height`, `--wgcna-threads`) can be overridden. `scripts/benchmark_sims.py` drives the same harness over seven laptop-sized reference scenarios; `scripts/benchmark_confounding.py` runs the abundance-confounding sets (abundance-only, usage-only, usage-to-gene-signal ratio, switch versus usage) and stratifies false divergence by isoform abundance; `scripts/benchmark_preservation.py` measures reproducibility across random initialisations, donor bootstraps, an independent cohort and a partially changed tissue (ARI, module preservation Z_density / cor_kME, and reproducibility of isoform-specific divergence for genes with and without a true switch).
