## Changelog

### Unreleased — transcript-level review

Correctness fixes

* KL term is scaled per observed value (`kl_per_element`, default on) so `--beta 1` is a true ELBO weight; previously the effective β grew with the number of samples and collapsed γ. Use `--no-kl-per-element` for the old behaviour.
* Phase 1 (warm-up) now reconstructs the same z-scored target as Phase 2 when `--normalize-input` is set.
* `--kl-warmup-epochs` ramps inside the GMM phase, and evaluation always uses full β.
* Dead-component revival splits the most-used component (π halved, Adam state reset) instead of assigning a near-zero mixing weight; it never runs after the final epoch or during the transition ramp; the floor is absolute (`--collapse-min-usage`, default 0.002) rather than `0.5/K`, so small real modules survive.
* GMM σ floor is a softplus parameterisation (no vanishing gradients at the floor); separation loss is averaged over pairs; encoder log-variance is clamped; `--masked-recon` keeps negative values.
* `tx2gene` loading auto-detects headers, strips Ensembl version suffixes and GENCODE pipe fields, warns on low coverage, and no longer infers genes from transcript numbers.
* `--isoform-stratified` uses a gene-grouped batch sampler so the hierarchical loss sees complete isoform groups; the loss itself is a hinge weighted by the positive expression correlation of each pair.
* `bsvae-sweep-k` shares its parser and builders with `bsvae-train`, selects on held-out ELBO, splits by gene when `--tx2gene` is given, and defaults `--free-bits` to 0.
* Latent export/analysis apply input normalisation; `latent-analysis --covariates` correlates covariates with module eigengenes; `--use-leiden` no longer densifies the adjacency; `.tsv.gz` separators are inferred correctly.
* Removed dead legacy modules (`main.py`, `utils/evaluate.py`, `networks/cli.py`, `utils/initialization.py`, `hyperparam.ini`).

New

* WGCNA-style post-processing of GMM assignments (`bsvae.networks.postprocess`): grey class, minimum module size, eigengene-correlation merging, kME; on by default in `extract-modules` and `bsvae-simulate benchmark`.
* Grey-aware metrics (ARI/NMI on all or assigned-only features, grey precision/recall, best-match Jaccard, gene-level majority-vote ARI, split-detection rate).
* Transcript-level simulation family `nb_isoform_modules` (variable isoforms per gene, switching with mixing weight, null genes) and simulator fixes (paired RNG streams, gene-specific batch confounding and nonlinearity, null features, overlap truth, YAML configs).
* `bsvae-benchmark`: resumable BSVAE-vs-WGCNA harness with a packaged R runner; BSVAE is over-clustered (3x the expected module count) and merged back by post-processing, which recovers small modules that a K equal to the true count absorbs.

Simulation normalisation

* Simulated counts are normalised with median-of-ratios size factors instead of plain CPM. Under CPM, features without module signal co-varied with each other (|r| ≈ 0.35) against the large modules, so the benchmark's "null" features were not null and grey-detection metrics measured a compositional artefact.
* Post-processing applies `--min-gamma` to the merged responsibilities (after eigengene merging). A training-time or inference-time "background" mixture component was evaluated and rejected: trained jointly it absorbs every feature; applied at inference it never fires because null features are not latent-space outliers.

Calibration

* `isoform_specific_calls(loo_eigengene=True)` recomputes a module's eigengene without the tested gene's transcripts before scoring them (leave-one-gene-out), removing the selection bias of a transcript contributing to the eigengene it is tested against. `scripts/benchmark_calibration.py` checks the residual p-values under abundance-only null simulations (type-I error at 0.05, KS against uniform) with and without the correction, and the held-out replication of transcriptome-wide calls (labels from one half of the samples, calls validated on the other half). `summarize` reports percentile-bootstrap 95 % intervals over runs.
* Simulated cohorts can differ in their switches (`switch_keep_frac`, `switch_extra_rate`) and share donors (`donor_seed`, `donor_effect_sd`: donor-level module-factor offsets common to every region). `target-genes-combine` applies Strube's correction for dependent cohorts by default (`--correlation empirical`, the between-cohort correlation of z-scores estimated on null-like isoforms) and accepts `--min-called`.

Cross-cohort targeted tests

* `bsvae-networks target-genes-combine` pools `target-genes` tables across cohorts, tissues or regions with Stouffer's method (√n weights), reporting per-cohort modules and a combined FDR, so rare isoforms that no single 150-sample region can call become reportable.

Preservation and reproducibility

* `nb_isoform_modules` gains `sample_seed` (an independent cohort of the same truth) and `tissue_shift` with `tissue_shift_mode` `modules` (that fraction of modules dissolves in the cohort, their genes scattering over the others) or `genes` (that fraction of genes moves at random); `modules_gene.csv` flags moved genes and the metadata lists dissolved modules. On the reference isoform scenario BSVAE's solutions agree across independent cohorts at ARI 0.83 (WGCNA 0.51), across donor bootstraps at 0.80 (0.68) and across random initialisations at 0.89; divergent genes with a true switch reproduce across cohorts 95 % of the time versus 77 % for unswitched divergent genes, which shows that abundance-driven false divergence is systematic rather than noise (gene coherence lowers it to 48 % and removes most such genes). WGCNA's within-module kME ordering is better preserved across cohorts (cor_kME 0.95 vs 0.57), a consequence of BSVAE's smaller, over-clustered modules. In a tissue where 30 % of modules dissolve, both methods' preservation statistics separate intact modules (Z_density 26 for BSVAE, 32 for WGCNA) from dissolved ones (Z 0.5–4), and BSVAE keeps ARI 0.84 on the unchanged genes versus 0.52 for WGCNA. `scripts/benchmark_preservation.py` evaluates random initialisations, donor bootstraps, cross-cohort and cross-tissue agreement (ARI, `module_preservation` Z_density / cor_kME against the fraction of shifted members) and the reproducibility of isoform-specific divergence for switched versus unswitched genes.

Comparison methods

* Gene coherence is the default module-annotation policy whenever transcript-to-gene information is available: `extract-modules` applies it when `--expr` and `--tx2gene` are given (`--no-gene-coherence` to disable), always writes the raw labels to `modules_raw.csv`, and runs the isoform-specific residual scan over every multi-isoform transcript on the raw labels (`isoform_specific.csv`, `--no-isoform-calls` to skip), so module assignment and isoform-specific divergence are two independent outputs. The caller and targeted mode annotate `isoform_module` and `label_discordant` (raw label differs from the gene's majority) instead of filtering on them. In the harness `bsvae` and `wgcna` now apply the policy, `bsvae_raw` / `wgcna_raw` do not, and `wgcna_coherent` gives WGCNA the same gene-aware correction so the value of the VAE is identifiable.
* Comparison-method results on the reference scenarios (3 seeds; assigned-only ARI rewards conservative methods, so the grey fraction and all-feature ARI are reported with it): on the long-tail gene scenario acorde reaches assigned ARI 0.87 with 20 % grey (all-feature ARI 0.62) against BSVAE 0.79 with 2 % grey (0.76) and WGCNA 0.75 (0.63); NMF assigns only an easy core (assigned ARI 1.0, 65 % grey, all-feature 0.13) and ICA 0.72 with 59 % grey. On the isoform scenarios BSVAE leads on all-feature ARI (0.71 / 0.70 vs acorde 0.52 / 0.53, WGCNA 0.53 / 0.53) and split detection (0.90 / 0.43 vs acorde 0.49 / 0.33); the usage network, by construction, does not recover abundance modules (ARI 0.07) but separates isoforms in 61–75 % of switched genes at a 70 % false-split rate; TWN reaches gene–gene edge precision 0.999 at recall 0.13 and links 47 % (hard) / 20 % (partial) of switched isoform ratios to their target module at a 6–8 % false rate, at 10–25 minutes per fit.
* Benchmark methods `usage_network` (WGCNA on logit isoform usage), `ica`, `nmf`, `acorde` (acorde pipeline re-implemented with dynamicTreeCut; its single-cell correlation filter and forced expansion off by default on bulk data) and `twn` (transcriptome-wide network with the Saha et al. penalty structure solved by R glasso, scored with edge-level metrics: gene–gene precision/recall and isoform-ratio edges of switched vs unswitched isoforms). Module methods report the same gene-level edge metrics from their labels.

Abundance confounding

* `nb_isoform_modules` gains usage-driven regulation (`usage_rate`, `usage_strength`, `usage_from` = signal | null | all): one non-dominant isoform's share of its gene follows a module factor with the shares re-normalised per sample, so the gene total is unchanged and only the isoform split varies. The switch table records `mechanism` (switch | usage | none), `is_usage` and `usage_strength`; `is_switched` covers both mechanisms.
* `extract-modules --gene-coherence` (and the `bsvae_coherent` benchmark method): isoforms placed apart from their gene's majority module are moved back unless their sibling-residual kME supports the other membership. On the confounding simulations this cuts BSVAE's false-split rate from 28–46 % to 11–19 % (WGCNA 22–34 %) and raises transcript ARI by ~0.1, at the cost of switch detection (0.89 → 0.63 for abundance switches; WGCNA 0.73). `bsvae_hier` benchmarks the hierarchical loss (`--hier-strength 0.05 --isoform-stratified`) for the same purpose (false splits 25–35 %, detection 0.81).
* Gene-level metrics add `false_split_rate` (unswitched multi-isoform genes whose assigned transcripts are spread over two or more modules). `scripts/benchmark_confounding.py` runs the abundance-only, usage-only, usage-vs-gene-signal ratio and switch-versus-usage scenario sets and stratifies false divergence by isoform abundance.

Consensus and isoform-specific modules

* `bsvae-networks consensus-modules` and the `bsvae_consensus` benchmark method: runs are aligned by module overlap, features take the majority label, and the agreement fraction is a stability score.
* `bsvae-networks isoform-specific` and the `bsvae_isoform` benchmark method: each isoform's profile is regressed on its sibling isoforms (leave-one-out gene aggregate) and the residual's kME to other modules flags isoform-specific membership; `isoform-residuals` writes the residual matrix. Clustering residuals alone was evaluated and found uninformative (switched isoforms are too few per module to form clusters).
* Isoform-specific calls correlate the residual with eigengenes residualised on the gene's own module eigengene, which halves false calls among unswitched minor isoforms (leakage through correlated modules), and add a BH-corrected correlation test (`--alpha`, default 0.05) with `p_value`/`q_value`/`mean_expr` columns; the kME floor defaults to 0.3 (was 0.4) at equal or better precision. Rare (low-abundance) switched isoforms remain limited by sampling noise: `--masked-recon`, low-expression filtering and NB Pearson-residual input (`bsvae.utils.preprocess.pearson_residuals`, new, with `size_factors`) were evaluated for them; the first two reduce recovery and the third trades 0.05 transcript-level ARI for no gain, so none is a default. `isoform_residuals` / `isoform_specific_calls` accept `input_scale="linear"` for matrices that add directly. Simulated scenarios write the raw count matrix (`expr/counts_features_x_samples.tsv.gz`; `outputs.emit_counts: false` to skip).
* `bsvae-networks target-genes`: targeted-gene mode for candidate genes or GWAS loci (gene list, transcript list, or BED loci mapped to genes through a GTF/BED annotation by overlap, window or N closest). Every target isoform is tested for membership of a module other than its gene's against a permutation, abundance-matched or analytic null, with BH within the target set, an optional reliability-corrected kME from raw counts, a case/control interaction test, per-gene divergence, per-locus concentration and module-neighbour tables. Nothing is re-trained; the modules come from the transcriptome-wide run.
* Optional size-invariant neighbour-contrastive latent loss (`--contrastive-strength`).
* Optional diagonal (per-dimension) component covariance in the GMM prior (`--diag-covariance`).
* `bsvae-networks module-quality`: kME summaries, GMT enrichment with q-values, and preservation in a second dataset (permutation Z-density and kME correlation).
* `bsvae-networks regress-covariates` removes known sample covariates before training; `bsvae_regressed` / `wgcna_regressed` benchmark methods.

Performance

* Network edge lists are built with vectorised NumPy instead of Python double loops; the hierarchical loss evaluates all isoform pairs of a batch in one pass; `--amp` enables a bfloat16 forward pass on CUDA.

Default changes

* `--bal-type` entropy; KL per element; `--collapse-min-usage` 0.002; sweep `--free-bits` 0.0; extract-modules post-processing on with `--merge-cut-height` 0.4; GTEx-calibrated simulation defaults.


### Enhancements

* Added dynamic versioning via Poetry (`poetry-dynamic-versioning`).
* Added CLI tool for downloading STRING PPI caches.
* Expanded dataset support to TSV and compressed `.gz` formats.
* Added `--log-level` flag and improved logging controls.
* Standardized training loss logs to CSV format.

### Model & Metadata Improvements

* Normalized metadata for all StructuredFactorVAE checkpoints.
* Ensured model input dimensions and Laplacian buffers are fully persisted.
* Improved device handling for Laplacian matrices.

### Dataset & Evaluation Fixes

* Corrected gene expression matrix orientation.
* Fixed evaluation batch size defaults.
* Added validation for evaluation-time gene dimension mismatches.
* Improved sample splitting logic during evaluation.

### Stability & Reliability Fixes

* Handled missing loss log files gracefully.
* Corrected directory creation for sparse logging cases.
* Fixed test-loss loader and ensured evaluation uses `drop_last=False`.
