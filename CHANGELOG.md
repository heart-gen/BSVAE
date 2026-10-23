# Changelog

Versions follow the git tags (`vX.Y.Z`). Benchmark results quoted here are simulation results; the full tables and the scripts that produce them are described in [docs/benchmarking.md](docs/benchmarking.md).

## Unreleased

### Added

* `target-genes --max-target-corr R` (and `target_gene_tests(max_target_corr=...)`): exclude as targets the modules whose eigengene correlates with the gene's own module above `R`, as `isoform-specific` already does. The exclusion applies to the observed best module and inside the permutation, matched and analytic nulls. The output gains `n_targets`, the number of eligible target modules; an isoform with none gets `best_other_module = -1` and `p_value = 1`.

### Fixed

* `target-genes` on matrices whose feature IDs carry versions (GENCODE `ENST….N`): `resolve_target_transcripts` returned version-stripped IDs, which matched no row of the matrix, so `target_gene_tests` returned an empty table and the command failed with `KeyError: 'called'` (or `'gene_id'` with `--bed`). The resolved targets now keep the matrix's own feature IDs.

## 0.4.0 - 2026-10-17

Transcript-level review: gene-coherent modules and a calibrated isoform-specific residual scan as two independent outputs, targeted-gene and cross-cohort tests, WGCNA-style post-processing, a benchmark harness with comparison methods, and a set of training correctness fixes.

### Highlights

* **Two outputs at transcript level.** `extract-modules --expr --tx2gene` writes module assignments annotated by the gene-coherence policy (default; `--no-gene-coherence` to disable) and the isoform-specific residual scan over every multi-isoform transcript (`isoform_specific.csv`; `--no-isoform-calls` to skip). `modules_raw.csv` keeps the labels before the policy. A transcript's module label never decides whether it is tested.
* **Calibrated residual scan.** `--loo-eigengene {approx,exact,off}` removes a transcript's own gene from the eigengene it is tested against (default `approx`); `--covariates` regresses technical covariates out of the residuals and eigengenes; `--max-target-corr` excludes target modules that are split components of the gene's own module. Cross-fitting (`isoform-specific --folds`, with `--samples` / `--exclude-samples` on `bsvae-train` and `extract-modules`) scores each fold against modules learned without it. On the simulations these bring the null type-I error to about 0.05; the shipped defaults were not shown to meet a pre-registered type-I bound in every simulated cell, so treat `called` as an FDR-controlled ranking to confirm on held-out samples or another region.
* **Targeted genes and cross-cohort evidence.** `target-genes` tests the isoforms of candidate genes or GWAS loci (gene list, transcript list or BED) against the transcriptome-wide modules, with a permutation, matched or analytic null and an optional case/control interaction test. `target-genes-combine` pools tables across cohorts, tissues or regions with Stouffer's method and Strube's correction for shared donors (`--correlation empirical`, `--min-called`, `--min-cohorts`).
* **New `bsvae-networks` commands:** `consensus-modules` (majority vote over seeds with a stability score), `isoform-specific`, `isoform-residuals`, `target-genes`, `target-genes-combine`, `module-quality` (kME, GMT enrichment, preservation in a second dataset) and `regress-covariates`.
* **WGCNA-style post-processing** of GMM assignments (`bsvae.networks.postprocess`): grey class (`--min-gamma`), minimum module size, eigengene-correlation merging, kME; on by default in `extract-modules` and `bsvae-simulate benchmark`. Train with about three times the expected module count and let post-processing merge back.
* **`bsvae-benchmark`** (new entry point): resumable BSVAE-versus-WGCNA harness with comparison methods `usage_network`, `ica`, `nmf`, `acorde` and `twn`, grey-aware ARI/NMI, best-match Jaccard, gene-level and edge metrics, and matched-coverage metrics (`ari_at_grey_*`, `coverage_metrics`). Options: `--rep-ids`, `--scenario-ids`, `--over-cluster-factor`, `--no-resume`, `--no-normalize-input`, `summarize --by`. R is needed for the WGCNA, acorde and TWN methods.
* **Simulation.** Transcript-level family `nb_isoform_modules` (variable isoforms per gene, switching and usage-driven regulation, null genes, second cohorts via `sample_seed`, tissue shifts, shared donors), median-of-ratios normalisation instead of CPM, GTEx-calibrated defaults, paired random streams, YAML configs (`pip install bsvae[yaml]`), and raw counts written to `expr/counts_features_x_samples.tsv.gz`.
* **Benchmark scripts** in `scripts/`: `benchmark_sims`, `benchmark_confounding`, `benchmark_preservation`, `benchmark_kme_preservation`, `benchmark_calibration`, `benchmark_pooling`, `benchmark_rare_isoforms`, `benchmark_defaults` (residual-scan defaults sweep) and `calibrate_nb_from_gtex`.
* **Training additions:** `--amp` (bfloat16 forward pass on CUDA), optional neighbour-contrastive latent loss (`--contrastive-strength`), optional diagonal component covariance (`--diag-covariance`), `--isoform-stratified` gene-grouped batching for the hierarchical loss, and vectorised network edge construction.

### Fixed

* The KL term is scaled per observed value (`kl_per_element`, default on) so `--beta 1` is a true ELBO weight; previously the effective beta grew with the number of samples and collapsed gamma. `--no-kl-per-element` restores the old behaviour.
* Phase 1 (warm-up) reconstructs the same z-scored target as Phase 2 when `--normalize-input` is set; `--kl-warmup-epochs` ramps inside the GMM phase and evaluation uses full beta.
* Dead-component revival splits the most-used component instead of assigning a near-zero mixing weight, never runs after the final epoch or during the transition ramp, and uses an absolute floor (`--collapse-min-usage`, default 0.002).
* GMM sigma floor is a softplus parameterisation; separation loss is averaged over pairs; encoder log-variance is clamped; `--masked-recon` keeps negative values.
* `tx2gene` loading auto-detects headers, strips Ensembl versions and GENCODE pipe fields, warns on low coverage, and no longer infers genes from transcript numbers.
* `bsvae-sweep-k` shares its parser and builders with `bsvae-train`, selects on held-out ELBO and splits by gene when `--tx2gene` is given.
* Latent export and analysis apply input normalisation; `latent-analysis --covariates` correlates covariates with module eigengenes; `--use-leiden` no longer densifies the adjacency; `.tsv.gz` separators are inferred correctly.
* `isoform_specific_calls(max_target_corr=...)` no longer fails when the eigengene matrix has a single module; single-module target exclusion is handled in the CLI.

### Changed

* Defaults: `--bal-type entropy`, KL per element, `--collapse-min-usage 0.002`, sweep `--free-bits 0.0`, post-processing on in `extract-modules` with `--merge-cut-height 0.4`, kME floor 0.3 for the isoform-specific caller (was 0.4).
* `bsvae-benchmark run --k-mode` accepts only `oracle`: the `sweep` choice was never wired to a K selection (the harness always used the simulated module count), so it is now rejected instead of silently ignored.
* Removed dead legacy modules (`main.py`, `utils/evaluate.py`, `networks/cli.py`, `utils/initialization.py`) and the legacy `hyperparam.ini`; the CLI takes explicit flags only.
* Documentation: new [Benchmarking](docs/benchmarking.md) page, worked examples built from the benchmark repository, installation notes for the `anndata` and `yaml` extras and R, and a release procedure in [Contributing](docs/contributing.md).

## 0.3.2 - 2026-03-20

* Hierarchical loss deduplicates by dataset feature index and uses O(B) lookups; `--normalize-input` support, defaults tuning and correctness fixes; first isoform-level tests.

## 0.3.1 - 2026-02-27

* `bsvae-sweep-k` for choosing the number of modules; fixes for clustering collapse, frozen priors and the rho/gamma mix-up; documentation refresh.

## 0.3.0 - 2026-02-20

* `GMMModuleVAE` replaces the earlier PPI-based pipeline end to end; blockwise coexpression loss with soft thresholding; new tutorial and CLI documentation.

## 0.2.x

* 0.2.1: two-phase Leiden resolution search, sparse edge-list support in module extraction.
* 0.2.0: streaming and sparse/Parquet storage for network extraction to cut memory use; Leiden fixes.

## 0.1.x

* Initial releases (2025-11-01 to 2025-12-09).
