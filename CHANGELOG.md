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
