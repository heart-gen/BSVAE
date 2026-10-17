# Benchmarking

BSVAE ships a benchmark harness (`bsvae-benchmark`), a simulator (`bsvae-simulate`) and a set of evaluation scripts under `scripts/`. The full study that uses them, with SLURM drivers, task tables and the committed result tables, lives in the companion repository [`bsvae-benchmarking`](https://github.com/heart-gen/bsvae-benchmarking), which pins BSVAE to an exact commit. This page describes the workflow and the files it produces, so you can run a small version locally or reproduce the study on a cluster.

All benchmark data are **simulated** (the simulator is calibrated to GTEx v11 negative-binomial parameters, `scripts/gtex_calibrated_params.json`). Real-data analyses use the CLI chain in the [Tutorial](tutorial.md).

## 1. Run a small benchmark locally

```bash
# check torch and R (WGCNA, dynamicTreeCut, glasso) are available
bsvae-benchmark check-env

# a transcript-level scenario grid
bsvae-simulate init-config --output iso.yaml --isoform

# BSVAE against WGCNA, three replicates of every scenario
bsvae-benchmark run --config iso.yaml --outdir results/bench \
  --methods bsvae,wgcna --reps 3 --seeds 0 --epochs 150

# mean and sd per scenario and method
bsvae-benchmark summarize --results results/bench/results.tsv --by scenario_id,method
```

`run` appends one row per (scenario, replicate, method, seed) to `results.tsv` as it finishes, so re-running the same command resumes; `--no-resume` starts over. Restrict the work with `--scenario-ids S001__...,S002__...` or `--rep-ids 7` (one replicate, for example one SLURM array task).

Two options change what is being compared:

- `--over-cluster-factor` (default 3) sets the number of GMM components: BSVAE is trained with the true module count of the scenario times this factor and post-processing merges back. The harness always takes the count from the simulation (`--k-mode oracle`, the only mode); on real data choose it as in [Choosing K](hyperparameters.md#choosing-k).
- `--no-normalize-input` turns off the per-feature z-scoring of the input that the harness applies before encoding (on by default for BSVAE methods; the same step is `--normalize-input` on `bsvae-train`).

## 2. Methods

| Method | What it is |
| --- | --- |
| `bsvae_raw` | BSVAE with raw argmax-γ labels (no gene-coherence policy) |
| `bsvae` | BSVAE with post-processing and the gene-coherence policy (the default workflow) |
| `bsvae_hier` | `bsvae` trained with the hierarchical gene-to-isoform loss (`--hier-strength`) |
| `bsvae_isoform` | the isoform-specific residual scan; scored on switched-isoform recall and precision |
| `wgcna_raw`, `wgcna` | WGCNA without and with the same gene-coherence policy |
| `acorde`, `twn`, `usage_network`, `ica`, `nmf` | comparison methods; see the [CLI reference](cli.md) |

The gene-level scenarios skip `bsvae_hier` and `bsvae_isoform` because they need a transcript-to-gene map.

## 3. What a replicate contains

`bsvae-simulate generate-grid` (or `generate-scenario` for one replicate) writes, per replicate:

```text
scenarios/<scenario_id>/rep_000/
├── expr/
│   ├── features_x_samples.tsv.gz         # log-normalised matrix the methods read
│   └── counts_features_x_samples.tsv.gz  # raw counts, for count-aware methods
├── covariates.tsv.gz                     # sample_id, condition, batch, libsize
├── truth/
│   ├── modules_hard.csv                  # feature_id, module, is_null  (null = -1)
│   ├── tx2gene.tsv                       # transcript_id, gene_id   (isoform family)
│   ├── modules_gene.csv
│   ├── modules_long.csv                  # feature_id, module, is_hub, membership_weight
│   ├── switch_table.tsv.gz               # is_switched, mechanism, is_usage, is_hub, ...
│   ├── gene_metadata.tsv.gz
│   ├── edge_list.tsv.gz
│   └── module_latents.tsv.gz
├── metadata.json
├── method_inputs.json
└── methods/<method>/seed_0/
```

and each BSVAE method directory holds:

| File | Contents |
| --- | --- |
| `modules.csv` | `feature_id, module` after post-processing and the coherence policy (`-1` = grey) |
| `modules_raw.csv` | labels before the policy |
| `gamma.npz` | `gamma`, `mu`, `gamma_raw`, `feature_ids` (load with `allow_pickle=True`) |
| `gene_coherence.csv` | `feature_id, gene_id, gene_module, module, kme_residual, p_value, q_value, kept` |
| `isoform_specific_calls.csv` | `bsvae_isoform` only: the residual-scan columns documented under `isoform-specific` |
| `model.pt`, `specs.json`, `train_losses.csv`, `metrics.json` | model, settings, loss trace and scores |

WGCNA methods write `eigengenes.csv` (`sample_id, ME*`).

## 4. Metrics

Scores are grey-aware: `ari_all` / `nmi_all` treat unassigned features as their own label, `ari_assigned` / `nmi_assigned` drop features that are grey on either side. Also reported are grey precision and recall, best-match Jaccard and F1, and (transcript level) `gene_ari`, switched-isoform recall and precision.

**Matched-coverage metrics.** A method that leaves many features unassigned can score a high `ari_assigned` by being selective, so ARIs at different coverage are not comparable. Every label method is therefore also scored after abstaining on its lowest own-module-kME features at unassigned fractions of 5-50% (`ari_at_grey_*`, computed by `bsvae.simulation.metrics.coverage_metrics` and included in `summarize`), and the companion collector adds the ARI on the common support, the features that every compared method assigns. At equal coverage BSVAE leads WGCNA and acorde on the transcript-level scenarios (see below).

## 5. Run it on a cluster

The companion repository wraps the harness in SLURM array jobs. Each array task reads one row of a TSV task table (`SLURM_ARRAY_TASK_ID + TASK_OFFSET`) and runs

```bash
bsvae-benchmark run --config configs/recovery_isoform.yaml \
  --outdir <results>/rep_003/bsvae --methods bsvae \
  --scenario-ids S001__... --rep-ids 3 --seeds 0 --epochs 150
```

The study is organised in stages, each with its own `_h/` job scripts and `_m/` outputs:

| Stage | Question | Script driven |
| --- | --- | --- |
| `01_recovery` | module recovery on gene and transcript scenarios | `bsvae-benchmark run` |
| `02_preservation` | reproducibility across seeds, donors, cohorts and tissues | `scripts/benchmark_preservation.py`, `benchmark_kme_preservation.py` |
| `03_calibration` | type-I error of the residual scan | `scripts/benchmark_calibration.py` |
| `04_pooling` | combining regions with `target-genes-combine` | `scripts/benchmark_pooling.py` |
| `05_defaults` | sweep of residual-scan and assignment defaults | `scripts/benchmark_defaults.py` |
| `06_collect` | one set of summary tables | `python -m bsvae_benchmark.collect` |

Environment variables the wrappers honour: `BSVAE_ENV`, `BSVAE_SRC`, `EPOCHS` (150), `METHODS`, `SEEDS` (10), `ARRAY_THROTTLE`, `REFRESH_TASKS=1` (rebuild the task tables), `ROWS=r,r,...` (run selected rows). On the PSC cluster used for the study, jobs request `--cpus-per-task` and memory scales with it (2000 MB per CPU), so do not pass `--mem`. See [HPC](hpc.md) for a generic SLURM pattern.

Other scripts in `scripts/` you can run directly: `benchmark_sims.py` (seven laptop-sized reference scenarios), `benchmark_confounding.py` (abundance-only, usage-only and switch-versus-usage sets), `benchmark_rare_isoforms.py` (recovery of low-abundance switched isoforms and the remedies tried), and `calibrate_nb_from_gtex.py` (refits the simulator's negative-binomial parameters from GTEx).

### The defaults sweep

`scripts/benchmark_defaults.py` sweeps the residual-scan settings, each given as a comma list (`--loo approx,exact`, `--covariates on,off`, `--target-corrs none,0.6,0.4,0.3`, `--designs full,crossfit`, `--switch-rates`) over one simulated cell (`--n-genes`, `--n-samples`, `--n-modules`, `--factor-corr`, `--signal-scale`), and reports type-I error, power and false discovery per setting. Its companion in the benchmark repo (`python -m bsvae_benchmark.defaults assignment-sweep`) sweeps the assignment settings (`--min-gamma`, `--min-module-size`, `--merge-cut-height`, `--min-kme`, gene coherence) on trained models.

```bash
python scripts/benchmark_defaults.py --outdir results/defaults/base --seeds 0 \
  --epochs 150 --loo approx --covariates on,off --target-corrs none,0.4
```

## 6. Reading the results

The collector writes gzipped TSVs that load directly with pandas:

```python
import pandas as pd

mc = pd.read_csv("06_collect/_m/matched_coverage_all.tsv.gz", sep="\t",
                 keep_default_na=False, na_values=[""])
```

| Table | Contents |
| --- | --- |
| `harness_results_all`, `summary_<config>` | one row per run; mean, sd and bootstrap 95% CI per scenario and method |
| `matched_coverage_all` | ARI at the reference unassigned fraction and on the common support |
| `preservation_*`, `kme_preservation_by_module_all` | seed, donor, cohort and tissue reproducibility |
| `calibration_*`, `pooling_*` | residual-scan type-I error and multi-region pooling |
| `defaults_residual_*`, `defaults_assignment_*` | the defaults sweeps and the decision tables |

## 7. Headline results

Means over 10 replicates and the 24 isoform scenarios of the study (`recovery_isoform`) or the gene-level grid (`recovery_gene`), BSVAE trained for 150 epochs. These are simulation results, shown to illustrate the outputs and not as a claim about real data.

| Scenario family | Method | `ari_all` | `ari_assigned` | grey fraction |
| --- | --- | --- | --- | --- |
| `recovery_gene` | `bsvae` | 0.750 | 0.847 | 0.054 |
| | `wgcna` | 0.700 | 0.840 | 0.030 |
| | `acorde` | 0.582 | 0.808 | 0.162 |
| `recovery_isoform` | `bsvae` | 0.780 | 0.899 | 0.098 |
| | `bsvae_hier` | 0.734 | 0.849 | 0.104 |
| | `bsvae_raw` | 0.704 | 0.819 | 0.098 |
| | `wgcna` | 0.645 | 0.780 | 0.003 |
| | `wgcna_raw` | 0.537 | 0.651 | 0.003 |
| | `acorde` | 0.438 | 0.841 | 0.350 |

`acorde` leaves 35% of transcripts unassigned, so its `ari_assigned` is not comparable. At matched coverage (ARI at acorde's unassigned fraction, `recovery_isoform`) the values are 0.960 for `bsvae` and 0.937 for `wgcna`; on the common support they are 0.950 (`bsvae`), 0.836 (`wgcna`) and 0.841 (`acorde`).

Reproducibility (ARI): across seeds 0.90 (0.92 with the coherence policy); across donor bootstraps 0.79 (0.81) against 0.72 for WGCNA; across cohorts 0.82 (0.86) against 0.55; with a partially changed tissue 0.61 (0.64) against 0.41.

### Calibration caveat for the residual scan

The residual scan is calibrated on simulations, with these results (null scenarios, nominal level 0.05): type-I error 0.079 with the default approximate leave-one-gene-out eigengene (0.124 with the plain eigengene), and 0.22 among transcripts whose raw label is the module they are tested against. Cross-fitting, covariate adjustment and `--max-target-corr 0.4` bring the overall null type-I error close to nominal in the calibration study. The `05_defaults` sweep did not find a setting that met its pre-registered type-I bound (Wilson upper bound at most 0.075) in every simulated cell, so treat the scan's `called` column as a ranked, FDR-controlled list to be confirmed on held-out samples or another region (`target-genes-combine`), and do not read the shipped defaults as validated for every design.

The BSVAE-versus-competitor differences in the assignment sweep are paired and consistent: the top-ranked setting (`min_gamma 0.5`, `min_module_size 10`, `merge_cut_height 0.4`, `min_kme 0.4`, gene coherence on) has a mean `ari_all` of 0.734, and its paired gain over `acorde`, `wgcna` and `wgcna_raw` is +0.28, +0.13 and +0.22 with bootstrap intervals that exclude zero.
