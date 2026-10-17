# Frequently Asked Questions

| Problem | Cause | Solution |
| --- | --- | --- |
| `the following arguments are required: --dataset` | `bsvae-train` requires dataset path | Pass `--dataset path/to/matrix.csv`. |
| `RuntimeError` or OOM during training | Batch too large for available memory | Reduce `--batch-size`; disable CUDA with `--no-cuda` if needed. |
| `ImportError: faiss-cpu is required for Method B` | `gamma_knn` requested without FAISS | Install `faiss-cpu` in environment. |
| `ValueError` for hierarchical loss mapping | `--hier-strength` enabled without usable feature index mapping | Provide valid `--tx2gene` and ensure feature IDs align with dataset rows. |
| Missing `soft_eigengenes.csv` | `--soft-eigengenes` needs `--expr` | Add `--expr <features x samples matrix>`. |
| `Rscript not found` or WGCNA rows are `skipped` in `bsvae-benchmark` | R or the `WGCNA` package is missing | Install R with `WGCNA` (and `glasso` for TWN); check with `bsvae-benchmark check-env`. |
| Most features end up in the grey module (`-1`) | Strict post-processing (`--min-gamma`, `--min-module-size`, `--min-kme`) or too few training components | Train with about three times the expected module count and relax the thresholds; compare methods at matched coverage, not on `ari_assigned` alone. |
| `isoform-specific` calls many transcripts whose label equals the tested module | The module was learned on the same samples, so the p-value is selection-biased | Treat the p-value as a ranking; cross-fit (`--folds`), pass `--covariates`, set `--max-target-corr 0.4`, and confirm on held-out samples or with `target-genes-combine`. |
| `np.load` fails on `gamma.npz` with an object-array error | `feature_ids` is stored as an object array | Use `np.load(path, allow_pickle=True)`. |
| `gene_coherence.csv` or `isoform_specific.csv` is missing | `--tx2gene` was not passed, or `--no-gene-coherence` / `--no-isoform-calls` was set | Pass `--tx2gene` and `--expr` to `extract-modules`. |
