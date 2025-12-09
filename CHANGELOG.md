## Changelog

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
