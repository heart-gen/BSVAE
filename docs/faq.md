# Frequently Asked Questions

| Problem | Cause | Solution |
| --- | --- | --- |
| `the following arguments are required: --dataset` | `bsvae-train` requires dataset path | Pass `--dataset path/to/matrix.csv`. |
| `RuntimeError` or OOM during training | Batch too large for available memory | Reduce `--batch-size`; disable CUDA with `--no-cuda` if needed. |
| `ImportError: faiss-cpu is required for Method B` | `gamma_knn` requested without FAISS | Install `faiss-cpu` in environment. |
| `ValueError` for hierarchical loss mapping | `--hier-strength` enabled without usable feature index mapping | Provide valid `--tx2gene` and ensure feature IDs align with dataset rows. |
| Missing `soft_eigengenes.csv` | `--soft-eigengenes` needs `--expr` | Add `--expr <features x samples matrix>`. |
