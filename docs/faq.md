# Frequently Asked Questions

| Problem | Cause | Solution |
| --- | --- | --- |
| `Specify exactly one of --gene-expression-filename or --gene-expression-dir.` | Both dataset flags were provided. | Choose a single data source; omit the unused flag. |
| Training loss becomes `NaN`. | Learning rate too high or input data contains invalid values. | Lower `--lr`, ensure the dataset has finite values, and standardize expression counts. |
| GPU is not utilized. | PyTorch fails to detect CUDA or `--no-cuda` was set. | Remove `--no-cuda` and verify `torch.cuda.is_available()` returns `True`. |
| Latent factors look redundant. | β or L1 penalties too low. | Increase `--beta` and/or `--l1-strength` to encourage disentanglement and sparsity. |
| Eval-only run fails. | Model checkpoint is missing. | Ensure training completed successfully or double-check the experiment name passed to `bsvae-train`. |

💡 **Tip:** Enable debug logging by selecting the `[debug]` configuration section to inspect data loading and loss breakdowns.
