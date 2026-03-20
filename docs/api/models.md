# Model API

These notes cover the main stable model-facing objects that matter for users extending the package in Python.

## `GMMModuleVAE`

Defined in `src/bsvae/models/gmvae.py`.

```python
GMMModuleVAE(
    n_features: int,
    n_latent: int,
    n_modules: int,
    hidden_dims: list[int] | None = None,
    dropout: float = 0.1,
    use_batch_norm: bool = True,
    sigma_min: float = 0.3,
    normalize_input: bool = False,
)
```

The model is trained on feature profiles, so `n_features` here refers to the profile length, which is the number of samples in the expression matrix.

### Common Methods

- `encode(x) -> (mu, logvar)`
- `forward(x) -> recon_x, mu, logvar, z, gamma`
- `get_gamma(x) -> gamma`
- `get_hard_assignments(x) -> argmax(gamma)`

### Tensor Shapes

- `x`: `(batch, n_samples)`
- `mu`, `logvar`, `z`: `(batch, n_latent)`
- `gamma`: `(batch, n_modules)`

## Related Components

- `FeatureEncoder` in `src/bsvae/models/encoder.py`
- `FeatureDecoder` in `src/bsvae/models/decoder.py`
- `GaussianMixturePrior` in `src/bsvae/models/gmm_prior.py`

## Losses

`src/bsvae/models/losses.py` exposes the main training losses:

- `GMMVAELoss`
- `WarmupLoss`

`GMMVAELoss` combines reconstruction with GMM-aware KL and optional auxiliary losses such as separation, balance, hierarchical consistency, and correlation-preservation terms.
