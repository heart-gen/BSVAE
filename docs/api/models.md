# Model API

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
    init_sd: float = 0.02,
    sigma_min: float = 0.3,
    ema_alpha: float = 0.99,
)
```

### Forward signature

```python
recon_x, mu, logvar, z, gamma = model(x)
```

- `x`: `(batch, n_features)`
- `recon_x`: reconstruction
- `mu`, `logvar`: encoder posterior params
- `z`: reparameterized latent sample
- `gamma`: soft GMM assignments

### Convenience methods

- `encode(x) -> (mu, logvar)`
- `get_gamma(x) -> gamma`
- `get_hard_assignments(x) -> argmax(gamma)`

## Related model classes

- `FeatureEncoder` (`src/bsvae/models/encoder.py`)
- `FeatureDecoder` (`src/bsvae/models/decoder.py`)
- `GaussianMixturePrior` (`src/bsvae/models/gmm_prior.py`)

## Losses

`src/bsvae/models/losses.py` exposes:

- `GMMVAELoss`
- `WarmupLoss`
- helpers such as `kl_vade`, `hierarchical_loss`, `gaussian_nll`

`GMMVAELoss` combines reconstruction, KL (VaDE-style), separation, balance, and optional hierarchical terms.
