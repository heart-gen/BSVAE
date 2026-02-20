"""
GMMModuleVAE — Gaussian Mixture VAE for multi-modal module discovery.

Architecture
------------
  Feature profile x_f ∈ R^N
       ↓
  FeatureEncoder (shared MLP)
       ↓
  μ_f, logσ²_f  →  z_f ~ q(z | x_f)   ← z used only for reconstruction
       ↓
  γ_{fk} = q(c=k | μ_f, logσ²_f)      ← (μ, logvar)-based γ
       ↑
  GaussianMixturePrior: p(z) = Σ_k π_k N(z; μ_k, σ²_k I)
       ↓
  FeatureDecoder (linear: z @ W.T + b)
       ↓
  Reconstructed profile x̂_f

The sampled z is used exclusively for reconstruction; γ is computed
from (μ_f, logvar_f) for all regularisers and module extraction.
"""

import torch
import torch.nn as nn

from bsvae.models.vae import BaseVAE
from bsvae.models.encoder import FeatureEncoder
from bsvae.models.decoder import FeatureDecoder
from bsvae.models.gmm_prior import GaussianMixturePrior


class GMMModuleVAE(BaseVAE):
    """
    Gaussian Mixture VAE for biological module discovery.

    Parameters
    ----------
    n_features : int
        Number of samples N (input dimensionality of each feature profile).
    n_latent : int
        Dimensionality of the latent space D.
    n_modules : int
        Number of GMM components K (= number of biological modules).
    hidden_dims : list of int, optional
        Encoder hidden layer sizes. Default: [512, 256, 128].
    dropout : float
        Encoder dropout probability. Default: 0.1.
    use_batch_norm : bool
        Use BatchNorm1d in the encoder. Default: True.
    init_sd : float
        Decoder weight initialisation std. Default: 0.02.
    sigma_min : float
        GMM component σ floor. Default: 0.3.
    ema_alpha : float
        EMA decay for γ-usage balance. Default: 0.99.
    """

    def __init__(
        self,
        n_features: int,
        n_latent: int,
        n_modules: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        init_sd: float = 0.02,
        sigma_min: float = 0.3,
        ema_alpha: float = 0.99,
    ):
        super().__init__(n_features, n_latent)
        self.n_modules = n_modules

        self.encoder = FeatureEncoder(
            n_features=n_features,
            n_latent=n_latent,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        self.decoder = FeatureDecoder(
            n_features=n_features,
            n_latent=n_latent,
            init_sd=init_sd,
        )
        self.gmm_prior = GaussianMixturePrior(
            n_components=n_modules,
            n_latent=n_latent,
            sigma_min=sigma_min,
            ema_alpha=ema_alpha,
        )

    def forward(self, x: torch.Tensor):
        """
        Full forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N)
            Batch of feature profiles.

        Returns
        -------
        recon_x : torch.Tensor, shape (B, N)
            Reconstructed profiles.
        mu : torch.Tensor, shape (B, D)
            Posterior means.
        logvar : torch.Tensor, shape (B, D)
            Posterior log-variances.
        z : torch.Tensor, shape (B, D)
            Sampled latent codes (used for reconstruction only).
        gamma : torch.Tensor, shape (B, K)
            Soft GMM assignments γ_{fk}.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        gamma = self.gmm_prior.posterior_weights(mu, logvar)
        return recon_x, mu, logvar, z, gamma

    def encode(self, x: torch.Tensor):
        """Return (mu, logvar) without sampling."""
        return self.encoder(x)

    def get_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """Return soft GMM assignments for a batch, shape (B, K)."""
        mu, logvar = self.encoder(x)
        return self.gmm_prior.posterior_weights(mu, logvar)

    def get_hard_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Return hard module assignments (argmax γ), shape (B,)."""
        mu, logvar = self.encoder(x)
        return self.gmm_prior.hard_assignments(mu, logvar)
