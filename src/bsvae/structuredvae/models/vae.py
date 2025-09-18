"""
Structured Factor VAE: wraps encoder + decoder for end-to-end training.
"""
import torch
import torch.nn as nn
from .encoder import StructuredEncoder
from .decoder import StructuredDecoder
from ..utils.initialization import weights_init

class StructuredFactorVAE(nn.Module):
    """
    Structured Factor VAE for gene expression.

    Encoder: maps log-CPM input (batch, G) → latent mean + logvar.
    Decoder: maps latent z → reconstructed expression (batch, G).
             Supports optional gene-specific variance (log_var).

    Parameters
    ----------
    n_genes : int
        Number of input genes.
    n_latent : int
        Number of latent dimensions (modules).
    hidden_dims : list of int
        Hidden layer sizes for encoder.
    dropout : float
        Dropout probability in encoder.
    mask : torch.Tensor or None
        Optional binary gene×module mask for decoder (G, K).
    init_sd : float
        Std for decoder weight initialization.
    learn_var : bool
        If True, decoder learns per-gene log variance.
    """
    def __init__(self, n_genes: int, n_latent: int,
                 hidden_dims=None, 
                 dropout: float = 0.1,
                 mask: torch.Tensor = None,
                 init_sd: float = 0.02,
                 learn_var: bool = False):
        super().__init__()
        self.encoder = StructuredEncoder(
            n_genes=n_genes,
            n_latent=n_latent,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.decoder = StructuredDecoder(
            n_genes=n_genes,
            n_latent=n_latent,
            mask=mask,
            init_sd=init_sd,
            learn_var=learn_var
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick to sample z ~ N(mu, sigma^2)."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            # Use mean for deterministic inference
            return mu

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x, log_var = self.decoder(z)
        return recon_x, mu, logvar, z, log_var

    def group_sparsity_penalty(self, l1_strength: float = 1e-3):
        return self.decoder.group_sparsity_penalty(l1_strength)

    def laplacian_penalty(self, L: torch.Tensor, lap_strength: float = 1e-3):
        return self.decoder.laplacian_penalty(L, lap_strength)

    def reset_parameters(self):
        """Reset all learnable parameters with custom init."""
        self.apply(weights_init)

    def sample_latent(self, x):
        """Return a latent sample z given input x."""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
