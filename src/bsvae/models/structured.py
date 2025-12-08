"""
Structured Factor VAE: wraps encoder + decoder for end-to-end training.
"""
import torch
import torch.nn as nn
from .vae import BaseVAE
from .encoder import StructuredEncoder
from .decoder import StructuredDecoder
from ..utils.initialization import weights_init

class StructuredFactorVAE(BaseVAE):
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
                 hidden_dims=None, dropout: float = 0.1,
                 mask: torch.Tensor = None,
                 init_sd: float = 0.02,
                 learn_var: bool = False,
                 L: torch.Tensor = None):
        super().__init__(n_genes, n_latent)
        self.hidden_dims = hidden_dims or [512, 256, 128]
        self.dropout = dropout
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
        if L is not None:
            # Register Laplacian so it moves with the model across devices
            self.register_buffer("laplacian_matrix", L)
        else:
            self.laplacian_matrix = None

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x, log_var = self.decoder(z)
        return recon_x, mu, logvar, z, log_var

    def group_sparsity_penalty(self, l1_strength: float = 1e-3):
        return self.decoder.group_sparsity_penalty(l1_strength)

    def laplacian_penalty(self, L: torch.Tensor, lap_strength: float = 1e-3):
        return self.decoder.laplacian_penalty(L, lap_strength)

    def reset_parameters(self, activation: str = "relu"):
        """Reset all learnable parameters with custom init."""
        self.activation = activation
        self.apply(lambda m: weights_init(m, activation=self.activation))
