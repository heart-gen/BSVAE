"""
Encoder for StructuredFactorVAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredEncoder(nn.Module):
    """
    Encoder for StructuredFactorVAE.
    Maps normalized gene expression (log-CPM/TPM) to latent Gaussian parameters.

    Parameters
    ----------
    n_genes : int
        Number of input genes (features).
    n_latent : int
        Dimensionality of latent space (number of biological modules).
    hidden_dims : list of int, optional
        Sizes of hidden layers. Default: [512, 256, 128].
    dropout : float
        Dropout probability applied after hidden layers.
    """
    def __init__(self, n_genes: int, n_latent: int,
                 hidden_dims: list[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128]
        self.dropout = dropout

        # Build feedforward encoder network
        modules = []
        input_dim = n_genes
        for h_dim in self.hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.ReLU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            input_dim = h_dim
        self.encoder = nn.Sequential(*modules)

        # Outputs: mean and log-variance
        self.fc_mu = nn.Linear(input_dim, n_latent)
        self.fc_logvar = nn.Linear(input_dim, n_latent)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input expression data (batch, G).

        Returns
        -------
        mu : torch.Tensor
            Latent mean (batch, K).
        logvar : torch.Tensor
            Latent log variance (batch, K).
        z : torch.Tensor
            Sampled latent code (batch, K).
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar
