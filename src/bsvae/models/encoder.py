"""
FeatureEncoder for GMMModuleVAE.

Each data point is one feature (e.g., a gene) with its sample-level
expression profile x_f ∈ R^N.  The encoder maps this profile to the
parameters of a Gaussian approximate posterior q(z | x_f).
"""

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """
    Encoder for GMMModuleVAE.

    Maps a feature profile (expression over N samples) to Gaussian
    approximate posterior parameters (μ_f, logvar_f).

    Parameters
    ----------
    n_features : int
        Number of samples (input dimensionality for each feature profile).
    n_latent : int
        Dimensionality of the latent space (D).
    hidden_dims : list of int, optional
        Sizes of hidden layers. Default: [512, 256, 128].
    dropout : float
        Dropout probability applied after hidden layers. Default: 0.1.
    use_batch_norm : bool
        Whether to insert BatchNorm1d after each Linear layer. Default: True.
    """

    def __init__(
        self,
        n_features: int,
        n_latent: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_latent = n_latent
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128]
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Build feedforward encoder network
        layers = []
        input_dim = n_features
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        # Output heads: mean and log-variance of q(z | x_f)
        self.fc_mu = nn.Linear(input_dim, n_latent)
        self.fc_logvar = nn.Linear(input_dim, n_latent)

    def forward(self, x: torch.Tensor):
        """
        Encode a batch of feature profiles.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)
            Batch of feature profiles (each row = one feature's sample profile).

        Returns
        -------
        mu : torch.Tensor, shape (batch, n_latent)
            Posterior mean μ_f for each feature.
        logvar : torch.Tensor, shape (batch, n_latent)
            Posterior log-variance logσ²_f for each feature.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
