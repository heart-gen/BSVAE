"""
FeatureDecoder for GMMModuleVAE.

A single linear layer: z @ W.T + b.  Each column of W is the
"loading vector" for one latent dimension, matching the GNVAE/BSVAE
interpretation.  No PPI masking or sparsity penalty.
"""

import torch
import torch.nn as nn


class FeatureDecoder(nn.Module):
    """
    Linear decoder for GMMModuleVAE.

    Maps a latent code z ∈ R^D back to a reconstructed feature profile
    x̂_f ∈ R^N via a single affine transformation.

    Parameters
    ----------
    n_features : int
        Output dimensionality (number of samples N).
    n_latent : int
        Input dimensionality (latent dimension D).
    init_sd : float
        Standard deviation for weight initialization. Default: 0.02.
    """

    def __init__(
        self,
        n_features: int,
        n_latent: int,
        init_sd: float = 0.02,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_latent = n_latent

        self.W = nn.Parameter(torch.empty(n_features, n_latent))
        self.bias = nn.Parameter(torch.zeros(n_features))
        nn.init.normal_(self.W, std=init_sd)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to feature profiles.

        Parameters
        ----------
        z : torch.Tensor, shape (batch, n_latent)
            Sampled latent codes.

        Returns
        -------
        recon_x : torch.Tensor, shape (batch, n_features)
            Reconstructed feature profiles.
        """
        return z @ self.W.T + self.bias
