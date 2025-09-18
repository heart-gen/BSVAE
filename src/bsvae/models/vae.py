import torch
import torch.nn as nn

class BaseVAE(nn.Module):
    """
    Base class for VAE models.
    Provides the forward loop and reparameterization trick.
    """
    def __init__(self, n_genes: int, n_latent: int):
        super().__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Sample z using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Subclasses must implement forward pass.")

    def sample_latent(self, x: torch.Tensor):
        """Convenience: encode + sample latent z."""
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)
