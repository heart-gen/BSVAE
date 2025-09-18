"""
Losses for StructuredFactorVAE.
"""
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ["VAE", "beta"]

def get_loss_f(loss_name: str = "VAE",
               beta: float = 1.0,
               **kwargs) -> BaseLoss:
    """
    Return the StructuredFactorVAE loss function.

    Parameters
    ----------
    loss_name : str
        Loss type name ("VAE" or "beta").
    beta : float
        KL scaling factor (for beta-VAE style).
    kwargs : dict
        Extra kwargs (ignored, for API compatibility).

    Returns
    -------
    loss_function : BaseLoss instance
    """
    if loss_name not in LOSSES:
        raise ValueError(f"Unknown loss: {loss_name}. Must be one of {LOSSES}.")

    if loss_name == "VAE":
        return BaseLoss(beta=1.0)   # standard ELBO
    elif loss_name == "beta":
        return BaseLoss(beta=beta)  # scaled KL


class BaseLoss(nn.Module):
    """
    Base class for VAE losses with biological regularizers.
    """
    def __init__(self, beta: float = 1.0, record_loss_every: int = 50):
        super().__init__()
        self.beta = beta
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every

    def forward(self,
                x: torch.Tensor,
                recon_x: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor,
                decoder: nn.Module,
                L: Optional[torch.Tensor] = None,
                storer: Optional[Dict[str, list]] = None,
                is_train: bool = True) -> torch.Tensor:
        """
        Compute StructuredFactorVAE loss.

        Parameters
        ----------
        x : torch.Tensor
            Input data (batch, G).
        recon_x : torch.Tensor
            Reconstruction from decoder (batch, G).
        mu : torch.Tensor
            Latent mean (batch, K).
        logvar : torch.Tensor
            Latent log variance (batch, K).
        decoder : nn.Module
            Structured decoder (with sparsity/laplacian methods).
        L : torch.Tensor or None
            Graph Laplacian for Laplacian penalty (optional).
        storer : dict
            Dictionary for logging intermediate values.
        is_train : bool
            Whether in training mode.

        Returns
        -------
        loss : torch.Tensor
            Total loss for this batch.
        """

        # Reconstruction loss: Gaussian NLL if available, else MSE
        if hasattr(decoder, "log_var"):
            recon_loss = gaussian_nll(x, recon_x, log_var=decoder.log_var, reduction="mean")
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence
        kl_loss = kl_normal_loss(mu, logvar, reduction="mean")

        # Regularizers
        sparsity_loss = decoder.group_sparsity_penalty(l1_strength=1e-3)
        laplacian_loss = decoder.laplacian_penalty(L, lap_strength=1e-4) if L is not None else 0.0

        # Combine
        loss = recon_loss + self.beta * kl_loss + sparsity_loss + laplacian_loss

        # Logging
        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("sparsity_loss", []).append(float(sparsity_loss))
            if L is not None:
                storer.setdefault("laplacian_loss", []).append(float(laplacian_loss))
            storer.setdefault("loss", []).append(loss.item())

        return loss


def gaussian_nll(x, recon_x, log_var=None, reduction="mean"):
    """
    Gaussian negative log-likelihood per gene.

    Parameters
    -----------
    x : torch.Tensor
        Observed counts, shape (batch, G).
    recon_x : torch.Tensor
        Reconstructed gene expression (batch, G).
    log_var : torch.Tensor
        Gene-specific log-variance (G,).
    reduction : str
        "mean", "sum", or "none".
    """
    if log_var is None:
        return F.mse_loss(recon_x, x, reduction=reduction)

    var = torch.exp(log_var)
    nll = 0.5 * (torch.log(2 * torch.pi * var) +
                 (x - recon_x) ** 2 / var)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll


def kl_normal_loss(mu, logvar, reduction="sum"):
    """
    KL divergence between q(z|x)=N(mu, sigma^2) and prior p(z)=N(0,I).
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    else:
        return kl
