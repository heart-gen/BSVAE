"""
Losses for StructuredFactorVAE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def gaussian_nll(x, recon_x, log_var=None, reduction="mean"):
    """
    Gaussian negative log-likelihood per gene.
    """
    if log_var is None:
        return F.mse_loss(recon_x, x, reduction=reduction)

    var = torch.exp(log_var)
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (x - recon_x) ** 2 / var)
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


def kl_normal_loss_with_free_bits(mu, logvar, free_bits=0.0, reduction="mean"):
    """
    KL divergence with free bits (per-dimension minimum).

    Computes per-dimension KL averaged across the batch, then clamps each
    dimension to at least ``free_bits`` nats before summing across dimensions.

    Parameters
    ----------
    mu : torch.Tensor
        Latent mean (batch, K).
    logvar : torch.Tensor
        Latent log variance (batch, K).
    free_bits : float
        Minimum KL per latent dimension (nats). 0 disables.
    reduction : str
        "mean" (default) divides by batch size after summing dims;
        "sum" returns the raw sum; "none" returns per-sample vector.

    Returns
    -------
    kl : torch.Tensor
        Scalar KL loss (or per-sample if reduction="none").
    kl_per_dim : torch.Tensor
        Per-dimension KL averaged over the batch, shape (K,).
    """
    # Per-sample, per-dimension KL: (batch, K)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    # Average across batch to get per-dimension KL: (K,)
    kl_per_dim = kl.mean(dim=0)

    if free_bits > 0:
        # Clamp each dimension to at least free_bits nats
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
    else:
        kl_per_dim_clamped = kl_per_dim

    # Sum across dimensions
    kl_total = kl_per_dim_clamped.sum()

    if reduction == "mean":
        return kl_total, kl_per_dim
    elif reduction == "sum":
        return kl_total * mu.shape[0], kl_per_dim
    else:
        return kl, kl_per_dim


def coexpression_loss(x: torch.Tensor, recon_x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Frobenius MSE between input and reconstructed gene correlation matrices."""
    if x.ndim != 2 or recon_x.ndim != 2:
        raise ValueError("coexpression_loss expects 2D tensors (batch, genes)")
    if x.shape != recon_x.shape:
        raise ValueError("x and recon_x must have matching shapes")
    if x.shape[0] < 2:
        return x.new_tensor(0.0)

    x_c = x - x.mean(dim=0, keepdim=True)
    r_c = recon_x - recon_x.mean(dim=0, keepdim=True)
    denom = float(x.shape[0] - 1)

    x_cov = (x_c.T @ x_c) / denom
    r_cov = (r_c.T @ r_c) / denom

    x_std = torch.sqrt(torch.diag(x_cov).clamp(min=eps))
    r_std = torch.sqrt(torch.diag(r_cov).clamp(min=eps))
    x_corr = x_cov / torch.outer(x_std, x_std)
    r_corr = r_cov / torch.outer(r_std, r_std)
    return F.mse_loss(r_corr, x_corr)


class BaseLoss(nn.Module):
    """
    Base loss for StructuredFactorVAE with biological regularizers.
    """
    def __init__(self, beta: float = 1.0,
                 l1_strength: float = 1e-3,
                 lap_strength: float = 1e-4,
                 record_loss_every: int = 50,
                 kl_warmup_epochs: int = 0,
                 kl_anneal_mode: str = "linear",
                 kl_cycle_length: int = 50,
                 kl_n_cycles: int = 4,
                 free_bits: float = 0.0,
                 coexpr_strength: float = 0.0):
        super().__init__()
        self.beta = beta
        self.l1_strength = l1_strength
        self.lap_strength = lap_strength
        self.n_train_steps = 0
        self.record_loss_every = record_loss_every
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_mode = kl_anneal_mode
        self.kl_cycle_length = kl_cycle_length
        self.kl_n_cycles = kl_n_cycles
        self.free_bits = free_bits
        self.coexpr_strength = coexpr_strength

    def get_beta_for_epoch(self, epoch: int) -> float:
        """Return effective beta based on annealing schedule."""
        if self.kl_anneal_mode == "cyclical":
            # Cyclical annealing: repeat linear warmup over cycles
            cycle_pos = epoch % self.kl_cycle_length
            ratio = min(cycle_pos / max(self.kl_cycle_length // 2, 1), 1.0)
            return self.beta * ratio
        else:
            # Linear warmup then constant
            if self.kl_warmup_epochs <= 0:
                return self.beta
            ratio = min(epoch / self.kl_warmup_epochs, 1.0)
            return self.beta * ratio

    def forward(self,
                x: torch.Tensor, recon_x: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor,
                model: nn.Module,
                L: Optional[torch.Tensor] = None,
                storer: Optional[Dict[str, list]] = None,
                is_train: bool = True,
                epoch: int = 0) -> torch.Tensor:
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
        model : nn.Module
            Structured decoder (with sparsity/laplacian methods).
        storer : dict
            Dictionary for logging intermediate values.
        is_train : bool
            Whether in training mode.
        epoch : int
            Current epoch (for KL annealing).

        Returns
        -------
        loss : torch.Tensor
            Total loss for this batch.
        """
        # Reconstruction loss: Gaussian NLL if available, else MSE
        if hasattr(model.decoder, "log_var") and model.decoder.log_var is not None:
            recon_loss = gaussian_nll(x, recon_x, log_var=model.decoder.log_var,
                                      reduction="mean")
        else:
            recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence with free bits
        kl_loss, kl_per_dim = kl_normal_loss_with_free_bits(
            mu, logvar, free_bits=self.free_bits, reduction="mean"
        )

        # Effective beta from annealing schedule
        effective_beta = self.get_beta_for_epoch(epoch)

        # Biological regularizers
        sparsity_loss = model.decoder.group_sparsity_penalty(self.l1_strength)
        laplacian_loss = 0.0
        if model.laplacian_matrix is not None:
            laplacian_loss = model.decoder.laplacian_penalty(
                model.laplacian_matrix, self.lap_strength
            )

        coexpr_term = x.new_tensor(0.0)
        if self.coexpr_strength > 0:
            coexpr_term = self.coexpr_strength * coexpression_loss(x, recon_x)

        # Total loss
        loss = recon_loss + effective_beta * kl_loss + sparsity_loss + laplacian_loss + coexpr_term

        # Logging
        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("effective_beta", []).append(effective_beta)
            storer.setdefault("sparsity_loss", []).append(sparsity_loss.detach().item())
            if model.laplacian_matrix is not None:
                storer.setdefault("laplacian_loss", []).append(laplacian_loss.detach().item())
            if self.coexpr_strength > 0:
                storer.setdefault("coexpr_loss", []).append(coexpr_term.detach().item())
            storer.setdefault("loss", []).append(loss.item())
            # Per-dimension KL monitoring
            for dim_idx in range(kl_per_dim.shape[0]):
                storer.setdefault(f"kl_dim_{dim_idx}", []).append(
                    kl_per_dim[dim_idx].item()
                )

        return loss
