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


def coexpression_loss(
    x: torch.Tensor,
    recon_x: torch.Tensor,
    eps: float = 1e-8,
    block_size: int = 512,
    gamma: float = 6.0,
    max_genes: Optional[int] = None,
) -> torch.Tensor:
    """Frobenius MSE between soft-thresholded gene correlation matrices.

    Uses blockwise correlation to avoid materializing a full GxG matrix.
    """
    if x.ndim != 2 or recon_x.ndim != 2:
        raise ValueError("coexpression_loss expects 2D tensors (batch, genes)")
    if x.shape != recon_x.shape:
        raise ValueError("x and recon_x must have matching shapes")
    if x.shape[0] < 2:
        return x.new_tensor(0.0)

    if max_genes is not None and x.shape[1] > max_genes:
        idx = torch.randperm(x.shape[1], device=x.device)[:max_genes]
        x = x[:, idx]
        recon_x = recon_x[:, idx]

    # Center
    x_c = x - x.mean(dim=0, keepdim=True)
    r_c = recon_x - recon_x.mean(dim=0, keepdim=True)
    denom = float(x.shape[0] - 1)

    # Standard deviations for correlation normalization
    x_std = torch.sqrt((x_c.pow(2).sum(dim=0) / denom).clamp(min=eps))
    r_std = torch.sqrt((r_c.pow(2).sum(dim=0) / denom).clamp(min=eps))

    G = x.shape[1]
    block_size = max(1, int(block_size))

    def soft_threshold(u: torch.Tensor) -> torch.Tensor:
        if gamma == 1.0:
            return u
        return torch.sign(u) * torch.abs(u).pow(gamma)

    total_sq = x.new_tensor(0.0)
    for start in range(0, G, block_size):
        end = min(start + block_size, G)
        x_block = x_c[:, start:end]
        r_block = r_c[:, start:end]

        cov_x = (x_block.T @ x_c) / denom  # (b, G)
        cov_r = (r_block.T @ r_c) / denom

        x_std_block = x_std[start:end]
        r_std_block = r_std[start:end]

        x_corr = cov_x / (x_std_block[:, None] * x_std[None, :])
        r_corr = cov_r / (r_std_block[:, None] * r_std[None, :])

        diff = soft_threshold(r_corr) - soft_threshold(x_corr)
        total_sq = total_sq + diff.pow(2).sum()

    # Normalize by total elements to keep scale stable across G
    return total_sq / float(G * G)


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
                 coexpr_strength: float = 1.0,
                 coexpr_warmup_epochs: int = 50,
                 coexpr_gamma: float = 6.0,
                 coexpr_block_size: int = 512,
                 coexpr_max_genes: Optional[int] = None,
                 coexpr_auto_scale: bool = False,
                 coexpr_ema_decay: float = 0.99,
                 coexpr_scale_cap: float = 10.0):
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
        self.coexpr_warmup_epochs = coexpr_warmup_epochs
        self.coexpr_gamma = coexpr_gamma
        self.coexpr_block_size = coexpr_block_size
        self.coexpr_max_genes = coexpr_max_genes
        self.coexpr_auto_scale = coexpr_auto_scale
        self.coexpr_ema_decay = coexpr_ema_decay
        self.coexpr_scale_cap = coexpr_scale_cap
        self._ema_recon = None
        self._ema_coexpr = None

    def get_beta_for_epoch(self, epoch: int) -> float:
        """Return effective beta based on annealing schedule."""
        if self.kl_anneal_mode == "cyclical":
            # Cyclical annealing: repeat linear warmup over cycles
            if self.kl_cycle_length <= 0:
                return self.beta
            if self.kl_n_cycles is not None and self.kl_n_cycles > 0:
                cycle_idx = epoch // self.kl_cycle_length
                if cycle_idx >= self.kl_n_cycles:
                    return self.beta
            cycle_pos = epoch % self.kl_cycle_length
            ratio = min(cycle_pos / max(self.kl_cycle_length // 2, 1), 1.0)
            return self.beta * ratio
        else:
            # Linear warmup then constant
            if self.kl_warmup_epochs <= 0:
                return self.beta
            ratio = min(epoch / self.kl_warmup_epochs, 1.0)
            return self.beta * ratio

    def get_coexpr_strength_for_epoch(self, epoch: int) -> float:
        """Return effective coexpression weight based on warmup schedule."""
        if self.coexpr_strength <= 0:
            return 0.0
        if self.coexpr_warmup_epochs <= 0:
            return self.coexpr_strength
        ratio = min(epoch / self.coexpr_warmup_epochs, 1.0)
        return self.coexpr_strength * ratio

    def _update_ema(self, name: str, value: torch.Tensor) -> None:
        if name == "recon":
            ema = self._ema_recon
        else:
            ema = self._ema_coexpr
        val = float(value.detach().item())
        if ema is None:
            ema = val
        else:
            ema = self.coexpr_ema_decay * ema + (1.0 - self.coexpr_ema_decay) * val
        if name == "recon":
            self._ema_recon = ema
        else:
            self._ema_coexpr = ema

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
        coexpr_raw = None
        effective_coexpr = self.get_coexpr_strength_for_epoch(epoch)
        if effective_coexpr > 0:
            coexpr_raw = coexpression_loss(
                x,
                recon_x,
                block_size=self.coexpr_block_size,
                gamma=self.coexpr_gamma,
                max_genes=self.coexpr_max_genes,
            )
            if self.coexpr_auto_scale and is_train:
                self._update_ema("recon", recon_loss)
                self._update_ema("coexpr", coexpr_raw)
                if self._ema_coexpr is not None and self._ema_recon is not None:
                    scale = self._ema_recon / max(self._ema_coexpr, 1e-12)
                    scale = min(scale, self.coexpr_scale_cap)
                    effective_coexpr = effective_coexpr * scale
            coexpr_term = effective_coexpr * coexpr_raw

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
            if effective_coexpr > 0:
                storer.setdefault("coexpr_loss", []).append(coexpr_term.detach().item())
                storer.setdefault("coexpr_strength", []).append(effective_coexpr)
                if coexpr_raw is not None:
                    storer.setdefault("coexpr_raw", []).append(coexpr_raw.detach().item())
                if self.coexpr_auto_scale:
                    if self._ema_recon is not None:
                        storer.setdefault("coexpr_ema_recon", []).append(self._ema_recon)
                    if self._ema_coexpr is not None:
                        storer.setdefault("coexpr_ema_coexpr", []).append(self._ema_coexpr)
            storer.setdefault("loss", []).append(loss.item())
            # Per-dimension KL monitoring
            for dim_idx in range(kl_per_dim.shape[0]):
                storer.setdefault(f"kl_dim_{dim_idx}", []).append(
                    kl_per_dim[dim_idx].item()
                )

        return loss
