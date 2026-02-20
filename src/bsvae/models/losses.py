"""
Loss functions for GMMModuleVAE.

GMMVAELoss
----------
Total loss:
  L = L_recon + β(t)·KL_VaDE + λ_sep·L_sep + λ_bal·L_balance + λ_hier·L_hier

VaDE analytic KL (exact, no SGVB noise):
  KL_VaDE = Σ_k γ_k · KL(N(μ_f, diag(exp(logvar_f))) || N(μ_k, σ²_k I))
           + KL(Cat(γ) || Cat(π))

KL annealing is ported unchanged from the original BaseLoss:
  - "linear": β rises linearly from 0 over kl_warmup_epochs
  - "cyclical": repeated linear warmup each kl_cycle_length epochs

Also exports standalone helper functions for backward compatibility
and unit testing.
"""

import math
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone helpers (kept for backward compatibility / unit tests)
# ---------------------------------------------------------------------------

def gaussian_nll(x, recon_x, log_var=None, reduction="mean"):
    """Gaussian negative log-likelihood per feature."""
    if log_var is None:
        return F.mse_loss(recon_x, x, reduction=reduction)
    var = torch.exp(log_var)
    nll = 0.5 * (torch.log(2.0 * math.pi * var) + (x - recon_x) ** 2 / var)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    return nll


def kl_normal_loss(mu, logvar, reduction="sum"):
    """KL(q(z|x) || N(0,I)) — standard VAE KL."""
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    return kl


def kl_normal_loss_with_free_bits(mu, logvar, free_bits=0.0, reduction="mean"):
    """
    KL(N(μ,σ²) || N(0,I)) with per-dimension free-bits.

    Returns
    -------
    kl_total : torch.Tensor, scalar
    kl_per_dim : torch.Tensor, shape (K,)
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # (B, K)
    kl_per_dim = kl.mean(dim=0)                              # (K,)

    if free_bits > 0:
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
    else:
        kl_per_dim_clamped = kl_per_dim

    kl_total = kl_per_dim_clamped.sum()

    if reduction == "mean":
        return kl_total, kl_per_dim
    elif reduction == "sum":
        return kl_total * mu.shape[0], kl_per_dim
    return kl, kl_per_dim


# ---------------------------------------------------------------------------
# VaDE analytic KL
# ---------------------------------------------------------------------------

def kl_vade(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    gamma: torch.Tensor,
    mu_k: torch.Tensor,
    sigma2_k: torch.Tensor,
    log_pi: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """
    VaDE analytic KL divergence.

    KL_VaDE = Σ_k γ_k · KL(N(μ_f, diag(exp(logvar_f))) || N(μ_k, σ²_k I))
             + KL(Cat(γ) || Cat(π))

    where:
      KL(N_f || N_k) = ½ Σ_d [(exp(logvar_f_d) + (μ_f_d - μ_k_d)²) / σ²_k
                               + log σ²_k - logvar_f_d - 1]

    Parameters
    ----------
    mu : (B, D) — encoder means
    logvar : (B, D) — encoder log-variances
    gamma : (B, K) — soft GMM assignments
    mu_k : (K, D) — GMM centroids
    sigma2_k : (K,) — GMM component variances (with σ floor already applied)
    log_pi : (K,) — log mixing proportions
    free_bits : float — per-dimension KL lower bound (nats)

    Returns
    -------
    kl : torch.Tensor, scalar — mean KL over the batch
    """
    B, D = mu.shape
    K = mu_k.shape[0]
    eps = logvar.exp()                                        # (B, D)

    # ----- Gaussian KL: (B, K) -----
    # KL(N_f || N_k) per (sample, component)
    # diff² = (μ_f - μ_k)²: (B, K, D)
    diff = mu.unsqueeze(1) - mu_k.unsqueeze(0)               # (B, K, D)
    diff_sq = diff ** 2                                       # (B, K, D)

    # sigma²_k broadcast: (1, K, 1)
    sigma2 = sigma2_k.unsqueeze(0).unsqueeze(-1)              # (1, K, 1)

    # Per-dim KL(N_f || N_k): (B, K, D)
    kl_dim = 0.5 * (
        (eps.unsqueeze(1) + diff_sq) / sigma2
        + torch.log(sigma2)
        - logvar.unsqueeze(1)
        - 1.0
    )                                                         # (B, K, D)

    # Apply free-bits: average over batch first, clamp, then restore
    if free_bits > 0:
        # kl_dim_mean: (K, D) — mean over batch per (component, dim)
        kl_dim_mean = kl_dim.mean(dim=0)                     # (K, D)
        kl_dim_mean = torch.clamp(kl_dim_mean, min=free_bits)
        # Broadcast clamped floor back to (B, K, D)
        # Use the element-wise max of actual and the clamped mean floor
        floor = kl_dim_mean.unsqueeze(0).expand_as(kl_dim)
        kl_dim = torch.maximum(kl_dim, floor)

    # Sum over D → (B, K)
    kl_gauss = kl_dim.sum(dim=-1)                            # (B, K)

    # Weight by γ and sum over K → (B,)
    kl_gauss_weighted = (gamma * kl_gauss).sum(dim=-1)       # (B,)

    # ----- Categorical KL: (B,) -----
    # KL(Cat(γ) || Cat(π)) = Σ_k γ_k (log γ_k - log π_k)
    kl_cat = (gamma * (torch.log(gamma.clamp(min=1e-8)) - log_pi.unsqueeze(0))).sum(dim=-1)

    # Total KL, mean over batch
    kl_total = (kl_gauss_weighted + kl_cat).mean()
    return kl_total


# ---------------------------------------------------------------------------
# Hierarchical loss
# ---------------------------------------------------------------------------

def hierarchical_loss(
    mu: torch.Tensor,
    gene_groups: Dict[str, List[int]],
    feature_idx_in_batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean pairwise L2 in μ-space for isoforms of the same gene.

    Parameters
    ----------
    mu : (B, D) — encoder means for the current batch
    gene_groups : dict mapping gene_id → list of dataset-level feature indices
    feature_idx_in_batch : (B,) int tensor mapping batch positions to dataset indices.
        If None, assumes the batch contains exactly the features in gene_groups.

    Returns
    -------
    loss : scalar tensor
    """
    losses = []
    for gene_id, feat_indices in gene_groups.items():
        if feature_idx_in_batch is not None:
            # Find which batch positions correspond to these feature indices
            batch_pos = [
                (feature_idx_in_batch == idx).nonzero(as_tuple=True)[0]
                for idx in feat_indices
            ]
            # Filter to positions actually present in the batch
            present = [p for p in batch_pos if p.numel() > 0]
            if len(present) < 2:
                continue
            mu_iso = torch.cat([mu[p[0:1]] for p in present], dim=0)  # (n, D)
        else:
            if len(feat_indices) < 2:
                continue
            mu_iso = mu[feat_indices]   # (n, D)

        # Mean pairwise L2 distance
        n = mu_iso.shape[0]
        diffs = mu_iso.unsqueeze(0) - mu_iso.unsqueeze(1)  # (n, n, D)
        dists = diffs.norm(dim=-1)                          # (n, n)
        mask = torch.triu(torch.ones(n, n, device=mu.device), diagonal=1)
        losses.append((dists * mask).sum() / mask.sum())

    if not losses:
        return mu.new_zeros(1).squeeze()
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Main loss module
# ---------------------------------------------------------------------------

class GMMVAELoss(nn.Module):
    """
    Total loss for GMMModuleVAE.

    L = L_recon + β(t)·KL_VaDE + λ_sep·L_sep + λ_bal·L_balance + λ_hier·L_hier

    Parameters
    ----------
    beta : float
        KL weight at full scale. Default: 1.0.
    kl_warmup_epochs : int
        Epochs to linearly ramp β from 0 to target. Default: 0 (no warmup).
    kl_anneal_mode : str
        "linear" or "cyclical". Default: "linear".
    kl_cycle_length : int
        Cycle length for cyclical annealing. Default: 50.
    kl_n_cycles : int
        Number of cycles (informational). Default: 4.
    free_bits : float
        Per-dimension KL lower bound (nats). Default: 0.5.
    sep_strength : float
        λ_sep — weight for separation loss. Default: 0.1.
    sep_alpha : float
        α — margin multiplier for σ-scaled separation. Default: 2.0.
    bal_strength : float
        λ_bal — weight for γ-usage balance loss. Default: 0.01.
    hier_strength : float
        λ_hier — weight for hierarchical loss. Default: 0.0.
    """

    def __init__(
        self,
        beta: float = 1.0,
        kl_warmup_epochs: int = 0,
        kl_anneal_mode: str = "linear",
        kl_cycle_length: int = 50,
        kl_n_cycles: int = 4,
        free_bits: float = 0.5,
        sep_strength: float = 0.1,
        sep_alpha: float = 2.0,
        bal_strength: float = 0.01,
        hier_strength: float = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_mode = kl_anneal_mode
        self.kl_cycle_length = kl_cycle_length
        self.kl_n_cycles = kl_n_cycles
        self.free_bits = free_bits
        self.sep_strength = sep_strength
        self.sep_alpha = sep_alpha
        self.bal_strength = bal_strength
        self.hier_strength = hier_strength

    def get_beta_for_epoch(self, epoch: int) -> float:
        """Compute effective β from annealing schedule (ported from BaseLoss)."""
        if self.kl_anneal_mode == "cyclical":
            cycle_pos = epoch % self.kl_cycle_length
            ratio = min(cycle_pos / max(self.kl_cycle_length // 2, 1), 1.0)
            return self.beta * ratio
        else:
            if self.kl_warmup_epochs <= 0:
                return self.beta
            ratio = min(epoch / max(self.kl_warmup_epochs, 1), 1.0)
            return self.beta * ratio

    def forward(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        gamma: torch.Tensor,
        model: nn.Module,
        storer: Optional[Dict[str, list]] = None,
        epoch: int = 0,
        gene_groups: Optional[Dict] = None,
        feature_idx: Optional[torch.Tensor] = None,
        gmm_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute GMMModuleVAE total loss.

        Parameters
        ----------
        x : (B, N) — input feature profiles
        recon_x : (B, N) — reconstructed profiles
        mu : (B, D) — encoder means
        logvar : (B, D) — encoder log-variances
        gamma : (B, K) — soft GMM assignments
        model : GMMModuleVAE instance
        storer : optional loss log dict
        epoch : int — current training epoch
        gene_groups : optional dict for hierarchical loss
        feature_idx : optional (B,) int tensor for hierarchical loss
        gmm_weight : float — ramp weight for GMM terms (0→1 during transition)

        Returns
        -------
        loss : scalar tensor
        """
        # --- Reconstruction ---
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # --- KL (VaDE) ---
        prior = model.gmm_prior
        kl_loss = kl_vade(
            mu=mu,
            logvar=logvar,
            gamma=gamma,
            mu_k=prior.mu_k,
            sigma2_k=prior.sigma2_k,
            log_pi=prior.log_pi,
            free_bits=self.free_bits,
        )
        effective_beta = self.get_beta_for_epoch(epoch)

        # --- Separation ---
        sep_loss = prior.separation_loss(alpha=self.sep_alpha)

        # --- Balance ---
        bal_loss = prior.balance_loss(gamma, update_ema=model.training)

        # --- Hierarchical ---
        hier_loss = x.new_zeros(1).squeeze()
        if self.hier_strength > 0.0 and gene_groups:
            hier_loss = hierarchical_loss(mu, gene_groups, feature_idx)

        # --- Total ---
        loss = (
            recon_loss
            + effective_beta * gmm_weight * kl_loss
            + self.sep_strength * gmm_weight * sep_loss
            + self.bal_strength * gmm_weight * bal_loss
            + self.hier_strength * hier_loss
        )

        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("effective_beta", []).append(effective_beta)
            storer.setdefault("sep_loss", []).append(sep_loss.item())
            storer.setdefault("bal_loss", []).append(bal_loss.item())
            storer.setdefault("hier_loss", []).append(hier_loss.item())
            storer.setdefault("gmm_weight", []).append(gmm_weight)
            storer.setdefault("loss", []).append(loss.item())

        return loss


# ---------------------------------------------------------------------------
# Phase-1 warmup loss (standard N(0,I) KL)
# ---------------------------------------------------------------------------

class WarmupLoss(nn.Module):
    """
    Standard N(0,I) VAE loss for Phase 1 warm-start.

    Uses free-bits and the same annealing schedule as GMMVAELoss so
    that the encoder learns a reasonable μ before GMM initialisation.
    """

    def __init__(
        self,
        beta: float = 1.0,
        kl_warmup_epochs: int = 20,
        kl_anneal_mode: str = "linear",
        kl_cycle_length: int = 50,
        free_bits: float = 0.5,
    ):
        super().__init__()
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_mode = kl_anneal_mode
        self.kl_cycle_length = kl_cycle_length
        self.free_bits = free_bits

    def get_beta_for_epoch(self, epoch: int) -> float:
        if self.kl_anneal_mode == "cyclical":
            cycle_pos = epoch % self.kl_cycle_length
            ratio = min(cycle_pos / max(self.kl_cycle_length // 2, 1), 1.0)
            return self.beta * ratio
        if self.kl_warmup_epochs <= 0:
            return self.beta
        ratio = min(epoch / max(self.kl_warmup_epochs, 1), 1.0)
        return self.beta * ratio

    def forward(
        self,
        x: torch.Tensor,
        recon_x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        storer: Optional[Dict[str, list]] = None,
        epoch: int = 0,
    ) -> torch.Tensor:
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kl_loss, _ = kl_normal_loss_with_free_bits(
            mu, logvar, free_bits=self.free_bits, reduction="mean"
        )
        effective_beta = self.get_beta_for_epoch(epoch)
        loss = recon_loss + effective_beta * kl_loss

        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("effective_beta", []).append(effective_beta)
            storer.setdefault("loss", []).append(loss.item())

        return loss
