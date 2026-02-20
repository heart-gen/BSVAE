"""
GaussianMixturePrior for GMMModuleVAE.

Implements a learnable isotropic Gaussian mixture:
    p(z) = Σ_k π_k · N(z; μ_k, σ²_k I)

Key methods
-----------
- expected_component_log_prob(mu_f, logvar_f)  → (B, K)
    Expected log-likelihood under q(z|x_f), accounting for encoder
    uncertainty via the trace term Σ_d exp(logvar_f_d).
- posterior_weights(mu_f, logvar_f)  → (B, K)
    Soft GMM assignments γ_{fk} = q(c=k | μ_f, logσ²_f).
- hard_assignments(mu_f, logvar_f)  → (B,)
    argmax of posterior_weights.
- separation_loss(alpha)
    σ-scaled inter-centroid separation penalty.
- balance_loss(gamma, eps, use_ema)
    γ-usage balance: KL(uniform || ρ_ema).
- kmeans_init_(mu_samples)
    Warm-start from K-means on encoder mean samples.

Collapse guards
---------------
- σ floor: σ_min (default 0.3) prevents delta-function collapse.
- γ-usage balance loss (λ_bal): penalises empty components.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMixturePrior(nn.Module):
    """
    Learnable isotropic Gaussian mixture prior.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    n_latent : int
        Dimensionality of the latent space D.
    sigma_min : float
        Floor on component standard deviation (σ_k ≥ sigma_min).
        Default: 0.3 (safe; users may lower explicitly if needed).
    ema_alpha : float
        EMA decay for ρ_ema in balance loss. Default: 0.99.
    """

    def __init__(
        self,
        n_components: int,
        n_latent: int,
        sigma_min: float = 0.3,
        ema_alpha: float = 0.99,
    ):
        super().__init__()
        self.K = n_components
        self.D = n_latent
        self.sigma_min = sigma_min
        self.ema_alpha = ema_alpha

        # Learnable parameters
        self.log_pi_unnorm = nn.Parameter(torch.zeros(n_components))
        self.mu_k = nn.Parameter(torch.randn(n_components, n_latent) * 0.5)
        # Initialise log σ² so σ ≈ 1.0 (well above sigma_min)
        self.log_sigma2_k = nn.Parameter(torch.zeros(n_components))

        # EMA buffer for γ-usage balance
        self.register_buffer(
            "rho_ema", torch.ones(n_components) / n_components
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def log_pi(self) -> torch.Tensor:
        """Log mixing proportions (normalised), shape (K,)."""
        return F.log_softmax(self.log_pi_unnorm, dim=0)

    @property
    def pi(self) -> torch.Tensor:
        """Mixing proportions, shape (K,)."""
        return torch.exp(self.log_pi)

    @property
    def sigma2_k(self) -> torch.Tensor:
        """Component variances with σ floor, shape (K,)."""
        log_min = 2.0 * math.log(self.sigma_min)
        return torch.exp(torch.clamp(self.log_sigma2_k, min=log_min))

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def expected_component_log_prob(
        self, mu_f: torch.Tensor, logvar_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Expected log N(z; μ_k, σ²_k I) under q(z | x_f) = N(μ_f, diag(exp(logvar_f))).

        E_{q}[log N(z; μ_k, σ²_k I)]
          = -D/2 log(2π σ²_k)
            - (‖μ_f - μ_k‖² + Σ_d exp(logvar_f_d)) / (2σ²_k)

        Parameters
        ----------
        mu_f : torch.Tensor, shape (B, D)
        logvar_f : torch.Tensor, shape (B, D)

        Returns
        -------
        log_prob : torch.Tensor, shape (B, K)
        """
        B, D = mu_f.shape
        K = self.K

        sigma2 = self.sigma2_k  # (K,)

        # ‖μ_f - μ_k‖²  →  (B, K)
        # mu_f: (B, D), mu_k: (K, D)
        diff = mu_f.unsqueeze(1) - self.mu_k.unsqueeze(0)  # (B, K, D)
        sq_dist = (diff ** 2).sum(dim=-1)                   # (B, K)

        # trace term: Σ_d exp(logvar_f_d)  →  (B,)
        trace = torch.exp(logvar_f).sum(dim=-1)             # (B,)

        # Expected log-likelihood: (B, K)
        log_prob = (
            -0.5 * D * torch.log(2.0 * math.pi * sigma2).unsqueeze(0)
            - (sq_dist + trace.unsqueeze(1)) / (2.0 * sigma2.unsqueeze(0))
        )
        return log_prob

    def posterior_weights(
        self, mu_f: torch.Tensor, logvar_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Soft GMM assignment γ_{fk} = q(c=k | μ_f, logσ²_f).

        γ_{fk} ∝ π_k · exp(E_{q}[log N(z; μ_k, σ²_k I)])

        Parameters
        ----------
        mu_f : torch.Tensor, shape (B, D)
        logvar_f : torch.Tensor, shape (B, D)

        Returns
        -------
        gamma : torch.Tensor, shape (B, K)  — rows sum to 1.
        """
        log_prob = self.expected_component_log_prob(mu_f, logvar_f)  # (B, K)
        log_gamma = self.log_pi.unsqueeze(0) + log_prob               # (B, K)
        return F.softmax(log_gamma, dim=-1)

    def hard_assignments(
        self, mu_f: torch.Tensor, logvar_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Hard GMM assignment argmax_k γ_{fk}, shape (B,).
        """
        return self.posterior_weights(mu_f, logvar_f).argmax(dim=-1)

    # ------------------------------------------------------------------
    # Regularization losses
    # ------------------------------------------------------------------

    def separation_loss(self, alpha: float = 2.0) -> torch.Tensor:
        """
        σ-scaled inter-centroid separation penalty.

        L_sep = Σ_{i<j} max(0, α·(σ_i + σ_j) − ‖μ_i − μ_j‖)²

        Parameters
        ----------
        alpha : float
            Margin multiplier. Default: 2.0.

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        K = self.K
        sigma = torch.sqrt(self.sigma2_k)  # (K,)

        # All pairwise centroid distances
        diff = self.mu_k.unsqueeze(0) - self.mu_k.unsqueeze(1)  # (K, K, D)
        dist = torch.norm(diff, dim=-1)                          # (K, K)

        # Required separation: α·(σ_i + σ_j)
        sigma_sum = sigma.unsqueeze(0) + sigma.unsqueeze(1)      # (K, K)
        margin = alpha * sigma_sum

        violation = torch.clamp(margin - dist, min=0.0) ** 2    # (K, K)

        # Sum upper triangle (i < j)
        mask = torch.triu(torch.ones(K, K, device=dist.device), diagonal=1)
        return (violation * mask).sum()

    def balance_loss(
        self,
        gamma: torch.Tensor,
        eps: float = 1e-8,
        update_ema: bool = True,
    ) -> torch.Tensor:
        """
        γ-usage balance loss: KL(uniform || ρ_ema + ε).

        L_bal = (1/K) Σ_k log(1 / (K · (ρ_ema_k + ε)))

        Parameters
        ----------
        gamma : torch.Tensor, shape (B, K)
            Soft assignments for the current batch.
        eps : float
            Stability term. Default: 1e-8.
        update_ema : bool
            Whether to update the EMA buffer (should be False at eval time).

        Returns
        -------
        loss : torch.Tensor, scalar
        """
        rho_batch = gamma.mean(dim=0)  # (K,)

        if update_ema and self.training:
            with torch.no_grad():
                self.rho_ema.mul_(self.ema_alpha).add_(
                    rho_batch.detach() * (1.0 - self.ema_alpha)
                )

        rho = self.rho_ema + eps
        # KL(uniform || rho): (1/K) Σ_k log(1 / (K * rho_k))
        #   = (1/K) Σ_k [-log(K) - log(rho_k)]
        K = float(self.K)
        loss = -(torch.log(rho) + math.log(K)).mean()
        return loss

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def kmeans_init_(self, mu_samples: torch.Tensor) -> None:
        """
        Warm-start centroids from K-means on encoder mean samples.

        Also initialises log_sigma2_k from the within-cluster variance
        and resets log_pi_unnorm to uniform.

        Parameters
        ----------
        mu_samples : torch.Tensor, shape (N, D)
            Accumulated encoder means from Phase 1 training.
        """
        from sklearn.cluster import MiniBatchKMeans

        K = self.K
        X = mu_samples.cpu().numpy()

        km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3)
        labels = km.fit_predict(X)
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)

        # Initialise centroids
        self.mu_k.copy_(centers.to(self.mu_k.device))

        # Estimate within-cluster variance per component
        log_min = 2.0 * math.log(self.sigma_min)
        for k in range(K):
            mask = labels == k
            if mask.sum() > 1:
                cluster_pts = torch.tensor(X[mask], dtype=torch.float32)
                var_k = (cluster_pts - centers[k]).pow(2).mean()
                log_var_k = torch.log(var_k + 1e-6).clamp(min=log_min)
            else:
                log_var_k = torch.tensor(0.0)  # σ² = 1

            self.log_sigma2_k[k] = log_var_k.to(self.log_sigma2_k.device)

        # Reset mixing weights to uniform
        nn.init.zeros_(self.log_pi_unnorm)
