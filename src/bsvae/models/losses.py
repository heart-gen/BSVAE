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

def pairwise_corr_loss(x: torch.Tensor, recon_x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pairwise Pearson correlation preservation loss (reconstruction space).

    Computes the B×B Pearson correlation matrices of the input batch and the
    reconstructed batch, then returns their MSE.  Gradient flows through the
    decoder back to the encoder.

    Parameters
    ----------
    x       : (B, N) — input profiles (z-scored if normalize_input=True)
    recon_x : (B, N) — reconstructed profiles (same space as x)
    eps     : float  — denominator floor for L2-normalisation

    Returns
    -------
    loss : scalar tensor — MSE between the two B×B correlation matrices
    """
    x_c = x - x.mean(dim=1, keepdim=True)
    r_c = recon_x - recon_x.mean(dim=1, keepdim=True)
    x_n = F.normalize(x_c, p=2, dim=1, eps=eps)   # (B, N)
    r_n = F.normalize(r_c, p=2, dim=1, eps=eps)    # (B, N)
    C_x = x_n @ x_n.T   # (B, B)
    C_r = r_n @ r_n.T   # (B, B)
    return F.mse_loss(C_r, C_x.detach())


def latent_corr_loss(x: torch.Tensor, mu: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Latent-space correlation alignment loss.

    Aligns pairwise cosine similarities in encoder μ-space with Pearson
    correlations in input space.  Gradient flows directly through μ to the
    encoder weights, making it a more direct signal than the reconstruction-
    space version.

    For features i, j that co-vary in expression (r_{ij} ≈ 0.8), this loss
    pushes their latent means μ_i, μ_j to be similarly oriented — exactly
    the structure needed for module recovery.

    Parameters
    ----------
    x   : (B, N) — input profiles (z-scored if normalize_input=True)
    mu  : (B, D) — encoder means
    eps : float  — L2-normalisation floor

    Returns
    -------
    loss : scalar tensor — MSE between B×B input correlation and μ cosine-similarity matrices
    """
    # Input Pearson correlations (target, no gradient needed)
    x_c = x - x.mean(dim=1, keepdim=True)
    x_n = F.normalize(x_c, p=2, dim=1, eps=eps)
    C_x = x_n @ x_n.T                             # (B, B) — detached target

    # Latent cosine similarities (learned). Centre μ across the batch first so
    # the similarity is a correlation in latent space too, not an angle that
    # depends on a shared offset.
    mu_c = mu - mu.mean(dim=0, keepdim=True)
    mu_n = F.normalize(mu_c, p=2, dim=1, eps=eps)
    S_mu = mu_n @ mu_n.T                           # (B, B)

    return F.mse_loss(S_mu, C_x.detach())


def neighbour_contrastive_loss(
    x: torch.Tensor,
    mu: torch.Tensor,
    k_pos: int = 5,
    tau: float = 0.1,
    min_corr: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Size-invariant contrastive alignment of the latent with co-expression.

    For every feature in the batch its ``k_pos`` most correlated batch-mates
    (Pearson r ≥ ``min_corr``) are positives; an InfoNCE loss on the cosine
    similarity of (batch-centred) μ pulls each feature toward its positives
    and away from the rest of the batch.  Every feature contributes one term
    regardless of the size of its module, unlike the reconstruction loss,
    where a module's influence on the latent grows with its size.

    Parameters
    ----------
    x   : (B, N) input profiles (z-scored if normalize_input)
    mu  : (B, D) encoder means
    k_pos : positives per anchor
    tau : softmax temperature
    min_corr : minimum Pearson correlation for a positive
    """
    B = x.shape[0]
    if B < 3:
        return mu.new_zeros(1).squeeze()
    with torch.no_grad():
        x_c = x - x.mean(dim=1, keepdim=True)
        x_n = F.normalize(x_c, p=2, dim=1, eps=eps)
        C = x_n @ x_n.T
        C.fill_diagonal_(-2.0)
        k = min(int(k_pos), B - 1)
        top_val, top_idx = C.topk(k, dim=1)                    # (B, k)
        pos_mask = torch.zeros_like(C, dtype=torch.bool)
        pos_mask.scatter_(1, top_idx, top_val >= float(min_corr))
    if not pos_mask.any():
        return mu.new_zeros(1).squeeze()
    mu_c = mu - mu.mean(dim=0, keepdim=True)
    mu_n = F.normalize(mu_c, p=2, dim=1, eps=eps)
    logits = (mu_n @ mu_n.T) / float(tau)
    logits = logits.masked_fill(torch.eye(B, dtype=torch.bool, device=mu.device), -1e4)
    log_prob = F.log_softmax(logits, dim=1)
    n_pos = pos_mask.sum(dim=1)
    anchors = n_pos > 0
    per_anchor = -torch.where(pos_mask, log_prob, torch.zeros_like(log_prob)).sum(dim=1)[anchors] / n_pos[anchors]
    return per_anchor.mean()


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
        # True per-sample sum: clamp per-dim then sum over batch and dims
        kl_sum = torch.clamp(kl, min=free_bits).sum() if free_bits > 0 else kl.sum()
        return kl_sum, kl_per_dim
    raise ValueError(f"Unknown reduction '{reduction}'; expected 'mean' or 'sum'.")


def _recon_target(
    x: torch.Tensor,
    model: Optional[nn.Module],
    normalize_input: bool,
    masked_recon: bool,
):
    """
    Build the reconstruction target (and optional mask) for a batch.

    The model's ``normalize_input`` flag is the single source of truth when a
    model is given; otherwise the loss's own flag is used.  The mask is taken
    on the raw input and keeps every non-zero entry, so log-scale data with
    negative values is handled correctly.
    """
    should_normalize = getattr(model, "normalize_input", normalize_input)
    mask = (x != 0).float() if masked_recon else None
    x_target = x
    if should_normalize:
        mu_x = x.mean(dim=-1, keepdim=True)
        sigma_x = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        x_target = (x - mu_x) / sigma_x
    return x_target, mask


def _recon_loss(recon_x: torch.Tensor, x_target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        sq_err = (recon_x - x_target) ** 2
        return (sq_err * mask).sum() / mask.sum().clamp(min=1.0)
    return F.mse_loss(recon_x, x_target, reduction="mean")


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
    sigma2_k : (K,) or (K, D) — GMM component variances (with σ floor already applied)
    log_pi : (K,) — log mixing proportions
    free_bits : float — per (component, dimension) KL lower bound (nats).
        Applied after averaging over the batch: the batch-mean KL for each
        (k, d) pair is clamped to at least ``free_bits``, then this floor is
        broadcast back to individual samples.  This is stricter than standard
        per-dimension free-bits (K×D floors vs D floors) because each GMM
        component gets its own per-dimension floor.

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

    # sigma²_k broadcast: (1, K, 1) for isotropic, (1, K, D) for diagonal
    sigma2 = sigma2_k.unsqueeze(0) if sigma2_k.dim() == 2 else sigma2_k.unsqueeze(0).unsqueeze(-1)

    # Per-dim KL(N_f || N_k): (B, K, D)
    kl_dim = 0.5 * (
        (eps.unsqueeze(1) + diff_sq) / sigma2
        + torch.log(sigma2)
        - logvar.unsqueeze(1)
        - 1.0
    )                                                         # (B, K, D)

    # Apply free-bits per (component, dim): average over batch, clamp,
    # then use the clamped value as a floor for individual samples.
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

def _pair_term(
    mu_iso: torch.Tensor,
    x_iso: Optional[torch.Tensor],
    margin: float,
    corr_weight: bool,
) -> Optional[torch.Tensor]:
    """Weighted mean hinge distance over isoform pairs of one gene (or None)."""
    n = mu_iso.shape[0]
    diffs = mu_iso.unsqueeze(0) - mu_iso.unsqueeze(1)          # (n, n, D)
    dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)        # (n, n)
    if margin > 0.0:
        dists = torch.clamp(dists - margin, min=0.0)
    mask = torch.triu(torch.ones(n, n, device=mu_iso.device), diagonal=1)
    if corr_weight and x_iso is not None:
        with torch.no_grad():
            xc = x_iso - x_iso.mean(dim=1, keepdim=True)
            xn = F.normalize(xc, p=2, dim=1, eps=1e-6)
            w = torch.clamp(xn @ xn.T, min=0.0, max=1.0)          # (n, n)
        mask = mask * w
    denom = mask.sum()
    if denom <= 0:
        return None
    return (dists * mask).sum() / denom


def hierarchical_loss(
    mu: torch.Tensor,
    gene_groups: Dict[str, List[int]],
    feature_idx_in_batch: Optional[torch.Tensor] = None,
    idx_to_gene: Optional[Dict[int, str]] = None,
    x: Optional[torch.Tensor] = None,
    margin: float = 0.0,
    corr_weight: bool = False,
) -> torch.Tensor:
    """
    Mean pairwise hinge L2 in μ-space for isoforms of the same gene.

    With ``margin > 0`` only pairs farther apart than ``margin`` are pulled
    together, so isoforms that already share a component are left alone.
    With ``corr_weight=True`` (requires ``x``) each pair is weighted by the
    positive part of the Pearson correlation of the two input profiles, so
    isoforms whose expression diverges (isoform switching) are not forced
    into the same module.

    Parameters
    ----------
    mu : (B, D) — encoder means for the current batch
    gene_groups : dict mapping gene_id → list of dataset-level feature indices
    feature_idx_in_batch : (B,) int tensor mapping batch positions to dataset indices.
        If None, assumes the batch contains exactly the features in gene_groups.
    idx_to_gene : optional precomputed reverse mapping dataset_idx → gene_id.
        When provided alongside feature_idx_in_batch, enables an O(B) fast path
        (128 Python dict lookups) instead of O(G × n_iso) GPU tensor equality ops.
    x : (B, N) optional input profiles (used only when ``corr_weight``)
    margin : float — hinge margin in latent units (0 = plain attraction)
    corr_weight : bool — weight pairs by max(0, Pearson(x_i, x_j))

    Returns
    -------
    loss : scalar tensor
    """
    if feature_idx_in_batch is not None:
        losses = []

        def _collect_gene_positions(dataset_idx_to_gene: Dict[int, str]) -> None:
            # Collapse repeated samples of the same dataset feature by averaging
            # their encoder means, so the hierarchical term is order-invariant
            # and only compares distinct isoforms per gene.
            gene_to_batch_pos: Dict[str, Dict[int, List[int]]] = {}
            feat_idx_list = feature_idx_in_batch.tolist()
            for batch_pos, dataset_idx in enumerate(feat_idx_list):
                dataset_idx = int(dataset_idx)
                gid = dataset_idx_to_gene.get(dataset_idx)
                if gid is not None:
                    gene_to_batch_pos.setdefault(gid, {}).setdefault(dataset_idx, []).append(batch_pos)

            # Build all isoform pairs of all genes at once (one GPU pass).
            left, right, gene_of_pair = [], [], []
            iso_rows: List[List[int]] = []
            g = 0
            for positions_by_idx in gene_to_batch_pos.values():
                if len(positions_by_idx) < 2:
                    continue
                base = len(iso_rows)
                iso_rows.extend(list(positions_by_idx.values()))
                n = len(positions_by_idx)
                for a in range(n):
                    for b in range(a + 1, n):
                        left.append(base + a); right.append(base + b); gene_of_pair.append(g)
                g += 1
            if not left:
                return
            # Average repeated draws of the same feature (usually a single position).
            if all(len(r) == 1 for r in iso_rows):
                sel = torch.tensor([r[0] for r in iso_rows], device=mu.device)
                mu_iso = mu[sel]
                x_iso = x[sel] if (corr_weight and x is not None) else None
            else:
                mu_iso = torch.stack([mu[r].mean(dim=0) for r in iso_rows])
                x_iso = torch.stack([x[r].mean(dim=0) for r in iso_rows]) if (corr_weight and x is not None) else None
            li = torch.tensor(left, device=mu.device); ri = torch.tensor(right, device=mu.device)
            gi = torch.tensor(gene_of_pair, device=mu.device)
            d = torch.sqrt(((mu_iso[li] - mu_iso[ri]) ** 2).sum(dim=-1) + 1e-12)
            if margin > 0.0:
                d = torch.clamp(d - margin, min=0.0)
            w = torch.ones_like(d)
            if corr_weight and x_iso is not None:
                with torch.no_grad():
                    xc = x_iso - x_iso.mean(dim=1, keepdim=True)
                    xn = F.normalize(xc, p=2, dim=1, eps=1e-6)
                    w = torch.clamp((xn[li] * xn[ri]).sum(dim=1), min=0.0, max=1.0)
            # Weighted mean per gene, then mean over genes with any weight.
            num = torch.zeros(g, device=mu.device).index_add_(0, gi, d * w)
            den = torch.zeros(g, device=mu.device).index_add_(0, gi, w)
            ok = den > 0
            if ok.any():
                losses.append((num[ok] / den[ok]).mean())

        if idx_to_gene is not None:
            # Fast path: O(B) Python dict lookups — no GPU tensor ops per gene
            _collect_gene_positions(idx_to_gene)
        else:
            # Fallback: build reverse index on the fly (O(G × n_iso) Python, no GPU ops)
            _idx_to_gene: Dict[int, str] = {}
            for gid, idxs in gene_groups.items():
                for idx in idxs:
                    _idx_to_gene[idx] = gid

            _collect_gene_positions(_idx_to_gene)
    else:
        losses = []
        for gene_id, feat_indices in gene_groups.items():
            if len(feat_indices) < 2:
                continue
            B = mu.shape[0]
            if any(idx >= B or idx < 0 for idx in feat_indices):
                raise IndexError(
                    f"hierarchical_loss: feat_indices {feat_indices} out of range for "
                    f"batch size {B}. Pass feature_idx_in_batch to use dataset-level indices."
                )
            mu_iso = mu[feat_indices]   # (n, D)
            x_iso = x[feat_indices] if (corr_weight and x is not None) else None
            term = _pair_term(mu_iso, x_iso, margin, corr_weight)
            if term is not None:
                losses.append(term)

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
        Per-dimension KL lower bound (nats). Default: 0.0 (disabled).
        Non-zero values can prevent individual KL dimensions from collapsing
        to zero, but empirically hurt GMM cluster recovery by creating a floor
        that keeps γ uniform. Leave at 0.0 for normal use.
    sep_strength : float
        λ_sep — weight for separation loss. Default: 0.1.
    sep_alpha : float
        α — margin multiplier for σ-scaled separation. Default: 2.0.
    contrastive_strength : float
        λ_nce — weight of the neighbour-contrastive latent alignment
        (``neighbour_contrastive_loss``). Size-invariant: every feature pulls
        toward its most correlated batch-mates. Default: 0.0 (disabled).
    contrastive_k, contrastive_tau, contrastive_min_corr : positives per
        anchor (5), temperature (0.1) and minimum correlation (0.3).
    kl_per_element : bool
        If True (default), scale the KL term by 1/N (N = profile length) so
        that, like the per-element MSE reconstruction, it is expressed per
        observed value and ``beta=1`` is a true ELBO weight.  Set False for
        the legacy behaviour where the KL was summed over latent dimensions
        while the reconstruction was averaged over N, which made the
        effective KL weight grow with the number of samples.
    bal_strength : float
        λ_bal — weight for γ-usage balance loss. Default: 0.1.
        With the default ``bal_type="entropy"`` this only discourages dead
        components; ``bal_type="uniform"`` additionally pushes toward equal
        module sizes, which hurts recovery of imbalanced modules.
        Lower values risk dead components; the collapse-revival mechanism
        handles this, but very small values (< 0.01) may cause cascading
        collapse.
    pi_entropy_strength : float
        λ_pi — weight for KL(uniform || π) penalty. Default: 0.0 (disabled).
    bal_ema_blend : float
        Blend factor for EMA vs batch usage in balance loss. Default: 0.5.
    hier_strength : float
        λ_hier — weight for hierarchical loss. Default: 0.0.
    hier_margin : float
        Hinge margin for the hierarchical loss (latent units). Default: 0.0.
    hier_corr_weight : bool
        Weight isoform pairs by max(0, Pearson(x_i, x_j)). Default: False.
    normalize_input : bool
        If True, z-score x before computing the reconstruction loss so the
        target space matches GMMModuleVAE(normalize_input=True) output.
        Must be set consistently with the model flag. Default: False.
    masked_recon : bool
        If True, compute reconstruction loss only on originally non-zero
        entries of x (mask = x > 0). Reduces zero-inflation dominance in
        sparse count data. Applied to raw x before any normalization.
        Default: False.
    corr_strength : float
        λ_corr — weight for reconstruction-space pairwise correlation loss.
        Penalises MSE between the B×B Pearson correlation matrices of inputs
        and reconstructions. Gradient flows through the decoder to the encoder.
        Recommended range: 0.05–0.2 for NB count data. Default: 0.0 (disabled).
    latent_corr_strength : float
        λ_lcorr — weight for latent-space correlation alignment loss.
        Aligns cosine similarities of encoder μ with Pearson correlations in
        input space.  Gradient flows directly to the encoder, bypassing the
        decoder.  More direct than corr_strength; recommended range: 0.1–1.0
        for NB count data with normalize_input=True. Default: 0.0 (disabled).
    """

    def __init__(
        self,
        beta: float = 1.0,
        kl_warmup_epochs: int = 0,
        kl_anneal_mode: str = "linear",
        kl_cycle_length: int = 50,
        kl_n_cycles: int = 4,
        free_bits: float = 0.0,
        sep_strength: float = 0.1,
        sep_alpha: float = 2.0,
        bal_strength: float = 0.1,
        bal_type: str = "entropy",
        bal_scale_gmm: bool = False,
        pi_entropy_strength: float = 0.0,
        bal_ema_blend: float = 0.5,
        hier_strength: float = 0.0,
        normalize_input: bool = False,
        masked_recon: bool = False,
        corr_strength: float = 0.0,
        latent_corr_strength: float = 0.0,
        kl_per_element: bool = True,
        hier_margin: float = 0.0,
        hier_corr_weight: bool = False,
        contrastive_strength: float = 0.0,
        contrastive_k: int = 5,
        contrastive_tau: float = 0.1,
        contrastive_min_corr: float = 0.3,
    ):
        super().__init__()
        self.beta = beta
        self.contrastive_strength = contrastive_strength
        self.contrastive_k = contrastive_k
        self.contrastive_tau = contrastive_tau
        self.contrastive_min_corr = contrastive_min_corr
        self.kl_per_element = kl_per_element
        self.hier_margin = hier_margin
        self.hier_corr_weight = hier_corr_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_mode = kl_anneal_mode
        self.kl_cycle_length = kl_cycle_length
        self.kl_n_cycles = kl_n_cycles
        self.free_bits = free_bits
        self.sep_strength = sep_strength
        self.sep_alpha = sep_alpha
        self.bal_strength = bal_strength
        self.bal_type = bal_type
        self.bal_scale_gmm = bal_scale_gmm
        self.pi_entropy_strength = pi_entropy_strength
        self.bal_ema_blend = bal_ema_blend
        self.hier_strength = hier_strength
        self.normalize_input = normalize_input
        self.masked_recon = masked_recon
        self.corr_strength = corr_strength
        self.latent_corr_strength = latent_corr_strength

    def get_beta_for_epoch(self, epoch: Optional[int]) -> float:
        """Effective β for a phase-relative epoch; ``None`` means full β (evaluation)."""
        if epoch is None:
            return self.beta
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
        idx_to_gene: Optional[Dict[int, str]] = None,
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
        epoch : int or None — epoch relative to the start of the GMM phase
            (None = full β, used for evaluation)
        gene_groups : optional dict for hierarchical loss
        feature_idx : optional (B,) int tensor for hierarchical loss
        gmm_weight : float — ramp weight for GMM terms (0→1 during transition)
        idx_to_gene : optional precomputed reverse mapping dataset_idx → gene_id
            for O(B) fast path in hierarchical_loss

        Returns
        -------
        loss : scalar tensor
        """
        # --- Reconstruction ---
        x_target, mask = _recon_target(x, model, self.normalize_input, self.masked_recon)
        recon_loss = _recon_loss(recon_x, x_target, mask)

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
        if self.kl_per_element:
            kl_loss = kl_loss / float(x.shape[1])
        effective_beta = self.get_beta_for_epoch(epoch)

        # --- Separation ---
        sep_loss = prior.separation_loss(alpha=self.sep_alpha)

        # --- Balance ---
        bal_loss = prior.balance_loss(
            gamma,
            update_ema=model.training,
            blend_alpha=self.bal_ema_blend,
            bal_type=self.bal_type,
        )

        # --- Pi entropy (encourage non-collapsed mixing weights) ---
        pi_entropy = -(prior.pi * prior.log_pi).sum()
        pi_entropy_loss = (math.log(float(prior.K)) - pi_entropy)

        # --- Reconstruction-space correlation preservation ---
        corr_loss = x.new_zeros(1).squeeze()
        if self.corr_strength > 0.0:
            corr_loss = pairwise_corr_loss(x_target, recon_x)

        # --- Latent-space correlation alignment ---
        latent_corr = x.new_zeros(1).squeeze()
        if self.latent_corr_strength > 0.0:
            latent_corr = latent_corr_loss(x_target, mu)

        # --- Neighbour contrastive (size-invariant) ---
        nce_loss = x.new_zeros(1).squeeze()
        if self.contrastive_strength > 0.0:
            nce_loss = neighbour_contrastive_loss(
                x_target, mu, k_pos=self.contrastive_k, tau=self.contrastive_tau,
                min_corr=self.contrastive_min_corr,
            )

        # --- Hierarchical ---
        hier_loss = x.new_zeros(1).squeeze()
        if self.hier_strength > 0.0 and gene_groups:
            hier_loss = hierarchical_loss(
                mu, gene_groups, feature_idx, idx_to_gene=idx_to_gene,
                x=x_target, margin=self.hier_margin, corr_weight=self.hier_corr_weight,
            )

        # --- Total ---
        # bal_scale_gmm=True ramps the balance loss with gmm_weight so it does not
        # oppose GMM learning at full strength while the KL signal is still building.
        bal_scale = gmm_weight if self.bal_scale_gmm else 1.0
        loss = (
            recon_loss
            + effective_beta * gmm_weight * kl_loss
            + self.sep_strength * gmm_weight * sep_loss
            + self.bal_strength * bal_scale * bal_loss
            + self.pi_entropy_strength * pi_entropy_loss
            + self.corr_strength * corr_loss
            + self.latent_corr_strength * latent_corr
            + self.contrastive_strength * nce_loss
            + self.hier_strength * hier_loss
        )

        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("effective_beta", []).append(effective_beta)
            storer.setdefault("sep_loss", []).append(sep_loss.item())
            storer.setdefault("bal_loss", []).append(bal_loss.item())
            storer.setdefault("pi_entropy", []).append(pi_entropy.item())
            storer.setdefault("pi_entropy_loss", []).append(pi_entropy_loss.item())
            storer.setdefault("corr_loss", []).append(corr_loss.item())
            storer.setdefault("latent_corr_loss", []).append(latent_corr.item())
            storer.setdefault("contrastive_loss", []).append(nce_loss.item())
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
        free_bits: float = 0.0,
        normalize_input: bool = False,
        masked_recon: bool = False,
        kl_per_element: bool = True,
    ):
        super().__init__()
        self.beta = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_anneal_mode = kl_anneal_mode
        self.kl_cycle_length = kl_cycle_length
        self.free_bits = free_bits
        self.normalize_input = normalize_input
        self.masked_recon = masked_recon
        self.kl_per_element = kl_per_element

    def get_beta_for_epoch(self, epoch: Optional[int]) -> float:
        if epoch is None:
            return self.beta
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
        epoch: Optional[int] = 0,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        x_target, mask = _recon_target(x, model, self.normalize_input, self.masked_recon)
        recon_loss = _recon_loss(recon_x, x_target, mask)
        kl_loss, _ = kl_normal_loss_with_free_bits(
            mu, logvar, free_bits=self.free_bits, reduction="mean"
        )
        if self.kl_per_element:
            kl_loss = kl_loss / float(x.shape[1])
        effective_beta = self.get_beta_for_epoch(epoch)
        loss = recon_loss + effective_beta * kl_loss

        if storer is not None:
            storer.setdefault("recon_loss", []).append(recon_loss.item())
            storer.setdefault("kl_loss", []).append(kl_loss.item())
            storer.setdefault("effective_beta", []).append(effective_beta)
            storer.setdefault("loss", []).append(loss.item())

        return loss
