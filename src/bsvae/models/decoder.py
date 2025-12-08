"""
Decoder for StructuredFactorVAE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredDecoder(nn.Module):
    """
    Structured decoder for gene expression:
      - Each latent factor corresponds to a module.
      - Decoder weights are optionally masked by prior gene×module assignments.
      - Supports sparsity and Laplacian penalties.
      - Outputs mean and dispersion for Negative Binomial likelihood.

    Parameters
    ----------
    n_genes : int
        Number of genes (G).
    n_latent : int
        Number of latent factors/modules (K).
    mask : torch.Tensor or None
        Optional binary mask of shape (G, K) where 1 means gene↔factor allowed.
    init_sd : float
        Std for normal init of decoder weights.
    """
    def __init__(self, n_genes: int, n_latent: int,
                 mask: torch.Tensor = None,
                 init_sd: float = 0.02,
                 learn_var: bool = False):
        super().__init__()
        self.n_genes = n_genes
        self.n_latent = n_latent
        self.learn_var = learn_var

        # Weight matrix W: (G, K)
        self.W = nn.Parameter(torch.randn(n_genes, n_latent) * init_sd)
        self.bias = nn.Parameter(torch.zeros(n_genes))

        # Mask (non-trainable)
        if mask is not None:
            assert mask.shape == (n_genes, n_latent)
            self.register_buffer("mask", mask.float())
        else:
            self.mask = None

        # Gene-specific variance (optional)
        if self.learn_var:
            self.log_var = nn.Parameter(torch.zeros(n_genes))
        else:
            self.log_var = None

    def forward(self, z: torch.Tensor):
        """
        Parameters
        ----------
        z : torch.Tensor
            Latent codes, shape (batch, K).

        Returns
        -------
        recon_x : torch.Tensor
            Reconstructed gene expression (batch, G).
        log_var : torch.Tensor
            Gene-specific log-variance (G,).
        """
        # Apply mask if provided
        W_eff = self.W
        if self.mask is not None:
            W_eff = W_eff * self.mask  # elementwise mask

        recon_x = z @ W_eff.T + self.bias
        return recon_x, self.log_var

    # Regularizers
    def group_sparsity_penalty(self, l1_strength: float = 1e-3):
        """
        Group sparsity across genes per latent factor.
        """
        W_eff = self.W * self.mask if self.mask is not None else self.W
        return l1_strength * torch.sum(torch.abs(W_eff))

    def laplacian_penalty(self, L: torch.Tensor, lap_strength: float = 1e-3):
        """
        Laplacian smoothness: tr(W^T L W).
        """
        W_eff = self.W * self.mask if self.mask is not None else self.W
        if L.device != W_eff.device:
            L = L.to(W_eff.device)
        LW = torch.sparse.mm(L, W_eff) if L.is_sparse else torch.matmul(L, W_eff)
        penalty = torch.sum(W_eff * LW)  # trace approx
        return lap_strength * penalty
