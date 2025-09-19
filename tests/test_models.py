import pytest
import torch

from bsvae.models.encoder import StructuredEncoder
from bsvae.models.decoder import StructuredDecoder
from bsvae.models.vae import BaseVAE
from bsvae.models.structured import StructuredFactorVAE
from bsvae.models.losses import BaseLoss, gaussian_nll, kl_normal_loss


@pytest.mark.parametrize("batch,n_genes,n_latent", [(4, 50, 8), (2, 10, 3)])
def test_encoder_shapes(batch, n_genes, n_latent):
    x = torch.randn(batch, n_genes)
    enc = StructuredEncoder(n_genes, n_latent)
    mu, logvar = enc(x)
    assert mu.shape == (batch, n_latent)
    assert logvar.shape == (batch, n_latent)


@pytest.mark.parametrize("learn_var", [True, False])
def test_decoder_forward_and_penalties(learn_var):
    n_genes, n_latent, batch = 20, 5, 3
    z = torch.randn(batch, n_latent)
    mask = torch.ones(n_genes, n_latent)

    dec = StructuredDecoder(n_genes, n_latent, mask=mask, learn_var=learn_var)
    recon_x, log_var = dec(z)

    assert recon_x.shape == (batch, n_genes)
    if learn_var:
        assert log_var.shape == (n_genes,)
    else:
        assert log_var is None

    # Sparsity penalty
    l1 = dec.group_sparsity_penalty(1e-2)
    assert l1.item() >= 0

    # Laplacian penalty (use identity)
    L = torch.eye(n_genes)
    lap = dec.laplacian_penalty(L, 1e-2)
    assert lap.item() >= 0


def test_basevae_reparameterize_eval_vs_train():
    mu = torch.zeros(5, 3)
    logvar = torch.zeros(5, 3)
    vae = BaseVAE(3, 3)

    # Training mode → stochastic
    vae.train()
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)

    # Eval mode → deterministic
    vae.eval()
    z3 = vae.reparameterize(mu, logvar)
    assert torch.allclose(z3, mu)


def test_structured_factor_vae_forward_and_penalties():
    batch, n_genes, n_latent = 4, 30, 6
    x = torch.randn(batch, n_genes)
    L = torch.eye(n_genes)

    vae = StructuredFactorVAE(n_genes, n_latent, L=L)
    recon_x, mu, logvar, z, log_var = vae(x)

    assert recon_x.shape == (batch, n_genes)
    assert mu.shape == (batch, n_latent)
    assert logvar.shape == (batch, n_latent)
    assert z.shape == (batch, n_latent)

    # Penalties
    assert vae.group_sparsity_penalty().item() >= 0
    assert vae.laplacian_penalty(L).item() >= 0


def test_losses_end_to_end():
    batch, n_genes, n_latent = 5, 15, 4
    x = torch.randn(batch, n_genes)
    vae = StructuredFactorVAE(n_genes, n_latent)

    recon_x, mu, logvar, z, log_var = vae(x)

    loss_f = BaseLoss(beta=1.0, l1_strength=1e-2, lap_strength=1e-3)
    storer = {}
    loss = loss_f(x, recon_x, mu, logvar, vae, storer)

    assert torch.is_tensor(loss)
    assert loss.dim() == 0
    assert "loss" in storer
    assert storer["loss"]  # not empty


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_gaussian_nll_and_kl(reduction):
    x = torch.randn(3, 4)
    recon_x = torch.randn(3, 4)
    log_var = torch.zeros(4)

    nll = gaussian_nll(x, recon_x, log_var=log_var, reduction="mean")
    assert torch.is_tensor(nll)

    mu = torch.zeros(3, 2)
    logvar = torch.zeros(3, 2)
    kl = kl_normal_loss(mu, logvar, reduction="sum")
    assert kl.item() >= 0
