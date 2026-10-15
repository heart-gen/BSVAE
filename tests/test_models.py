"""Unit tests for GMMModuleVAE model components."""

import math
import pytest
import torch
import numpy as np

from bsvae.models.vae import BaseVAE
from bsvae.models.encoder import FeatureEncoder
from bsvae.models.decoder import FeatureDecoder
from bsvae.models.gmm_prior import GaussianMixturePrior
from bsvae.models.gmvae import GMMModuleVAE
from bsvae.models.losses import (
    GMMVAELoss,
    WarmupLoss,
    kl_vade,
    gaussian_nll,
    kl_normal_loss,
    kl_normal_loss_with_free_bits,
)


# ---------------------------------------------------------------------------
# BaseVAE
# ---------------------------------------------------------------------------

def test_basevae_reparameterize_train_stochastic():
    vae = BaseVAE(10, 4)
    vae.train()
    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)
    z1 = vae.reparameterize(mu, logvar)
    z2 = vae.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)


def test_basevae_reparameterize_eval_deterministic():
    vae = BaseVAE(10, 4)
    vae.eval()
    mu = torch.randn(5, 4)
    logvar = torch.zeros(5, 4)
    z = vae.reparameterize(mu, logvar)
    assert torch.allclose(z, mu)


# ---------------------------------------------------------------------------
# FeatureEncoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch,n_feat,n_lat", [(4, 50, 8), (2, 10, 3)])
def test_feature_encoder_shapes(batch, n_feat, n_lat):
    enc = FeatureEncoder(n_feat, n_lat, hidden_dims=[32, 16])
    x = torch.randn(batch, n_feat)
    mu, logvar = enc(x)
    assert mu.shape == (batch, n_lat)
    assert logvar.shape == (batch, n_lat)


def test_feature_encoder_no_batch_norm():
    enc = FeatureEncoder(20, 4, hidden_dims=[16], use_batch_norm=False)
    x = torch.randn(3, 20)
    mu, logvar = enc(x)
    assert mu.shape == (3, 4)


# ---------------------------------------------------------------------------
# FeatureDecoder
# ---------------------------------------------------------------------------

def test_feature_decoder_shapes():
    dec = FeatureDecoder(n_features=30, n_latent=5)
    z = torch.randn(4, 5)
    recon = dec(z)
    assert recon.shape == (4, 30)


def test_feature_decoder_linear():
    dec = FeatureDecoder(n_features=10, n_latent=3, init_sd=1.0)
    z = torch.zeros(2, 3)
    recon = dec(z)
    # Output should equal bias when z=0
    assert torch.allclose(recon, dec.bias.unsqueeze(0).expand(2, -1))


# ---------------------------------------------------------------------------
# GaussianMixturePrior
# ---------------------------------------------------------------------------

def test_gmm_prior_shapes():
    K, D, B = 5, 8, 4
    prior = GaussianMixturePrior(n_components=K, n_latent=D)
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)

    gamma = prior.posterior_weights(mu, logvar)
    assert gamma.shape == (B, K)
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_gmm_prior_hard_assignments():
    K, D, B = 4, 6, 3
    prior = GaussianMixturePrior(n_components=K, n_latent=D)
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)
    hard = prior.hard_assignments(mu, logvar)
    assert hard.shape == (B,)
    assert (hard >= 0).all() and (hard < K).all()


def test_gmm_prior_sigma_floor():
    prior = GaussianMixturePrior(n_components=3, n_latent=4, sigma_min=0.3)
    # Push the raw parameter far below the floor
    prior.sigma2_raw.data.fill_(-100.0)
    sigma2 = prior.sigma2_k
    assert (sigma2 >= 0.3 ** 2 - 1e-6).all()
    assert torch.allclose(sigma2, torch.full_like(sigma2, 0.3 ** 2), atol=1e-6)


def test_gmm_prior_sigma_floor_keeps_gradient():
    """Softplus floor: gradient w.r.t. the raw parameter is non-zero at the floor."""
    prior = GaussianMixturePrior(n_components=3, n_latent=4, sigma_min=0.3)
    prior.sigma2_raw.data.fill_(-3.0)
    prior.sigma2_k.sum().backward()
    assert (prior.sigma2_raw.grad.abs() > 0).all()


def test_gmm_prior_set_log_sigma2_roundtrip():
    prior = GaussianMixturePrior(n_components=2, n_latent=4, sigma_min=0.3)
    prior.set_log_sigma2_(0, 0.7)
    prior.set_log_sigma2_(1, -5.0)  # below floor → clamped to floor
    assert abs(prior.log_sigma2_k[0].item() - 0.7) < 1e-4
    assert abs(prior.sigma2_k[1].item() - 0.3 ** 2) < 1e-5


def test_gmm_prior_loads_legacy_state_dict():
    prior = GaussianMixturePrior(n_components=2, n_latent=4, sigma_min=0.3)
    legacy = {k: v for k, v in prior.state_dict().items() if k != "sigma2_raw"}
    legacy["log_sigma2_k"] = torch.tensor([0.5, -1.0])
    prior.load_state_dict(legacy)
    assert abs(prior.log_sigma2_k[0].item() - 0.5) < 1e-4
    assert abs(prior.log_sigma2_k[1].item() - (-1.0)) < 1e-4


def test_gmm_prior_log_pi_normalised():
    prior = GaussianMixturePrior(n_components=5, n_latent=3)
    pi = prior.pi
    assert torch.allclose(pi.sum(), torch.tensor(1.0), atol=1e-5)


def test_gmm_prior_separation_loss_nonneg():
    prior = GaussianMixturePrior(n_components=4, n_latent=8)
    loss = prior.separation_loss(alpha=2.0)
    assert loss.item() >= 0.0


def test_gmm_prior_balance_loss():
    K, D, B = 4, 6, 8
    prior = GaussianMixturePrior(n_components=K, n_latent=D)
    prior.train()
    gamma = torch.ones(B, K) / K  # uniform — should give near-zero balance loss
    loss = prior.balance_loss(gamma)
    assert loss.item() >= 0.0


def test_gmm_prior_kmeans_init():
    K, D = 3, 4
    prior = GaussianMixturePrior(n_components=K, n_latent=D)
    samples = torch.randn(50, D)
    prior.kmeans_init_(samples)
    # Centroids should be updated
    assert not torch.all(prior.mu_k == 0.0)


def test_expected_component_log_prob_shape():
    K, D, B = 6, 5, 3
    prior = GaussianMixturePrior(n_components=K, n_latent=D)
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)
    log_prob = prior.expected_component_log_prob(mu, logvar)
    assert log_prob.shape == (B, K)


# ---------------------------------------------------------------------------
# GMMModuleVAE
# ---------------------------------------------------------------------------

def test_gmm_module_vae_forward_shapes():
    B, N, D, K = 4, 30, 8, 5
    model = GMMModuleVAE(
        n_features=N, n_latent=D, n_modules=K, hidden_dims=[32, 16]
    )
    x = torch.randn(B, N)
    recon_x, mu, logvar, z, gamma = model(x)
    assert recon_x.shape == (B, N)
    assert mu.shape == (B, D)
    assert logvar.shape == (B, D)
    assert z.shape == (B, D)
    assert gamma.shape == (B, K)
    assert torch.allclose(gamma.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_gmm_module_vae_get_gamma():
    B, N, D, K = 3, 20, 6, 4
    model = GMMModuleVAE(n_features=N, n_latent=D, n_modules=K, hidden_dims=[16])
    model.eval()
    x = torch.randn(B, N)
    gamma = model.get_gamma(x)
    assert gamma.shape == (B, K)


def test_gmm_module_vae_hard_assignments():
    B, N, D, K = 3, 20, 6, 4
    model = GMMModuleVAE(n_features=N, n_latent=D, n_modules=K, hidden_dims=[16])
    model.eval()
    x = torch.randn(B, N)
    hard = model.get_hard_assignments(x)
    assert hard.shape == (B,)
    assert (hard >= 0).all() and (hard < K).all()


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def test_kl_vade_shape():
    B, D, K = 4, 8, 3
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)
    gamma = torch.softmax(torch.randn(B, K), dim=-1)
    mu_k = torch.randn(K, D)
    sigma2_k = torch.ones(K)
    log_pi = torch.log(torch.ones(K) / K)

    kl = kl_vade(mu, logvar, gamma, mu_k, sigma2_k, log_pi)
    assert kl.dim() == 0
    assert kl.item() >= 0.0


def test_kl_vade_free_bits():
    B, D, K = 4, 8, 3
    mu = torch.randn(B, D)
    logvar = torch.zeros(B, D)
    gamma = torch.softmax(torch.randn(B, K), dim=-1)
    mu_k = torch.randn(K, D)
    sigma2_k = torch.ones(K)
    log_pi = torch.log(torch.ones(K) / K)

    kl_no_fb = kl_vade(mu, logvar, gamma, mu_k, sigma2_k, log_pi, free_bits=0.0)
    kl_with_fb = kl_vade(mu, logvar, gamma, mu_k, sigma2_k, log_pi, free_bits=1.0)
    assert kl_with_fb.item() >= kl_no_fb.item() - 1e-5


def test_warmup_loss_forward():
    B, N = 4, 20
    x = torch.randn(B, N)
    recon_x = torch.randn(B, N)
    mu = torch.randn(B, 8)
    logvar = torch.zeros(B, 8)

    loss_f = WarmupLoss(beta=1.0, kl_warmup_epochs=10)
    storer = {}
    loss = loss_f(x, recon_x, mu, logvar, storer=storer, epoch=5)
    assert loss.dim() == 0
    assert "recon_loss" in storer


def test_gmm_vae_loss_forward():
    B, N, D, K = 4, 30, 8, 5
    model = GMMModuleVAE(n_features=N, n_latent=D, n_modules=K, hidden_dims=[16])
    model.train()
    x = torch.randn(B, N)
    recon_x, mu, logvar, z, gamma = model(x)

    loss_f = GMMVAELoss(beta=1.0, sep_strength=0.01, bal_strength=0.01)
    storer = {}
    loss = loss_f(
        x=x, recon_x=recon_x, mu=mu, logvar=logvar,
        gamma=gamma, model=model, storer=storer, epoch=5,
    )
    assert loss.dim() == 0
    assert loss.item() >= 0.0
    assert "recon_loss" in storer
    assert "kl_loss" in storer


def test_gmm_vae_loss_backward():
    """Loss must be differentiable end-to-end."""
    B, N, D, K = 3, 20, 6, 4
    model = GMMModuleVAE(n_features=N, n_latent=D, n_modules=K, hidden_dims=[16])
    model.train()
    x = torch.randn(B, N)
    recon_x, mu, logvar, z, gamma = model(x)

    loss_f = GMMVAELoss(beta=1.0)
    loss = loss_f(x=x, recon_x=recon_x, mu=mu, logvar=logvar, gamma=gamma, model=model)
    loss.backward()
    # Check that encoder params have gradients
    for name, param in model.encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_kl_annealing():
    loss_f = GMMVAELoss(beta=1.0, kl_warmup_epochs=10, kl_anneal_mode="linear")
    assert loss_f.get_beta_for_epoch(0) == 0.0
    assert abs(loss_f.get_beta_for_epoch(5) - 0.5) < 1e-5
    assert loss_f.get_beta_for_epoch(10) == 1.0
    assert loss_f.get_beta_for_epoch(20) == 1.0


def test_kl_annealing_cyclical():
    loss_f = GMMVAELoss(beta=2.0, kl_anneal_mode="cyclical", kl_cycle_length=10)
    b0 = loss_f.get_beta_for_epoch(0)
    b5 = loss_f.get_beta_for_epoch(5)
    b10 = loss_f.get_beta_for_epoch(10)
    assert b0 == 0.0
    assert b5 == 2.0  # at half cycle length → full beta
    assert b10 == 0.0  # new cycle starts


def test_gaussian_nll_mse_fallback():
    x = torch.randn(3, 4)
    recon = torch.randn(3, 4)
    nll = gaussian_nll(x, recon, log_var=None, reduction="mean")
    mse = torch.nn.functional.mse_loss(recon, x, reduction="mean")
    assert torch.allclose(nll, mse)


def test_kl_normal_loss_zero_for_unit_gaussian():
    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)
    kl = kl_normal_loss(mu, logvar, reduction="sum")
    assert abs(kl.item()) < 1e-5


def test_kl_normal_loss_with_free_bits():
    mu = torch.zeros(5, 4)
    logvar = torch.zeros(5, 4)
    kl_total, kl_per_dim = kl_normal_loss_with_free_bits(mu, logvar, free_bits=0.5)
    # KL for unit Gaussian = 0, but free bits clamps to 0.5 per dim
    assert abs(kl_total.item() - 4 * 0.5) < 1e-4  # 4 dims × 0.5 nats


# ---------------------------------------------------------------------------
# Phase-1 review fixes: KL scaling, shared recon target, annealing, hier loss
# ---------------------------------------------------------------------------

def _loss_inputs(B=6, N=20, D=4, K=3, seed=0):
    torch.manual_seed(seed)
    model = GMMModuleVAE(n_features=N, n_latent=D, n_modules=K, hidden_dims=[8], use_batch_norm=False)
    x = torch.randn(B, N) * 3 + 1
    recon_x, mu, logvar, z, gamma = model(x)
    return model, x, recon_x, mu, logvar, gamma


def test_kl_per_element_scales_kl_by_profile_length():
    model, x, recon_x, mu, logvar, gamma = _loss_inputs()
    N = x.shape[1]
    legacy = GMMVAELoss(beta=1.0, sep_strength=0.0, bal_strength=0.0, kl_per_element=False)
    scaled = GMMVAELoss(beta=1.0, sep_strength=0.0, bal_strength=0.0, kl_per_element=True)
    s_leg, s_new = {}, {}
    legacy(x, recon_x, mu, logvar, gamma, model, storer=s_leg)
    scaled(x, recon_x, mu, logvar, gamma, model, storer=s_new)
    assert abs(s_new["kl_loss"][0] - s_leg["kl_loss"][0] / N) < 1e-6
    assert abs(s_new["recon_loss"][0] - s_leg["recon_loss"][0]) < 1e-6


def test_warmup_loss_follows_model_normalize_flag():
    model, x, recon_x, mu, logvar, _ = _loss_inputs()
    model.normalize_input = True
    warm = WarmupLoss(beta=0.0, normalize_input=False)   # loss flag wrong on purpose
    s_model, s_raw = {}, {}
    warm(x, recon_x, mu, logvar, storer=s_model, model=model)
    warm(x, recon_x, mu, logvar, storer=s_raw)           # no model → raw target
    x_z = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp(min=1e-6)
    expected = torch.nn.functional.mse_loss(recon_x, x_z).item()
    assert abs(s_model["recon_loss"][0] - expected) < 1e-5
    assert abs(s_raw["recon_loss"][0] - torch.nn.functional.mse_loss(recon_x, x).item()) < 1e-5


def test_masked_recon_keeps_negative_values():
    model, x, recon_x, mu, logvar, gamma = _loss_inputs()
    x = x.clone()
    x[:, :5] = 0.0
    x[:, 5:10] = -2.0
    loss_f = GMMVAELoss(beta=0.0, sep_strength=0.0, bal_strength=0.0, masked_recon=True)
    s = {}
    loss_f(x, recon_x, mu, logvar, gamma, model, storer=s)
    mask = (x != 0).float()
    expected = (((recon_x - x) ** 2) * mask).sum() / mask.sum()
    assert abs(s["recon_loss"][0] - expected.item()) < 1e-5


def test_get_beta_none_returns_full_beta():
    lin = GMMVAELoss(beta=0.7, kl_warmup_epochs=100)
    cyc = GMMVAELoss(beta=0.7, kl_anneal_mode="cyclical", kl_cycle_length=10)
    assert lin.get_beta_for_epoch(0) == 0.0 and lin.get_beta_for_epoch(None) == 0.7
    assert cyc.get_beta_for_epoch(0) == 0.0 and cyc.get_beta_for_epoch(None) == 0.7
    assert WarmupLoss(beta=0.3, kl_warmup_epochs=10).get_beta_for_epoch(None) == 0.3


def test_separation_loss_is_mean_over_pairs():
    torch.manual_seed(0)
    small = GaussianMixturePrior(n_components=3, n_latent=2)
    big = GaussianMixturePrior(n_components=12, n_latent=2)
    with torch.no_grad():
        small.mu_k.zero_(); big.mu_k.zero_()        # every pair violates by α(σ_i+σ_j)
        small.sigma2_raw.fill_(0.0); big.sigma2_raw.fill_(0.0)
    assert torch.isclose(small.separation_loss(), big.separation_loss())


def test_encoder_logvar_is_bounded():
    enc = FeatureEncoder(n_features=5, n_latent=3, hidden_dims=[4], use_batch_norm=False)
    with torch.no_grad():
        enc.fc_logvar.weight.fill_(100.0)
        enc.fc_logvar.bias.fill_(1000.0)
    _, logvar = enc(torch.randn(2, 5))
    assert logvar.max() <= 10.0 and logvar.min() >= -10.0


def test_hierarchical_loss_margin_and_corr_weight():
    from bsvae.models.losses import hierarchical_loss
    mu = torch.zeros(4, 3)
    mu[1] = 0.5    # 0.5 away from mu[0]
    mu[3] = 5.0    # far from mu[2]
    groups = {"gA": [0, 1], "gB": [2, 3]}
    # Margin larger than the gA distance → gA contributes 0, gB contributes dist - margin.
    loss = hierarchical_loss(mu, groups, margin=1.0)
    expected = (5.0 * math.sqrt(3) - 1.0) / 2.0   # mean over the two genes (gA = 0)
    assert abs(loss.item() - expected) < 1e-4
    # Correlation weighting: anti-correlated isoforms get zero weight.
    t = torch.linspace(-1, 1, 10)
    x = torch.stack([t, t, t, -t])          # gA isoforms correlated, gB anti-correlated
    loss_w = hierarchical_loss(mu, groups, x=x, margin=0.0, corr_weight=True)
    assert abs(loss_w.item() - 0.5 * math.sqrt(3)) < 1e-4  # only gA survives


def test_latent_corr_loss_is_shift_invariant():
    from bsvae.models.losses import latent_corr_loss
    torch.manual_seed(0)
    x = torch.randn(8, 30)
    mu = torch.randn(8, 4)
    a = latent_corr_loss(x, mu)
    b = latent_corr_loss(x, mu + 10.0)   # shared offset must not change the loss
    assert torch.isclose(a, b, atol=1e-5)
