"""Tests for two-phase Trainer and modelIO."""

import os
import tempfile
import pytest
import torch
from torch import optim
from torch.utils.data import DataLoader

from bsvae.models.gmvae import GMMModuleVAE
from bsvae.models.losses import GMMVAELoss, WarmupLoss
from bsvae.utils.training import Trainer, Evaluator
from bsvae.utils.modelIO import save_model, load_model, load_metadata


def _make_loader(n_samples=30, n_items=40, batch_size=8):
    """Create a minimal DataLoader returning (profiles, feature_ids)."""
    profiles = torch.randn(n_items, n_samples)
    feat_ids = [f"feat_{i}" for i in range(n_items)]

    class SimpleDataset(torch.utils.data.Dataset):
        feature_ids = feat_ids  # required by Trainer._get_feature_id_to_idx

        def __len__(self):
            return len(feat_ids)

        def __getitem__(self, idx):
            return profiles[idx], feat_ids[idx]

    return DataLoader(SimpleDataset(), batch_size=batch_size, shuffle=True)


def _make_model(n_samples=30, n_latent=6, n_modules=4):
    return GMMModuleVAE(
        n_features=n_samples,
        n_latent=n_latent,
        n_modules=n_modules,
        hidden_dims=[16],
        use_batch_norm=False,
    )


def test_trainer_rejects_warmup_longer_than_training():
    """warmup_epochs >= epochs would never initialise the GMM prior."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model, optimizer=optimizer, gmm_loss_f=GMMVAELoss(beta=1.0),
            save_dir=tmpdir, is_progress_bar=False, warmup_epochs=3, transition_epochs=2,
        )
        with pytest.raises(ValueError, match="warmup_epochs"):
            trainer(loader, epochs=2)


def test_trainer_syncs_warmup_loss_with_model_normalization():
    """The warmup loss must reconstruct the same (z-scored) target as the GMM loss."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        model.normalize_input = True
        warm = WarmupLoss(beta=1.0, normalize_input=False)
        trainer = Trainer(
            model=model, optimizer=optim.Adam(model.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=1.0, masked_recon=True), warmup_loss_f=warm,
            save_dir=tmpdir, is_progress_bar=False, warmup_epochs=1, transition_epochs=1,
        )
        assert trainer.warmup_loss_f.normalize_input is True
        assert trainer.warmup_loss_f.masked_recon is True


def test_trainer_phase_relative_kl_epoch():
    """GMMVAELoss receives the epoch counted from the start of the GMM phase."""
    seen = []

    class SpyLoss(GMMVAELoss):
        def forward(self, *args, **kwargs):
            seen.append(kwargs.get("epoch"))
            return super().forward(*args, **kwargs)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        trainer = Trainer(
            model=model, optimizer=optim.Adam(model.parameters(), lr=1e-3),
            gmm_loss_f=SpyLoss(beta=0.1), save_dir=tmpdir, is_progress_bar=False,
            warmup_epochs=2, transition_epochs=1,
        )
        trainer(loader, epochs=4)
        assert min(seen) == 0 and max(seen) == 1


def _trainer_for_revival(tmpdir, usage_floor=0.002):
    model = _make_model(n_modules=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(
        model=model, optimizer=optimizer, gmm_loss_f=GMMVAELoss(beta=0.1),
        save_dir=tmpdir, is_progress_bar=False, warmup_epochs=1, transition_epochs=1,
        collapse_min_usage=usage_floor,
    )
    # Give the prior non-trivial mixing logits (as K-means init would) and
    # populate Adam state so the reset path is exercised.
    gmm = model.gmm_prior
    with torch.no_grad():
        gmm.log_pi_unnorm.copy_(torch.tensor([6.0, 5.0, 4.0, 3.0]))
        gmm.rho_ema.copy_(torch.tensor([0.6, 0.3, 0.09, 0.01]))
    loss = gmm.separation_loss() + gmm.log_pi.sum()
    loss.backward()
    optimizer.step()
    return trainer, model, optimizer


def test_revival_splits_best_component_and_resets_adam():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer, model, optimizer = _trainer_for_revival(tmpdir)
        gmm = model.gmm_prior
        pi_before = gmm.pi.detach().clone()
        usage = [0.6, 0.3, 0.09, 0.0005]   # component 3 dead
        trainer._maybe_reinit_collapsed_components(epoch=5, usage=usage)
        pi_after = gmm.pi.detach()
        # Split: revived component shares the best component's weight equally.
        assert torch.isclose(pi_after[3], pi_after[0])
        # Relative to an untouched component, the split component's weight halves.
        ratio_before = pi_before[0] / pi_before[1]
        ratio_after = pi_after[0] / pi_after[1]
        assert torch.isclose(ratio_after, ratio_before / 2, rtol=1e-4)
        assert torch.isclose(gmm.rho_ema[3], gmm.rho_ema[0])
        # Centroid near the split component, variance copied.
        assert (gmm.mu_k[3] - gmm.mu_k[0]).norm() < 3 * gmm.sigma2_k[0].sqrt()
        assert torch.isclose(gmm.sigma2_k[3], gmm.sigma2_k[0])
        # Adam moments for the touched rows are zeroed.
        state = optimizer.state[gmm.mu_k]
        assert state["exp_avg"][3].abs().sum() == 0 and state["exp_avg"][0].abs().sum() == 0
        assert state["exp_avg"][1].abs().sum() > 0


def test_revival_spreads_multiple_dead_components():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer, model, _ = _trainer_for_revival(tmpdir)
        gmm = model.gmm_prior
        usage = [0.7, 0.3, 0.0001, 0.0001]
        trainer._maybe_reinit_collapsed_components(epoch=5, usage=usage)
        # Dead components 2 and 3 are split from the two live components (0 and 1).
        assert (gmm.mu_k[2] - gmm.mu_k[0]).norm() < 3 * gmm.sigma2_k[0].sqrt()
        assert (gmm.mu_k[3] - gmm.mu_k[1]).norm() < 3 * gmm.sigma2_k[1].sqrt()


def test_revival_uses_absolute_usage_floor():
    """A small but live module (above the absolute floor) is not revived."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer, model, _ = _trainer_for_revival(tmpdir, usage_floor=0.002)
        gmm = model.gmm_prior
        mu_before = gmm.mu_k.detach().clone()
        usage = [0.9, 0.05, 0.03, 0.02]   # 2 % usage is < 0.5/K but > 0.002
        trainer._maybe_reinit_collapsed_components(epoch=5, usage=usage)
        assert torch.equal(gmm.mu_k.detach(), mu_before)


def test_no_revival_on_last_epoch_or_during_transition():
    calls = []

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        trainer = Trainer(
            model=model, optimizer=optim.Adam(model.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=0.1), save_dir=tmpdir, is_progress_bar=False,
            warmup_epochs=1, transition_epochs=2, collapse_min_usage=1.0,  # everything "dead"
        )
        trainer._maybe_reinit_collapsed_components = lambda epoch, usage: calls.append(epoch)
        trainer(loader, epochs=5)
        # epochs 0 (warmup), 1-2 (transition, gmm_weight<1) and 4 (last) are excluded.
        assert calls == [3]


def test_collapse_threshold_deprecated_warning(caplog):
    import logging
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        logger = logging.getLogger("test_collapse_dep")
        with caplog.at_level(logging.WARNING, logger="test_collapse_dep"):
            Trainer(
                model=model, optimizer=optim.Adam(model.parameters(), lr=1e-3),
                gmm_loss_f=GMMVAELoss(beta=0.1), save_dir=tmpdir, is_progress_bar=False,
                logger=logger, collapse_threshold=0.5,
            )
        assert any("deprecated" in r.message for r in caplog.records)


def test_losses_csv_written_at_warning_level():
    import logging
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        logger = logging.getLogger("quiet_logger")
        logger.setLevel(logging.WARNING)
        trainer = Trainer(
            model=model, optimizer=optim.Adam(model.parameters(), lr=1e-3),
            gmm_loss_f=GMMVAELoss(beta=0.1), save_dir=tmpdir, is_progress_bar=False,
            logger=logger, warmup_epochs=1, transition_epochs=1,
        )
        trainer(loader, epochs=2)
        with open(os.path.join(tmpdir, "train_losses.csv")) as f:
            lines = f.read().strip().splitlines()
        assert lines[0] == "Epoch,Loss,Value"
        assert any(",recon_loss," in l for l in lines[1:])


def test_trainer_full_training():
    """Full training cycle (warmup + transition + GMM phase) without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        gmm_loss = GMMVAELoss(beta=0.1, sep_strength=0.001, bal_strength=0.001)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            gmm_loss_f=gmm_loss,
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=2,
            transition_epochs=1,
        )
        trainer(loader, epochs=5, checkpoint_every=2)

        # Checkpoints should be saved
        assert os.path.exists(os.path.join(tmpdir, "model-2.pt"))
        assert os.path.exists(os.path.join(tmpdir, "model-4.pt"))


def test_trainer_gmm_weight_schedule():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        gmm_loss = GMMVAELoss(beta=1.0)
        trainer = Trainer(
            model=model, optimizer=optimizer, gmm_loss_f=gmm_loss,
            save_dir=tmpdir, is_progress_bar=False,
            warmup_epochs=5, transition_epochs=5,
        )
        assert trainer._gmm_weight(0) == 0.0
        assert trainer._gmm_weight(4) == 0.0
        assert abs(trainer._gmm_weight(5) - 0.0) < 1e-5
        assert abs(trainer._gmm_weight(7) - 0.4) < 1e-5
        assert abs(trainer._gmm_weight(10) - 1.0) < 1e-5
        assert trainer._gmm_weight(20) == 1.0


def test_save_load_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model(n_samples=25, n_latent=5, n_modules=3)
        save_model(model, tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "specs.json"))
        assert os.path.exists(os.path.join(tmpdir, "model.pt"))

        metadata = load_metadata(tmpdir)
        assert metadata["n_modules"] == 3
        assert metadata["n_latent"] == 5
        assert metadata["n_features"] == 25

        loaded = load_model(tmpdir, is_gpu=False)
        assert isinstance(loaded, GMMModuleVAE)
        assert loaded.n_modules == 3


def test_save_load_model_state_preserved():
    """Loaded model should produce same output as saved model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model(n_samples=20, n_latent=4, n_modules=3)
        model.eval()
        x = torch.randn(2, 20)
        with torch.no_grad():
            _, mu_orig, _, _, gamma_orig = model(x)

        save_model(model, tmpdir)
        loaded = load_model(tmpdir, is_gpu=False)
        loaded.eval()
        with torch.no_grad():
            _, mu_loaded, _, _, gamma_loaded = loaded(x)

        assert torch.allclose(mu_orig, mu_loaded, atol=1e-5)
        assert torch.allclose(gamma_orig, gamma_loaded, atol=1e-5)


def test_trainer_with_hier_loss():
    """Trainer with gene_groups + hier_strength > 0 runs through GMM phase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n_samples = 30
        model = _make_model(n_samples=n_samples)
        loader = _make_loader(n_samples=n_samples, n_items=40)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        gmm_loss = GMMVAELoss(beta=0.1, hier_strength=0.5)

        # feat_0,feat_1 are isoforms of geneA; feat_2,feat_3 of geneB
        gene_groups = {"geneA": [0, 1], "geneB": [2, 3]}
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            gmm_loss_f=gmm_loss,
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=2,
            transition_epochs=1,
            gene_groups=gene_groups,
        )
        trainer(loader, epochs=4)
        # Training completed: model should be back in eval mode
        assert not trainer.model.training


def test_evaluator():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        model.eval()
        loader = _make_loader()
        loss_f = GMMVAELoss(beta=0.1)
        evaluator = Evaluator(
            model=model, loss_f=loss_f, save_dir=tmpdir, is_progress_bar=False
        )
        mean_loss = evaluator(loader)
        assert isinstance(mean_loss, float)
        assert mean_loss >= 0.0
        comps = evaluator(loader, return_components=True)
        assert set(comps) == {"loss", "recon", "kl", "elbo"}
        assert abs(comps["elbo"] - (comps["recon"] + 0.1 * comps["kl"])) < 1e-6


def test_evaluator_uses_full_beta_despite_annealing():
    """Validation loss must include the KL even when training used a β ramp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        model.eval()
        loader = _make_loader()
        loss_f = GMMVAELoss(beta=1.0, kl_warmup_epochs=50)
        evaluator = Evaluator(model=model, loss_f=loss_f, save_dir=tmpdir, is_progress_bar=False)
        comps = evaluator(loader, return_components=True)
        assert comps["kl"] > 0
        assert comps["elbo"] > comps["recon"]


# ---------------------------------------------------------------------------
# normalize_input=True round-trip tests
# ---------------------------------------------------------------------------

def _make_model_normalized(n_samples=30, n_latent=6, n_modules=4):
    return GMMModuleVAE(
        n_features=n_samples,
        n_latent=n_latent,
        n_modules=n_modules,
        hidden_dims=[16],
        use_batch_norm=False,
        normalize_input=True,
    )


def test_save_load_normalize_input_flag():
    """normalize_input=True must survive save/load round-trip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model_normalized(n_samples=20, n_latent=4, n_modules=3)
        assert model.normalize_input is True

        save_model(model, tmpdir)
        metadata = load_metadata(tmpdir)
        assert metadata["normalize_input"] is True

        loaded = load_model(tmpdir, is_gpu=False)
        assert loaded.normalize_input is True


def test_save_load_normalize_input_state_preserved():
    """Loaded model with normalize_input=True produces same output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model_normalized(n_samples=20, n_latent=4, n_modules=3)
        model.eval()
        x = torch.randn(2, 20)
        with torch.no_grad():
            _, mu_orig, _, _, gamma_orig = model(x)

        save_model(model, tmpdir)
        loaded = load_model(tmpdir, is_gpu=False)
        loaded.eval()
        with torch.no_grad():
            _, mu_loaded, _, _, gamma_loaded = loaded(x)

        assert torch.allclose(mu_orig, mu_loaded, atol=1e-5)
        assert torch.allclose(gamma_orig, gamma_loaded, atol=1e-5)


def test_warmup_loss_normalize_input():
    """WarmupLoss with normalize_input=True should match normalized target."""
    B, N = 4, 20
    x = torch.randn(B, N) + 5.0  # non-zero mean to make normalization matter
    mu_x = x.mean(dim=-1, keepdim=True)
    sigma_x = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    x_normed = (x - mu_x) / sigma_x

    # Perfect reconstruction of the normalized target should yield zero loss
    loss_f = WarmupLoss(beta=0.0, normalize_input=True)
    loss = loss_f(x, x_normed, torch.zeros(B, 4), torch.zeros(B, 4))
    assert loss.item() < 1e-5

    # With normalize_input=False and the same inputs the loss should be large
    loss_f_raw = WarmupLoss(beta=0.0, normalize_input=False)
    loss_raw = loss_f_raw(x, x_normed, torch.zeros(B, 4), torch.zeros(B, 4))
    assert loss_raw.item() > 1.0  # raw x vs normalized recon = big error


def test_trainer_normalize_input_full_cycle():
    """Full training cycle with normalize_input=True completes without error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model_normalized()
        loader = _make_loader()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        gmm_loss = GMMVAELoss(beta=0.1, normalize_input=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            gmm_loss_f=gmm_loss,
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=2,
            transition_epochs=1,
        )
        # Verify the auto-constructed WarmupLoss inherited normalize_input
        assert trainer.warmup_loss_f.normalize_input is True

        trainer(loader, epochs=4)
        assert not trainer.model.training
