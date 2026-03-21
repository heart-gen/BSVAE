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


def test_trainer_warmup_phase():
    """Phase 1: WarmupLoss should be used; μ buffer should be populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_model()
        loader = _make_loader()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        gmm_loss = GMMVAELoss(beta=1.0)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            gmm_loss_f=gmm_loss,
            save_dir=tmpdir,
            is_progress_bar=False,
            warmup_epochs=3,
            transition_epochs=2,
        )
        trainer(loader, epochs=2)  # stays fully in warmup
        assert trainer._mu_buffer_count > 0 or len(trainer._mu_buffer) > 0


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
