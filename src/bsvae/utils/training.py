"""
Two-phase Trainer for GMMModuleVAE.

Phase 1 (epochs 1 .. warmup_epochs):
  Standard N(0,I) KL + reconstruction (WarmupLoss).
  Encoder means are accumulated in a circular buffer (max 50 K entries).

At epoch = warmup_epochs:
  MiniBatchKMeans(K) on the μ buffer → initialise GaussianMixturePrior.
  Switch to GMMVAELoss.
  Begin transition annealing (epochs 0 .. transition_epochs): GMM weight 0 → 1.

Phase 2 (remaining epochs):
  Full GMMVAELoss with KL annealing schedule.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from timeit import default_timer
from typing import Dict, List, Optional

import torch
from tqdm import trange

from bsvae.models.losses import GMMVAELoss, WarmupLoss


TRAIN_LOSSES_LOGFILE = "train_losses.csv"


class Trainer:
    """
    Two-phase Trainer for GMMModuleVAE.

    Parameters
    ----------
    model : GMMModuleVAE
    optimizer : torch.optim.Optimizer
    gmm_loss_f : GMMVAELoss
        Loss used in Phase 2 (GMM phase).
    device : torch.device
    logger : logging.Logger
    save_dir : str
    is_progress_bar : bool
    warmup_epochs : int
        Length of Phase 1 (N(0,I) KL). Default: 20.
    transition_epochs : int
        Length of GMM weight ramp-in (Phase 1 → Phase 2). Default: 10.
    mu_buffer_size : int
        Max encoder means accumulated during Phase 1. Default: 50_000.
    warmup_loss_f : WarmupLoss or None
        If None, a default WarmupLoss is constructed.
    gene_groups : dict or None
        For hierarchical loss (gene_id → list of feature dataset indices).
    """

    def __init__(
        self,
        model,
        optimizer,
        gmm_loss_f: GMMVAELoss,
        device: torch.device = torch.device("cpu"),
        logger: logging.Logger = logging.getLogger(__name__),
        save_dir: str = "results",
        is_progress_bar: bool = True,
        warmup_epochs: int = 20,
        transition_epochs: int = 10,
        mu_buffer_size: int = 50_000,
        warmup_loss_f: Optional[WarmupLoss] = None,
        gene_groups: Optional[Dict] = None,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.gmm_loss_f = gmm_loss_f
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.mu_buffer_size = mu_buffer_size
        self.gene_groups = gene_groups or {}
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger

        self.warmup_loss_f = warmup_loss_f or WarmupLoss(
            beta=gmm_loss_f.beta,
            kl_warmup_epochs=warmup_epochs,
            kl_anneal_mode=gmm_loss_f.kl_anneal_mode,
            kl_cycle_length=gmm_loss_f.kl_cycle_length,
            free_bits=gmm_loss_f.free_bits,
        )

        self._mu_buffer: List[torch.Tensor] = []
        self._mu_buffer_count = 0
        self._feature_id_to_idx: Optional[Dict[str, int]] = None

        self.losses_logger = LossesLogger(
            os.path.join(save_dir, TRAIN_LOSSES_LOGFILE),
            log_level=logger.level,
        )
        self.logger.info("Training device: %s", device)
        self.logger.info(
            "Phase 1: %d warmup epochs | transition: %d epochs",
            warmup_epochs, transition_epochs,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def __call__(self, data_loader, epochs: int = 100, checkpoint_every: int = 10):
        start = default_timer()
        self.model.train()

        for epoch in range(epochs):
            storer = defaultdict(list)
            in_warmup = epoch < self.warmup_epochs
            in_transition = (
                self.warmup_epochs <= epoch < self.warmup_epochs + self.transition_epochs
            )
            gmm_weight = self._gmm_weight(epoch)

            # K-means warm-start at the transition point
            if epoch == self.warmup_epochs and self._mu_buffer:
                self._kmeans_init()

            mean_loss = self._train_epoch(
                data_loader, storer, epoch, in_warmup, gmm_weight
            )
            self.logger.info(
                "Epoch %d | loss=%.4f | phase=%s | gmm_weight=%.2f",
                epoch + 1, mean_loss,
                "warmup" if in_warmup else ("transition" if in_transition else "gmm"),
                gmm_weight,
            )
            self.losses_logger.log(epoch, storer)

            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_dir, f"model-{epoch+1}.pt"),
                )

        self.model.eval()
        delta_time = (default_timer() - start) / 60
        self.logger.info("Finished training after %.1f min.", delta_time)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _gmm_weight(self, epoch: int) -> float:
        """Transition ramp: 0 during warmup, linear 0→1 during transition, 1 after."""
        if epoch < self.warmup_epochs:
            return 0.0
        t = epoch - self.warmup_epochs
        if self.transition_epochs <= 0:
            return 1.0
        return min(t / self.transition_epochs, 1.0)

    def _kmeans_init(self):
        """Run K-means on the accumulated μ buffer to warm-start the GMM prior."""
        self.logger.info(
            "Running K-means init on %d μ samples...", self._mu_buffer_count
        )
        mu_all = torch.cat(self._mu_buffer, dim=0)
        if mu_all.shape[0] > self.mu_buffer_size:
            idx = torch.randperm(mu_all.shape[0])[: self.mu_buffer_size]
            mu_all = mu_all[idx]
        self.model.gmm_prior.kmeans_init_(mu_all)
        # Free the buffer memory
        self._mu_buffer = []
        self._mu_buffer_count = 0
        self.logger.info("GMM prior initialised; switching to Phase 2.")

    def _train_epoch(self, data_loader, storer, epoch, in_warmup, gmm_weight):
        epoch_loss = 0.0
        with trange(
            len(data_loader), desc=f"Epoch {epoch+1}", leave=False,
            disable=not self.is_progress_bar
        ) as t:
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    feat_ids = batch[1] if len(batch) > 1 else None
                else:
                    x = batch
                    feat_ids = None
                x = x.to(self.device)

                feature_idx = None
                if self.gmm_loss_f.hier_strength > 0.0 and self.gene_groups:
                    if feat_ids is None:
                        raise ValueError(
                            "Hierarchical loss requires batch feature ids, but the DataLoader "
                            "returned only profiles. Ensure the dataset returns feature ids."
                        )
                    feature_idx = self._resolve_feature_idx(feat_ids, data_loader)

                iter_loss = self._train_iteration(
                    x, storer, epoch, in_warmup, gmm_weight, feature_idx
                )
                epoch_loss += iter_loss
                t.set_postfix(loss=iter_loss)
                t.update()

        return epoch_loss / max(len(data_loader), 1)

    def _train_iteration(self, x, storer, epoch, in_warmup, gmm_weight, feature_idx=None):
        recon_x, mu, logvar, z, gamma = self.model(x)

        if in_warmup:
            loss = self.warmup_loss_f(x, recon_x, mu, logvar, storer=storer, epoch=epoch)
            # Accumulate μ for K-means init
            with torch.no_grad():
                self._mu_buffer.append(mu.detach().cpu())
                self._mu_buffer_count += mu.shape[0]
                # Trim buffer if too large
                if self._mu_buffer_count > self.mu_buffer_size * 2:
                    mu_all = torch.cat(self._mu_buffer, dim=0)
                    self._mu_buffer = [mu_all[-self.mu_buffer_size:]]
                    self._mu_buffer_count = self.mu_buffer_size
        else:
            loss = self.gmm_loss_f(
                x=x, recon_x=recon_x, mu=mu, logvar=logvar, gamma=gamma,
                model=self.model, storer=storer, epoch=epoch,
                gene_groups=self.gene_groups or None,
                feature_idx=feature_idx,
                gmm_weight=gmm_weight,
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _resolve_feature_idx(self, feat_ids, data_loader) -> torch.Tensor:
        if isinstance(feat_ids, torch.Tensor):
            return feat_ids.to(self.device)

        if isinstance(feat_ids, (list, tuple)):
            if not feat_ids:
                raise ValueError("Empty feature id list in batch; cannot map to dataset indices.")
            if all(isinstance(fid, int) for fid in feat_ids):
                return torch.tensor(feat_ids, device=self.device, dtype=torch.long)

            id_to_idx = self._get_feature_id_to_idx(data_loader)
            missing = [fid for fid in feat_ids if str(fid) not in id_to_idx]
            if missing:
                raise ValueError(
                    f"{len(missing)} feature ids from batch not found in dataset feature_ids; "
                    "cannot map to dataset indices for hierarchical loss."
                )
            idx = [id_to_idx[str(fid)] for fid in feat_ids]
            return torch.tensor(idx, device=self.device, dtype=torch.long)

        if isinstance(feat_ids, int):
            return torch.tensor([feat_ids], device=self.device, dtype=torch.long)

        if isinstance(feat_ids, str):
            id_to_idx = self._get_feature_id_to_idx(data_loader)
            if feat_ids not in id_to_idx:
                raise ValueError(
                    "Feature id from batch not found in dataset feature_ids; "
                    "cannot map to dataset indices for hierarchical loss."
                )
            return torch.tensor([id_to_idx[feat_ids]], device=self.device, dtype=torch.long)

        raise TypeError(
            f"Unsupported feature id type {type(feat_ids)!r} for hierarchical loss mapping."
        )

    def _get_feature_id_to_idx(self, data_loader) -> Dict[str, int]:
        if self._feature_id_to_idx is None:
            dataset = getattr(data_loader, "dataset", None)
            feature_ids = getattr(dataset, "feature_ids", None)
            if feature_ids is None:
                raise ValueError(
                    "DataLoader dataset lacks feature_ids; cannot map batch feature ids "
                    "to dataset indices for hierarchical loss."
                )
            self._feature_id_to_idx = {str(fid): i for i, fid in enumerate(feature_ids)}
        return self._feature_id_to_idx


# ---------------------------------------------------------------------------
# Evaluator (kept for evaluation pass after training)
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Simple evaluator — computes loss on a dataloader without gradient updates.
    """

    def __init__(
        self,
        model,
        loss_f,
        device: torch.device = torch.device("cpu"),
        logger: logging.Logger = logging.getLogger(__name__),
        save_dir: str = "results",
        is_progress_bar: bool = True,
    ):
        self.model = model.to(device)
        self.loss_f = loss_f
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar

    def __call__(self, data_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(self.device)
                recon_x, mu, logvar, z, gamma = self.model(x)
                loss = self.loss_f(
                    x=x, recon_x=recon_x, mu=mu, logvar=logvar,
                    gamma=gamma, model=self.model,
                )
                total_loss += loss.item()

        mean_loss = total_loss / max(len(data_loader), 1)
        self.logger.info("Evaluation loss: %.4f", mean_loss)
        return mean_loss


# ---------------------------------------------------------------------------
# Loss logger
# ---------------------------------------------------------------------------

class LossesLogger:
    """Write epoch-level training losses to a CSV file."""

    def __init__(self, file_path_name: str, log_level: int = logging.DEBUG):
        dir_name = os.path.dirname(file_path_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if os.path.isfile(file_path_name):
            try:
                os.remove(file_path_name)
            except FileNotFoundError:
                pass

        self.logger = logging.getLogger("losses_logger")
        self.logger.handlers.clear()
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        fh = logging.FileHandler(file_path_name)
        fh.setLevel(logging.NOTSET)
        self.logger.addHandler(fh)
        self.logger.info("Epoch,Loss,Value")

    def log(self, epoch: int, losses_storer: dict):
        for k, v in losses_storer.items():
            mean_val = sum(v) / len(v)
            self.logger.info(f"{epoch},{k},{mean_val:.6f}")
