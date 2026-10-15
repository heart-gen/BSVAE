"""
Two-phase Trainer for GMMModuleVAE.

Phase 1 (epochs 1 .. warmup_epochs):
  Standard N(0,I) KL + reconstruction (WarmupLoss).

At epoch = warmup_epochs:
  Fresh encoder means for every feature → K-means (MiniBatchKMeans at
  transcript scale) → initialise GaussianMixturePrior.
  Switch to GMMVAELoss.
  Begin transition annealing (epochs 0 .. transition_epochs): GMM weight 0 → 1.

Phase 2 (remaining epochs):
  Full GMMVAELoss with KL annealing schedule (epochs counted from the start
  of the GMM phase).  Dead components are revived by splitting the most-used
  component, never during the transition ramp or after the final epoch.
"""

from __future__ import annotations

import logging
import math
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
    warmup_loss_f : WarmupLoss or None
        If None, a default WarmupLoss is constructed.  Its ``normalize_input``
        and ``masked_recon`` flags are always synchronised with the model and
        the GMM loss so both phases reconstruct the same target.
    gene_groups : dict or None
        For hierarchical loss (gene_id → list of feature dataset indices).
    collapse_min_usage : float
        Absolute usage floor: a component is dead when its epoch-averaged
        soft usage (mean γ over all features) falls below this fraction.
        Default: 0.002 (0.2 % of features), independent of K so that small
        real modules are not treated as collapsed.
    collapse_noise_scale : float
        Perturbation magnitude for dead-component revival expressed as a
        multiple of the split component's σ. Default: 0.5.
    seed : int
        Random seed for K-means initialisation. Default: 0.
    collapse_threshold : float or None
        Deprecated relative floor (threshold / K). Ignored with a warning;
        use ``collapse_min_usage``.
    amp : bool
        Mixed precision (bfloat16 autocast) for the forward pass on CUDA;
        losses are computed in float32. Default: False.
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
        freeze_gmm_epochs: int = 0,
        warmup_loss_f: Optional[WarmupLoss] = None,
        gene_groups: Optional[Dict] = None,
        collapse_min_usage: float = 0.002,
        collapse_noise_scale: float = 0.5,
        seed: int = 0,
        collapse_threshold: Optional[float] = None,
        mu_buffer_size: Optional[int] = None,
        amp: bool = False,
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.gmm_loss_f = gmm_loss_f
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        self.freeze_gmm_epochs = max(int(freeze_gmm_epochs), 0)
        self.gene_groups = gene_groups or {}
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self._gmm_prior_trainable = True
        self.collapse_min_usage = float(collapse_min_usage)
        self.collapse_noise_scale = collapse_noise_scale
        self.seed = int(seed)
        self.amp = bool(amp) and device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda") if self.amp else None
        if collapse_threshold is not None:
            self.logger.warning(
                "collapse_threshold is deprecated and ignored; use collapse_min_usage "
                "(absolute usage fraction, default 0.002)."
            )

        self.warmup_loss_f = warmup_loss_f or WarmupLoss(
            beta=gmm_loss_f.beta,
            kl_warmup_epochs=warmup_epochs,
            kl_anneal_mode=gmm_loss_f.kl_anneal_mode,
            kl_cycle_length=gmm_loss_f.kl_cycle_length,
            free_bits=gmm_loss_f.free_bits,
            kl_per_element=getattr(gmm_loss_f, "kl_per_element", True),
        )
        # Both phases must reconstruct the same target.
        model_norm = getattr(self.model, "normalize_input", gmm_loss_f.normalize_input)
        self.warmup_loss_f.normalize_input = bool(model_norm)
        self.warmup_loss_f.masked_recon = bool(getattr(gmm_loss_f, "masked_recon", False))

        self._feature_id_to_idx: Optional[Dict[str, int]] = None
        self._usage_sum: Optional[torch.Tensor] = None
        self._usage_count = 0

        # Precomputed reverse mapping: dataset_idx (int) → gene_id (str)
        # Used by hierarchical_loss for O(B) fast path instead of O(G × n_iso) GPU ops
        self._idx_to_gene: Dict[int, str] = {}
        for gid, idxs in self.gene_groups.items():
            for idx in idxs:
                self._idx_to_gene[idx] = gid

        self.losses_logger = LossesLogger(
            os.path.join(save_dir, TRAIN_LOSSES_LOGFILE),
        )
        self.logger.info("Training device: %s", device)
        self.logger.info(
            "Phase 1: %d warmup epochs | transition: %d epochs",
            warmup_epochs, transition_epochs,
        )
        if self.freeze_gmm_epochs > 0:
            self.logger.info(
                "Freeze GMM prior for first %d GMM epochs",
                self.freeze_gmm_epochs,
            )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def __call__(self, data_loader, epochs: int = 100, checkpoint_every: int = 10):
        start = default_timer()
        if self.warmup_epochs >= epochs:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be smaller than epochs "
                f"({epochs}); otherwise the GMM prior is never initialised."
            )
        if self.warmup_epochs == 0:
            self.logger.warning(
                "warmup_epochs=0: K-means initialisation will run on an untrained encoder."
            )
        self.model.train()

        for epoch in range(epochs):
            storer = defaultdict(list)
            in_warmup = epoch < self.warmup_epochs
            in_transition = (
                self.warmup_epochs <= epoch < self.warmup_epochs + self.transition_epochs
            )
            gmm_weight = self._gmm_weight(epoch)

            # K-means warm-start at the transition point
            if epoch == self.warmup_epochs:
                self._kmeans_init(data_loader)

            # Optionally freeze GMM prior params for early GMM epochs
            self._maybe_freeze_gmm_prior(epoch)

            self._reset_usage()
            mean_loss = self._train_epoch(
                data_loader, storer, epoch, in_warmup, gmm_weight
            )
            usage = self._epoch_usage()
            for k, val in enumerate(usage):
                storer[f"usage_soft_{k}"].append(float(val))
            self.logger.info(
                "Epoch %d | loss=%.4f | phase=%s | gmm_weight=%.2f",
                epoch + 1, mean_loss,
                "warmup" if in_warmup else ("transition" if in_transition else "gmm"),
                gmm_weight,
            )
            self.losses_logger.log(epoch, storer)

            # Revive dead components only once the GMM terms are at full
            # strength, and never after the last epoch (the revived centroid
            # would be saved untrained).
            if not in_warmup and gmm_weight >= 1.0 and epoch < epochs - 1:
                self._maybe_reinit_collapsed_components(epoch, usage)

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

    # ------------------------------------------------------------------
    # Usage accounting (no per-batch host syncs)
    # ------------------------------------------------------------------

    def _reset_usage(self) -> None:
        self._usage_sum = None
        self._usage_count = 0

    def _accumulate_usage(self, gamma: torch.Tensor) -> None:
        with torch.no_grad():
            batch_sum = gamma.sum(dim=0)
            if self._usage_sum is None:
                self._usage_sum = batch_sum.clone()
            else:
                self._usage_sum += batch_sum
            self._usage_count += gamma.shape[0]

    def _epoch_usage(self) -> List[float]:
        if self._usage_sum is None or self._usage_count == 0:
            return [0.0] * self.model.gmm_prior.K
        return (self._usage_sum / self._usage_count).cpu().tolist()

    # ------------------------------------------------------------------
    # Dead-component revival
    # ------------------------------------------------------------------

    def _reset_optimizer_state_rows(self, param: torch.nn.Parameter, k: int) -> None:
        """Zero Adam moment estimates for row ``k`` of ``param`` (if present)."""
        state = self.optimizer.state.get(param)
        if not state:
            return
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            buf = state.get(key)
            if buf is not None and buf.shape == param.shape:
                buf[k].zero_()

    def _maybe_reinit_collapsed_components(self, epoch: int, usage: List[float]) -> None:
        """Gradient-free revival of dead GMM components by splitting.

        A component is dead when its epoch-averaged soft usage falls below
        ``collapse_min_usage``.  Each dead component is re-seeded next to one
        of the most-used live components: the centroid is copied with a small
        perturbation, the variance is copied, and the mixing weight of the
        split component is shared equally between the two (a true split, so
        the revived component starts with a usable π instead of ≈0).  Adam
        moment estimates for the overwritten rows are reset.  Several dead
        components are spread across the top live components rather than all
        placed on the same one.
        """
        gmm = self.model.gmm_prior
        K = gmm.K
        threshold = self.collapse_min_usage
        dead = [k for k, u in enumerate(usage) if u < threshold]
        if not dead:
            return

        live_sorted = sorted(
            (k for k in range(K) if k not in dead),
            key=lambda k: usage[k], reverse=True,
        )
        if not live_sorted:
            self.logger.warning(
                "Epoch %d | all %d components below usage %.4f; skipping revival",
                epoch + 1, K, threshold,
            )
            return

        with torch.no_grad():
            D = gmm.D
            for i, k in enumerate(dead):
                best = live_sorted[i % len(live_sorted)]
                sigma_best = float(gmm.sigma_k_scalar[best])
                noise = torch.randn(D, device=gmm.mu_k.device) * sigma_best * self.collapse_noise_scale
                gmm.mu_k.data[k] = gmm.mu_k.data[best] + noise
                gmm.sigma2_raw.data[k] = gmm.sigma2_raw.data[best]
                # Split the mixing weight: π_best → π_best/2 for both.
                gmm.log_pi_unnorm.data[best] -= math.log(2.0)
                gmm.log_pi_unnorm.data[k] = gmm.log_pi_unnorm.data[best]
                half = gmm.rho_ema[best] / 2.0
                gmm.rho_ema[best] = half
                gmm.rho_ema[k] = half
                for p in (gmm.mu_k, gmm.sigma2_raw, gmm.log_pi_unnorm):
                    self._reset_optimizer_state_rows(p, k)
                    self._reset_optimizer_state_rows(p, best)
                self.logger.info(
                    "Epoch %d | component %d dead (usage=%.4f < %.4f); split from component %d",
                    epoch + 1, k, usage[k], threshold, best,
                )

    def _gmm_weight(self, epoch: int) -> float:
        """Transition ramp: 0 during warmup, linear 0→1 during transition, 1 after."""
        if epoch < self.warmup_epochs:
            return 0.0
        t = epoch - self.warmup_epochs
        if self.transition_epochs <= 0:
            return 1.0
        return min(t / self.transition_epochs, 1.0)

    def _maybe_freeze_gmm_prior(self, epoch: int) -> None:
        if self.freeze_gmm_epochs <= 0:
            return
        if epoch < self.warmup_epochs:
            return
        gmm_epoch = epoch - self.warmup_epochs
        should_train = gmm_epoch >= self.freeze_gmm_epochs
        if should_train == self._gmm_prior_trainable:
            return
        for p in self.model.gmm_prior.parameters():
            p.requires_grad_(should_train)
        self._gmm_prior_trainable = should_train

    def _kmeans_init(self, data_loader):
        """Warm-start GMM prior via K-means on fresh encoder means.

        Runs a full forward pass over the entire dataset with the current
        (end-of-warmup) encoder to obtain up-to-date μ representations,
        then calls ``gmm_prior.kmeans_init_``.

        A temporary DataLoader without drop_last / shuffle is created so
        that every feature is seen exactly once, regardless of the training
        loader's batch-drop policy.
        """
        import torch.utils.data as tud
        # Full-coverage, no-shuffle loader over the same dataset
        init_loader = tud.DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size or 256,
            shuffle=False,
            drop_last=False,
        )
        self.logger.info("Computing fresh encoder means for K-means init...")
        self.model.eval()
        mu_parts = []
        with torch.no_grad():
            for batch in init_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                x = x.to(self.device)
                mu, _ = self.model.encode(x)
                mu_parts.append(mu.cpu())
        self.model.train()

        mu_all = torch.cat(mu_parts, dim=0)
        self.logger.info("Running K-means init on %d fresh μ samples...", mu_all.shape[0])
        self.model.gmm_prior.kmeans_init_(mu_all, seed=self.seed)
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
            recon_x, mu, logvar, z, gamma = self.model(x)
        if self.amp:   # losses in float32 for the KL / log-softmax terms
            recon_x, mu, logvar, gamma = recon_x.float(), mu.float(), logvar.float(), gamma.float()

        # Soft usage is accumulated on device and reduced once per epoch
        # (γ from the untrained prior during warmup is not informative).
        if gamma is not None and not in_warmup:
            self._accumulate_usage(gamma.detach())

        if in_warmup:
            loss = self.warmup_loss_f(
                x, recon_x, mu, logvar, storer=storer, epoch=epoch, model=self.model,
            )
        else:
            loss = self.gmm_loss_f(
                x=x, recon_x=recon_x, mu=mu, logvar=logvar, gamma=gamma,
                model=self.model, storer=storer,
                epoch=epoch - self.warmup_epochs,   # phase-relative for KL annealing
                gene_groups=self.gene_groups or None,
                feature_idx=feature_idx,
                gmm_weight=gmm_weight,
                idx_to_gene=self._idx_to_gene or None,
            )

        self.optimizer.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
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

    def __call__(self, data_loader, return_components: bool = False):
        """
        Evaluate the loss at full β (no annealing) with the GMM terms at
        full weight.

        Returns
        -------
        float, or dict when ``return_components=True`` with keys
        ``loss`` (full objective), ``recon``, ``kl`` and ``elbo``
        (= recon + β·kl, i.e. the data-fit part without regularisers).
        """
        self.model.eval()
        storer: Dict[str, list] = defaultdict(list)
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
                    gamma=gamma, model=self.model, storer=storer, epoch=None,
                )
                total_loss += loss.item()

        n = max(len(data_loader), 1)
        mean_loss = total_loss / n
        recon = float(sum(storer.get("recon_loss", [0.0])) / n)
        kl = float(sum(storer.get("kl_loss", [0.0])) / n)
        beta = self.loss_f.get_beta_for_epoch(None)
        elbo = recon + beta * kl
        self.logger.info(
            "Evaluation loss: %.4f | recon=%.4f kl=%.4f elbo=%.4f", mean_loss, recon, kl, elbo,
        )
        if return_components:
            return {"loss": mean_loss, "recon": recon, "kl": kl, "elbo": elbo}
        return mean_loss


# ---------------------------------------------------------------------------
# Loss logger
# ---------------------------------------------------------------------------

class LossesLogger:
    """Write epoch-level training losses to a CSV file."""

    def __init__(self, file_path_name: str, log_level: int = logging.INFO):
        dir_name = os.path.dirname(file_path_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if os.path.isfile(file_path_name):
            try:
                os.remove(file_path_name)
            except FileNotFoundError:
                pass

        self.logger = logging.getLogger(f"losses_logger.{id(self)}")
        self.logger.handlers.clear()
        # Always write the CSV regardless of the application log level.
        self.logger.setLevel(min(int(log_level), logging.INFO))
        self.logger.propagate = False
        fh = logging.FileHandler(file_path_name)
        fh.setLevel(logging.NOTSET)
        self.logger.addHandler(fh)
        self.logger.info("Epoch,Loss,Value")

    def log(self, epoch: int, losses_storer: dict):
        for k, v in losses_storer.items():
            mean_val = sum(v) / len(v)
            self.logger.info(f"{epoch},{k},{mean_val:.6f}")
