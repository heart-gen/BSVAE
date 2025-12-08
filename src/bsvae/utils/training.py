import os
import logging
from tqdm import trange
from timeit import default_timer
from collections import defaultdict

import torch

TRAIN_LOSSES_LOGFILE = "train_losses.csv"


class Trainer:
    """
    Trainer for StructuredFactorVAE models.

    Parameters
    ----------
    model : nn.Module
        The VAE model (StructuredFactorVAE).
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    loss_f : nn.Module
        Loss function (e.g., BaseLoss).
    device : torch.device
        Device to run training on.
    logger : logging.Logger
        Logger for output messages.
    save_dir : str
        Directory to save checkpoints and logs.
    is_progress_bar : bool
        Show tqdm progress bar.
    """

    def __init__(self, model, optimizer, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.model = model.to(self.device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger = logger
        self.losses_logger = LossesLogger(
            os.path.join(self.save_dir, TRAIN_LOSSES_LOGFILE),
            log_level=self.logger.level,
        )
        self.logger.info(f"Training Device: {self.device}")

    def __call__(self, data_loader, epochs=10, checkpoint_every=10):
        start = default_timer()
        self.model.train()

        for epoch in range(epochs):
            storer = defaultdict(list)
            mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
            self.logger.info(f"Epoch {epoch + 1} | Avg loss: {mean_epoch_loss:.4f}")
            self.losses_logger.log(epoch, storer)

            if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.save_dir, f"model-{epoch+1}.pt"))

        self.model.eval()
        delta_time = (default_timer() - start) / 60
        self.logger.info(f"Finished training after {delta_time:.1f} min.")

    def _train_epoch(self, data_loader, storer, epoch):
        epoch_loss = 0.0
        with trange(len(data_loader), desc=f"Epoch {epoch+1}",
                    leave=False, disable=not self.is_progress_bar) as t:
            for data in data_loader:
                # for gene expression we expect just tensors (not (x,y))
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(self.device)

                iter_loss = self._train_iteration(data, storer)
                epoch_loss += iter_loss
                t.set_postfix(loss=iter_loss)
                t.update()

        return epoch_loss / len(data_loader)

    def _train_iteration(self, x, storer):
        x = x.to(self.device)
        if hasattr(self.loss_f, "call_optimize"):
            loss = self.loss_f.call_optimize(data, self.model,
                                             self.optimizer, storer)
        else:
            recon_x, mu, logvar, z, log_var = self.model(x)
            loss = self.loss_f(
                x, recon_x, mu, logvar, self.model,
                L=getattr(self.model, "laplacian_matrix", None),
                storer=storer, is_train=self.model.training
            )

            if hasattr(self.model, "l1_strength"):
                storer.setdefault("l1_strength", []).append(self.model.l1_strength)
            if hasattr(self.model, "lap_strength"):
                storer.setdefault("lap_strength", []).append(self.model.lap_strength)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class LossesLogger:
    """Write training losses to log file."""

    def __init__(self, file_path_name, log_level=logging.DEBUG):
        dir_name = os.path.dirname(file_path_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if os.path.isfile(file_path_name):
            try:
                os.remove(file_path_name)
            except FileNotFoundError:
                # The file might disappear between the check and removal
                # (e.g., when multiple runs share the same directory).
                pass

        self.logger = logging.getLogger("losses_logger")
        self.logger.handlers.clear()
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(logging.NOTSET)
        self.logger.addHandler(file_handler)
        self.logger.info("Epoch,Loss,Value")

    def log(self, epoch, losses_storer):
        for k, v in losses_storer.items():
            mean_val = sum(v) / len(v)
            self.logger.info(f"{epoch},{k},{mean_val}")
