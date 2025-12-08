import os
import logging
from collections import defaultdict
from timeit import default_timer
from tqdm import tqdm

import torch

TEST_LOSSES_FILE = "test_losses.pt"


class Evaluator:
    """
    Evaluator for StructuredFactorVAE.

    Computes reconstruction + KL + penalties on test data.
    """

    def __init__(self, model, loss_f,
                 device=torch.device("cpu"),
                 logger=logging.getLogger(__name__),
                 save_dir="results",
                 is_progress_bar=True):

        self.device = device
        self.loss_f = loss_f
        self.model = model.to(self.device)
        self.logger = logger
        self.save_dir = save_dir
        self.is_progress_bar = is_progress_bar
        self.logger.info(f"Testing Device: {self.device}")

    def __call__(self, data_loader, is_losses=True):
        start = default_timer()
        is_still_training = self.model.training
        self.model.eval()

        losses = None
        if is_losses:
            self.logger.info("Computing test losses...")
            losses = self.compute_losses(data_loader)
            self.logger.info(f"Test Losses: {losses}")
            torch.save(losses, os.path.join(self.save_dir, TEST_LOSSES_FILE))

        if is_still_training:
            self.model.train()

        compute_time = ((default_timer() - start) / 60)
        self.logger.info(f"Finished evaluating after {compute_time:.1f} min.")

        return losses

    def compute_losses(self, dataloader):
        storer = defaultdict(list)
        with torch.no_grad():
            for data in tqdm(dataloader, leave=False,
                             disable=not self.is_progress_bar):
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(self.device)

                if hasattr(self.loss_f, "call_optimize"):
                    _ = self.loss_f.call_optimize(data, self.model, None, storer)
                else:
                    recon_batch, mu, logvar, z, log_var = self.model(data)
                    _ = self.loss_f(data, recon_batch, mu, logvar, self.model,
                                    L=getattr(self.model, "laplacian_matrix", None),
                                    is_train=self.model.training,
                                    storer=storer)

                    if hasattr(self.model, "l1_strength"):
                        storer.setdefault("l1_strength", []).append(self.model.l1_strength)
                    if hasattr(self.model, "lap_strength"):
                        storer.setdefault("lap_strength", []).append(self.model.lap_strength)

        losses = {k: sum(v) / len(v) for k, v in storer.items()}
        return losses
