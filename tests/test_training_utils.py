import logging
import sys
import types
from pathlib import Path


dummy_tqdm_module = types.ModuleType("tqdm")
dummy_tqdm_module.tqdm = lambda *a, **k: a[0] if a else range(0)
dummy_tqdm_module.trange = lambda *a, **k: range(0)

dummy_torch_module = types.ModuleType("torch")
dummy_torch_module.autograd = types.SimpleNamespace(set_detect_anomaly=lambda *a, **k: None)
dummy_torch_module.device = lambda *a, **k: "cpu"

sys.modules.setdefault("tqdm", dummy_tqdm_module)
sys.modules.setdefault("torch", dummy_torch_module)

import bsvae.utils.training as training


def test_losses_logger_handles_missing_file(tmp_path, monkeypatch):
    log_path = tmp_path / "runs" / training.TRAIN_LOSSES_LOGFILE
    removed = False

    def fake_remove(path):
        nonlocal removed
        removed = True
        raise FileNotFoundError

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stale")

    monkeypatch.setattr(training.os, "remove", fake_remove)

    logger = training.LossesLogger(str(log_path))
    logger.log(0, {"loss": [1.0, 3.0]})

    contents = Path(log_path).read_text()
    assert removed is True
    assert "loss" in contents


def test_losses_logger_creates_parent_directories(tmp_path):
    log_path = tmp_path / "nested" / "deeper" / training.TRAIN_LOSSES_LOGFILE

    logger = training.LossesLogger(str(log_path))
    logger.log(1, {"loss": [2.0]})

    assert log_path.exists()
    assert "loss" in log_path.read_text()


# Silence noisy handlers that might persist across tests
logging.getLogger("losses_logger").handlers.clear()
