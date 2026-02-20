"""BSVAE — Gaussian Mixture VAE for multi-modal biological module discovery."""

from importlib.metadata import PackageNotFoundError, version

from . import latent, networks, simulation

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["networks", "latent", "simulation", "__version__"]
