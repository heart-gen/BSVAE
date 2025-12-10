"""BSVAE package entrypoint."""

from importlib.metadata import PackageNotFoundError, version

from . import latent, networks

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["networks", "latent", "__version__"]
