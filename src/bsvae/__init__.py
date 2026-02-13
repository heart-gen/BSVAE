"""BSVAE package entrypoint."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["latent", "networks", "__version__"]


def __getattr__(name):
    if name in {"latent", "networks"}:
        module = import_module(f"bsvae.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
