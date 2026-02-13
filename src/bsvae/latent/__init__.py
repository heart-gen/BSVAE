"""Latent-space utilities."""

from importlib import import_module

__all__ = ["latent_analysis", "latent_export"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"bsvae.latent.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
