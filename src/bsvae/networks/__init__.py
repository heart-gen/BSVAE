"""Network extraction utilities."""

from importlib import import_module

__all__ = ["extract_networks", "module_extraction", "utils", "cli"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"bsvae.networks.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
