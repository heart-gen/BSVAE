"""Network extraction utilities.

This package preserves backward-compatible helper re-exports from
``extract_networks``, ``module_extraction``, and ``utils`` while still lazily
loading submodules.
"""

from __future__ import annotations

from importlib import import_module

_SUBMODULES = {
    "extract_networks": "bsvae.networks.extract_networks",
    "module_extraction": "bsvae.networks.module_extraction",
    "utils": "bsvae.networks.utils",
    "cli": "bsvae.networks.cli",
}

_HELPER_MODULES = {
    "bsvae.networks.extract_networks",
    "bsvae.networks.module_extraction",
    "bsvae.networks.utils",
}

# Backward-compatible explicit exports for submodules.
__all__ = list(_SUBMODULES.keys())


def _load_module(module_path: str):
    module = import_module(module_path)
    return module


def __getattr__(name):
    # Submodule access (e.g., bsvae.networks.extract_networks)
    if name in _SUBMODULES:
        module = _load_module(_SUBMODULES[name])
        globals()[name] = module
        return module

    # Helper re-exports (e.g., from bsvae.networks import run_extraction)
    for module_path in _HELPER_MODULES:
        module = _load_module(module_path)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    names = set(globals().keys())
    names.update(_SUBMODULES.keys())
    for module_path in _HELPER_MODULES:
        module = _load_module(module_path)
        names.update(getattr(module, "__all__", []))
    return sorted(names)
