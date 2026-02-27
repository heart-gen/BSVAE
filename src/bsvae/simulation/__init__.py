"""Simulation utilities for benchmarking GMMModuleVAE."""
from .generate import simulate_omics_data
from .metrics import compute_ari, compute_nmi, benchmark_methods
from .scenario import (
    expand_scenarios,
    generate_grid,
    generate_scenario,
    load_config,
    validate_grid,
    write_starter_config,
)

__all__ = [
    "simulate_omics_data",
    "compute_ari",
    "compute_nmi",
    "benchmark_methods",
    "load_config",
    "write_starter_config",
    "expand_scenarios",
    "generate_scenario",
    "generate_grid",
    "validate_grid",
]
