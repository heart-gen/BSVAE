"""Simulation utilities for benchmarking GMMModuleVAE."""
from .generate import simulate_omics_data
from .metrics import compute_ari, compute_nmi, benchmark_methods

__all__ = ["simulate_omics_data", "compute_ari", "compute_nmi", "benchmark_methods"]
