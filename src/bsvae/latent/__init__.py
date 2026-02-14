"""Latent-space utilities.

This package keeps backward-compatible convenience re-exports:

- ``export_latents`` (wrapper around :mod:`bsvae.latent.latent_export`)
- ``run_latent_analysis`` (wrapper around :mod:`bsvae.latent.latent_analysis`)
"""

from __future__ import annotations

from importlib import import_module
from typing import Optional

__all__ = [
    "latent_analysis",
    "latent_export",
    "export_latents",
    "run_latent_analysis",
]


def __getattr__(name):
    if name in {"latent_analysis", "latent_export"}:
        module = import_module(f"bsvae.latent.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def export_latents(model, dataloader, output_path: str, format: Optional[str] = None, device=None):
    """Backward-compatible helper to extract and save latents in one call."""
    latent_export = import_module("bsvae.latent.latent_export")
    mu, logvar, sample_ids = latent_export.extract_latents(model, dataloader, device=device)
    latent_export.save_latents(mu, logvar, sample_ids, output_path, format=format)


def run_latent_analysis(
    model,
    dataloader,
    output_dir: str,
    kmeans_k: int = 0,
    gmm_k: int = 0,
    compute_umap: bool = False,
    compute_tsne: bool = False,
    tsne_perplexity: float = 30.0,
    covariates=None,
    device=None,
):
    """Backward-compatible end-to-end latent analysis helper."""
    latent_analysis = import_module("bsvae.latent.latent_analysis")

    mu, logvar, z = latent_analysis.extract_latents(model, dataloader, device=device)
    cluster_labels = None
    embedding = None
    correlation_df = None

    if kmeans_k and kmeans_k > 0:
        cluster_labels = latent_analysis.kmeans_on_mu(mu, k=kmeans_k)
    elif gmm_k and gmm_k > 0:
        cluster_labels, _ = latent_analysis.gmm_on_z(z, n_components=gmm_k)

    if compute_umap:
        embedding = latent_analysis.umap_mu(mu)
    elif compute_tsne:
        embedding = latent_analysis.tsne_mu(mu, perplexity=tsne_perplexity)

    if covariates is not None:
        correlation_df = latent_analysis.correlate_with_covariates(mu, covariates)

    # Extract sample ids from dataloader order if available; else integer ids.
    sample_ids = []
    for _x, ids in dataloader:
        if isinstance(ids, (list, tuple)):
            sample_ids.extend([str(i) for i in ids])
        else:
            sample_ids.append(str(ids))

    if len(sample_ids) != mu.shape[0]:
        sample_ids = [str(i) for i in range(mu.shape[0])]

    latent_analysis.save_latent_results(
        mu,
        logvar,
        sample_ids,
        output_dir,
        cluster_labels=cluster_labels,
        embedding=embedding,
        correlation_df=correlation_df,
    )
