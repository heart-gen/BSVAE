"""Sample-level latent space analysis utilities.

The functions here operate on encoder outputs (``mu``/``logvar``) to enable
clustering, visualization, and covariate correlation analyses.

Example
-------
>>> from bsvae.latent.latent_analysis import extract_latents, kmeans_on_mu, umap_mu
>>> mu, logvar, z = extract_latents(model, dataloader)
>>> labels = kmeans_on_mu(mu, k=5)
>>> embedding = umap_mu(mu)
>>> sample_ids = [...]  # identifiers aligned to the dataloader order
>>> save_latent_results(mu, logvar, sample_ids, "results/", cluster_labels=labels, embedding=embedding)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def extract_latents(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    disable_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract latent statistics for all samples.

    Parameters
    ----------
    model:
        Trained model with an ``encoder`` returning ``(mu, logvar)``.
    dataloader:
        Iterable yielding ``(expression, sample_id)`` pairs.
    device:
        Torch device; defaults to CUDA when available.
    disable_progress:
        Placeholder for future progress bar support.

    Returns
    -------
    mu : np.ndarray
        Posterior means of shape ``(n_samples, K)``.
    logvar : np.ndarray
        Posterior log-variances of shape ``(n_samples, K)``.
    z_samples : np.ndarray
        Reparameterized latent samples of shape ``(n_samples, K)``.
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    mu_list = []
    logvar_list = []
    z_list = []

    with torch.no_grad():
        for batch, _ids in dataloader:
            batch = batch.to(device)
            mu, logvar = model.encoder(batch)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
            z_list.append(z.cpu().numpy())

    mu_arr = np.concatenate(mu_list, axis=0)
    logvar_arr = np.concatenate(logvar_list, axis=0)
    z_arr = np.concatenate(z_list, axis=0)
    logger.info("Extracted latents for %d samples", mu_arr.shape[0])
    return mu_arr, logvar_arr, z_arr


def kmeans_on_mu(mu: np.ndarray, k: int = 10) -> np.ndarray:
    """Run K-means clustering on latent means."""

    logger.info("Running K-means with k=%d", k)
    model = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = model.fit_predict(mu)
    return labels


def gmm_on_z(z: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, GaussianMixture]:
    """Fit a Gaussian mixture model to sampled latents."""

    logger.info("Fitting GaussianMixture with %d components", n_components)
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0)
    gmm.fit(z)
    labels = gmm.predict(z)
    return labels, gmm


def umap_mu(mu: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> pd.DataFrame:
    """Compute a 2D UMAP embedding of latent means."""

    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("umap-learn is required for UMAP embeddings") from exc

    logger.info("Running UMAP on %d samples", mu.shape[0])
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    embedding = reducer.fit_transform(mu)
    return pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])


def tsne_mu(mu: np.ndarray, perplexity: float = 30.0) -> pd.DataFrame:
    """Compute a 2D t-SNE embedding of latent means."""

    logger.info("Running t-SNE on %d samples", mu.shape[0])
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0, init="pca")
    embedding = tsne.fit_transform(mu)
    return pd.DataFrame(embedding, columns=["TSNE1", "TSNE2"])


def correlate_with_covariates(mu: np.ndarray | pd.DataFrame, cov_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations between latents and covariates."""

    if isinstance(mu, pd.DataFrame):
        mu_df = mu.copy()
    else:
        mu_df = pd.DataFrame(mu, index=cov_df.index)

    if mu_df.shape[0] != cov_df.shape[0]:
        raise ValueError("mu and covariates must have the same number of samples")

    results = []
    for latent_name in mu_df.columns:
        latent_series = mu_df[latent_name]
        for cov in cov_df.columns:
            cov_series = cov_df[cov]
            if not np.issubdtype(cov_series.dtype, np.number):
                cov_numeric, _ = pd.factorize(cov_series)
                cov_series = pd.Series(cov_numeric, index=cov_series.index)
            paired = pd.concat([latent_series, cov_series], axis=1).dropna()
            if paired.shape[0] < 2:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
                spearman_r, spearman_p = stats.spearmanr(paired.iloc[:, 0], paired.iloc[:, 1])
            results.append(
                {
                    "latent": latent_name,
                    "covariate": cov,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                }
            )
    result_df = pd.DataFrame(results)
    logger.info("Computed correlations for %d latents against %d covariates", mu_df.shape[1], cov_df.shape[1])
    return result_df


def save_latent_results(
    mu: np.ndarray,
    logvar: np.ndarray,
    sample_ids: Sequence[str],
    output_dir: str,
    cluster_labels: Optional[Sequence[int]] = None,
    embedding: Optional[pd.DataFrame] = None,
    correlation_df: Optional[pd.DataFrame] = None,
) -> None:
    """Save latent statistics, clusters, embeddings, and correlations to CSV files."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mu_df = pd.DataFrame(mu, index=sample_ids)
    mu_df.index.name = "sample_id"
    mu_df.to_csv(out_path / "latent_mu.csv")

    logvar_df = pd.DataFrame(logvar, index=sample_ids)
    logvar_df.index.name = "sample_id"
    logvar_df.to_csv(out_path / "latent_logvar.csv")

    if cluster_labels is not None:
        cluster_series = pd.Series(cluster_labels, index=sample_ids, name="cluster")
        cluster_series.to_csv(out_path / "latent_clusters.csv")
        logger.info("Saved cluster assignments for %d samples", len(cluster_series))

    if embedding is not None:
        embed_df = embedding.copy()
        embed_df.index = sample_ids
        embed_df.index.name = "sample_id"
        embed_df.to_csv(out_path / "latent_embeddings.csv")
        logger.info("Saved latent embeddings with shape %s", embed_df.shape)

    if correlation_df is not None:
        correlation_df.to_csv(out_path / "latent_covariate_correlations.csv", index=False)
        logger.info("Saved covariate correlations")


__all__ = [
    "extract_latents",
    "kmeans_on_mu",
    "gmm_on_z",
    "umap_mu",
    "tsne_mu",
    "correlate_with_covariates",
    "save_latent_results",
]
