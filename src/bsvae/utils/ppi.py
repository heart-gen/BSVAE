"""
Utilities for loading Protein–Protein Interaction (PPI) priors.

This module provides reproducible access to STRING-DB networks and
conversion into Laplacian matrices aligned with input genes.
"""

import os
import gzip
import numpy as np
import pandas as pd
import urllib.request
import scipy.sparse as sp

# Default cache path, overridable via env var or function arg
DEFAULT_CACHE_DIR = os.path.expanduser(
    os.getenv("BSVAE_PPI_CACHE", "~/.bsvae/ppi")
)

STRING_URL_TEMPLATE = (
    "https://stringdb-static.org/download/protein.links.detailed.v12.0/{taxid}.protein.links.detailed.v12.0.txt.gz"
)


def download_string(taxid: str = "9606", cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """
    Download STRING PPI for a given species if not cached.

    Parameters
    ----------
    taxid : str
        NCBI taxonomy ID (default: "9606" = human).
    cache_dir : str
        Local cache directory (default: ~/.bsvae/ppi or BSVAE_PPI_CACHE).

    Returns
    -------
    filepath : str
        Path to downloaded STRING file.
    """
    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"{taxid}_string.txt.gz")
    url = STRING_URL_TEMPLATE.format(taxid=taxid)

    if not os.path.exists(filename):
        print(f"Downloading STRING PPI for {taxid}...")
        urllib.request.urlretrieve(url, filename)

    return filename


def load_string_ppi(
        taxid: str = "9606", min_score: int = 700,
        cache_dir: str = DEFAULT_CACHE_DIR
) -> pd.DataFrame:
    """
    Load STRING PPI edges for a given taxonomy ID.

    Parameters
    ----------
    taxid : str
        NCBI taxonomy ID (9606=human, 10090=mouse, 10116=rat, 7227=fly, etc).
    min_score : int
        Minimum combined score (default: 700 = high confidence).
    cache_dir : str
        Directory for cached STRING data.

    Returns
    -------
    edges : pd.DataFrame
        Columns: [protein1, protein2, score]
    """
    fpath = download_string(taxid, cache_dir)
    with gzip.open(fpath, "rt") as f:
        df = pd.read_csv(f, sep=" ")

    edges = df[["protein1", "protein2", "combined_score"]]\
        .rename(columns={"combined_score": "score"})
    edges = edges[edges["score"] >= min_score].copy()

    # Remove taxid prefix ("9606.ENSP...")
    edges["protein1"] = edges["protein1"].str.split(".").str[-1]
    edges["protein2"] = edges["protein2"].str.split(".").str[-1]

    return edges


def make_laplacian(edges: pd.DataFrame, gene_list: list) -> sp.csr_matrix:
    """
    Construct normalized Laplacian aligned to a gene list.

    Parameters
    ----------
    edges : pd.DataFrame
        Columns: [protein1, protein2, score]
    gene_list : list
        Ordered list of gene identifiers (Ensembl IDs recommended).

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Laplacian (G x G).
    """
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    G = len(gene_list)

    row, col, data = [], [], []
    for _, (g1, g2, w) in edges.iterrows():
        if g1 in gene_to_idx and g2 in gene_to_idx:
            i, j = gene_to_idx[g1], gene_to_idx[g2]
            row.extend([i, j])
            col.extend([j, i])
            data.extend([w, w])

    A = sp.csr_matrix((data, (row, col)), shape=(G, G))

    deg = np.array(A.sum(1)).ravel()
    D = sp.diags(deg)
    L = D - A
    return L


def load_ppi_laplacian(
        gene_list: list, taxid: str = "9606",
        min_score: int = 700,
        cache_dir: str = DEFAULT_CACHE_DIR
) -> sp.csr_matrix:
    """
    Convenience: download STRING → filter → Laplacian.

    Parameters
    ----------
    gene_list : list
        Genes in training data.
    taxid : str
        NCBI taxonomy ID (default: "9606"=human).
    min_score : int
        Minimum edge score (default: 700).
    cache_dir : str
        Local cache directory.

    Returns
    -------
    L : scipy.sparse.csr_matrix
        Laplacian aligned with `gene_list`.
    """
    edges = load_string_ppi(taxid=taxid, min_score=min_score,
                            cache_dir=cache_dir)
    return make_laplacian(edges, gene_list)
