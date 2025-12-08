"""
Utilities for loading Protein–Protein Interaction (PPI) priors.

This module provides reproducible access to STRING-DB networks and
conversion into Laplacian matrices aligned with input genes.
"""

import os
import gzip
import torch
import numpy as np
import pandas as pd
import networkx as nx
import urllib.request
import scipy.sparse as sp

# Default cache path, overridable via env var or function arg
DEFAULT_CACHE_DIR = os.path.expanduser(
    os.getenv("BSVAE_PPI_CACHE", "~/.bsvae/ppi")
)

STRING_URL_TEMPLATE = (
    "https://stringdb-static.org/download/protein.links.detailed.v12.0/"
    "{taxid}.protein.links.detailed.v12.0.txt.gz"
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


def build_graph_from_ppi(edges: pd.DataFrame) -> nx.Graph:
    """
    Build a NetworkX graph from STRING edges.
    """
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row['protein1'], row['protein2'], weight=row["score"])
    return G


def graph_to_laplacian(G: nx.Graph, gene_list: list, sparse: bool = True,
                       dtype: torch.dtype = torch.float32):
    """
    Construct normalized Laplacian aligned to a gene list.

    Parameters
    ----------
    G : nx.Graph
        Graph built from PPI.
    gene_list : list
        Ordered list of gene identifiers (Ensembl IDs recommended).
    sparse : bool
        Whether to return sparse or dense Laplacian.
    dtype : torch.dtype
        Torch dtype for output tensor.
    """
    # Reindex graph adjacency to gene_list order
    A = nx.to_scipy_sparse_array(G, nodelist=gene_list, weight="weight", dtype=np.float32)
    deg = np.array(A.sum(1)).ravel()
    D = sp.diags(deg)
    L = D - A

    if sparse:
        coo = L.tocoo()
        indices = torch.stack(
            (
                torch.as_tensor(coo.row, dtype=torch.long),
                torch.as_tensor(coo.col, dtype=torch.long),
            )
        )
        values = torch.as_tensor(coo.data, dtype=dtype)
        L_torch = torch.sparse_coo_tensor(indices, values, coo.shape)
        return L_torch.coalesce()
    else:
        return torch.tensor(L.toarray(), dtype=dtype)


def load_ppi_laplacian(
        gene_list: list, taxid: str = "9606",
        min_score: int = 700,
        cache_dir: str = DEFAULT_CACHE_DIR,
        sparse: bool = True
):
    """
    Convenience: download STRING → filter → Graph → Laplacian.

    Returns
    -------
    L : torch.Tensor
        Laplacian aligned with `gene_list`.
    G : nx.Graph
        Underlying PPI graph.
    """
    edges = load_string_ppi(taxid=taxid, min_score=min_score,
                            cache_dir=cache_dir)
    G = build_graph_from_ppi(edges)
    L = graph_to_laplacian(G, gene_list, sparse=sparse)
    return L, G
