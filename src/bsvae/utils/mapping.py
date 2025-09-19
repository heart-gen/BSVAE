"""
Gene annotation utilities.

Provides ID mapping between gene symbols, Ensembl gene IDs, and Ensembl protein IDs
(ENSP) for compatibility with STRING PPI networks.

Default: uses MyGene.info (fast, cached API).
Optional fallback: BioMart (for exact Ensembl release reproducibility).
"""
import os
import logging
import pandas as pd

import mygene

# Optional biomart fallback
try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

def fetch_gene_mapping(gene_list, species="human", source="mygene"):
    """
    Map input genes to Ensembl IDs and protein IDs.

    Parameters
    ----------
    gene_list : list of str
        Input genes (symbols or ENSG IDs).
    species : str or int
        Species name (e.g., "human", "mouse") or NCBI taxid (e.g., 9606).
    source : {"mygene", "biomart"}
        Which service to use. Default: "mygene".

    Returns
    -------
    df : pd.DataFrame
        Columns: input_id, symbol, ensembl_gene, ensembl_protein
    """
    if source == "mygene":
        return _fetch_with_mygene(gene_list, species)
    elif source == "biomart":
        if requests is None:
            raise ImportError("requests not installed; required for BioMart fallback.")
        return _fetch_with_biomart(gene_list, species)
    else:
        raise ValueError("source must be 'mygene' or 'biomart'")


def _fetch_with_mygene(gene_list, species):
    """Use MyGene.info to fetch gene → ENSG/ENSP mapping."""
    mg = mygene.MyGeneInfo()
    logger.info(f"Querying MyGene.info for {len(gene_list)} genes (species={species})")

    try:
        out = mg.querymany(
            gene_list,
            scopes=["symbol", "ensembl.gene"],
            fields="symbol,ensembl.gene,ensembl.protein",
            species=species,
            as_dataframe=True
        )
    except Exception as e:
        logger.error(f"MyGene query failed: {e}")
        raise RuntimeError("Failed to fetch data from MyGene.info")

    out = out.reset_index().rename(columns={"query": "input_id"})
    # Handle nested ensembl dicts
    if "ensembl" in out.columns:
        def extract_gene(x):
            if isinstance(x, list):
                if len(x) > 1:
                    logger.warning(f"Multiple mappings found: using first entry: {x}")
                return x[0].get("gene")
            if isinstance(x, dict):
                return x.get("gene")
            return None

        def extract_protein(x):
            if isinstance(x, list):
                if len(x) > 1:
                    logger.warning(f"Multiple protein mappings found: using first: {x}")
                return x[0].get("protein")
            if isinstance(x, dict):
                return x.get("protein")
            return None

        out["ensembl_gene"] = out["ensembl"].map(extract_gene)
        out["ensembl_protein"] = out["ensembl"].map(extract_protein)
        out = out.drop(columns=["ensembl"])

    # Report unmatched queries
    unmatched = out[out["_id"].isna()]
    if not unmatched.empty:
        logger.warning(f"{len(unmatched)} input genes could not be matched.")

    return out[["input_id", "symbol", "ensembl_gene", "ensembl_protein"]]


def _fetch_with_biomart(gene_list, species):
    """Query Ensembl BioMart REST API."""
    logger.info(f"Querying BioMart for {len(gene_list)} genes (species={species})")

    # Map species to dataset name
    species_map = {
        "human": "hsapiens_gene_ensembl",
        "mouse": "mmusculus_gene_ensembl",
        "rat": "rnorvegicus_gene_ensembl",
        "fly": "dmelanogaster_gene_ensembl",
    }
    dataset = species_map.get(str(species).lower(), "hsapiens_gene_ensembl")

    url = "https://www.ensembl.org/biomart/martservice"
    xml_query = f"""
    <!DOCTYPE Query>
    <Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count=""
           datasetConfigVersion="0.6">
      <Dataset name="{dataset}" interface="default">
        <Filter name="external_gene_name" value="{','.join(gene_list)}"/>
        <Attribute name="external_gene_name" />
        <Attribute name="ensembl_gene_id" />
        <Attribute name="ensembl_peptide_id" />
      </Dataset>
    </Query>
    """

    try:
        r = requests.get(url, params={"query": xml_query})
        r.raise_for_status()
    except Exception as e:
        logger.error(f"BioMart request failed: {e}")
        raise RuntimeError("Failed to fetch data from BioMart")

    from io import StringIO
    df = pd.read_csv(StringIO(r.text), sep="\t")

    expected_cols = ["Gene name", "Gene stable ID", "Protein stable ID"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Unexpected BioMart output columns: {df.columns}")

    df = df.rename(columns={
        "Gene name": "symbol",
        "Gene stable ID": "ensembl_gene",
        "Protein stable ID": "ensembl_protein"
    })
    df["input_id"] = df["symbol"]

    return df[["input_id", "symbol", "ensembl_gene", "ensembl_protein"]]


def map_genes_to_string(
        gene_list: list, annotation: pd.DataFrame,
        id_type: str = "ENSG", ensp_col: str = "ENSP") -> pd.DataFrame:
    """
    Map input genes to STRING protein IDs (ENSP).

    Parameters
    ----------
    gene_list : list of str
        Genes from your dataset (must match id_type).
    annotation : pd.DataFrame
        Annotation table with columns including id_type and ensp_col.
    id_type : {"ENSG", "symbol"}
        Type of gene IDs provided in gene_list.
    ensp_col : str
        Column name in annotation for STRING IDs (default: "ENSP").

    Returns
    -------
    mapping : pd.DataFrame
        Columns: [gene_id, ensp_col]
        Only includes genes that map to STRING.
    """
    if id_type not in annotation.columns or ensp_col not in annotation.columns:
        raise ValueError(f"Annotation must contain columns: {id_type}, {ensp_col}")

    df = pd.DataFrame({id_type: gene_list})
    merged = df.merge(annotation, how="left", on=id_type)

    mapping = merged[[id_type, ensp_col]].dropna().drop_duplicates()
    mapping = mapping.rename(columns={id_type: "gene_id"})

    return mapping


def subset_genes_and_laplacian(edges: pd.DataFrame, mapping: pd.DataFrame,
                               ensp_col: str = "ENSP"):
    """
    Subset PPI edges and gene list to only mapped genes.

    Parameters
    ----------
    edges : pd.DataFrame
        STRING edges: ["protein1", "protein2", "score"]
    mapping : pd.DataFrame
        Gene → ENSP mapping: ["gene_id", ensp_col]
    ensp_col : str
        Column in `mapping` that corresponds to STRING protein IDs.

    Returns
    -------
    edges_sub : pd.DataFrame
        Filtered PPI edges with only proteins in mapping.
    gene_list : list
        List of gene IDs aligned to rows/cols of Laplacian.
    """
    if ensp_col not in mapping.columns:
        raise ValueError(f"Column '{ensp_col}' not found in mapping.")

    ensps = set(mapping[ensp_col])
    edges_sub = edges[
        edges["protein1"].isin(ensps) & edges["protein2"].isin(ensps)
    ].copy()

    gene_list = mapping["gene_id"].tolist()
    return edges_sub, gene_list
