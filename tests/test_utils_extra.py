import io
import gzip
import torch
import pandas as pd
import pytest

import bsvae.utils.mapping as mapping
import bsvae.utils.ppi as ppi


# ---------------------------
# mapping.py
# ---------------------------

def test_fetch_with_mygene_dict_and_list(monkeypatch):
    class DummyMG:
        def querymany(self, *a, **k):
            # Return df with both dict and list ensembl formats
            return pd.DataFrame({
                "query": ["g1", "g2"],
                "symbol": ["sym1", "sym2"],
                "ensembl": [
                    {"gene": "E1", "protein": "P1"},
                    [{"gene": "E2", "protein": "P2"}],
                ],
                "_id": ["id1", "id2"]
            }).set_index("query")

    monkeypatch.setattr(mapping.mygene, "MyGeneInfo", lambda: DummyMG())
    df = mapping._fetch_with_mygene(["g1", "g2"], "human")
    assert set(df.columns) == {"input_id", "symbol", "ensembl_gene", "ensembl_protein"}
    assert "E1" in df["ensembl_gene"].values


def test_fetch_with_mygene_unmatched(monkeypatch, caplog):
    class DummyMG:
        def querymany(self, *a, **k):
            return pd.DataFrame({
                "query": ["g1"],
                "symbol": ["sym1"],
                "ensembl": [None],
                "_id": [None]  # triggers unmatched warning
            }).set_index("query")

    monkeypatch.setattr(mapping.mygene, "MyGeneInfo", lambda: DummyMG())
    df = mapping._fetch_with_mygene(["g1"], "human")
    assert df["ensembl_gene"].isna().all()
    assert any("could not be matched" in m for m in caplog.text.splitlines())


def test_fetch_with_biomart_success(monkeypatch):
    fake_tsv = "Gene name\tGene stable ID\tProtein stable ID\nSYM\tE1\tP1\n"
    class DummyResp:
        text = fake_tsv
        def raise_for_status(self): return None
    monkeypatch.setattr(mapping, "requests", type("R", (), {"get": lambda *a, **k: DummyResp()})())
    df = mapping._fetch_with_biomart(["SYM"], "human")
    assert list(df.columns) == ["input_id", "symbol", "ensembl_gene", "ensembl_protein"]
    assert df.iloc[0]["ensembl_gene"] == "E1"


def test_fetch_with_biomart_bad_columns(monkeypatch):
    fake_tsv = "wrongcol1\twrongcol2\nA\tB\n"
    class DummyResp:
        text = fake_tsv
        def raise_for_status(self): return None
    monkeypatch.setattr(mapping, "requests", type("R", (), {"get": lambda *a, **k: DummyResp()})())
    with pytest.raises(ValueError):
        mapping._fetch_with_biomart(["SYM"], "human")


def test_fetch_gene_mapping_invalid_source():
    with pytest.raises(ValueError):
        mapping.fetch_gene_mapping(["g1"], source="invalid")


# ---------------------------
# ppi.py
# ---------------------------

def test_download_string_and_load(monkeypatch, tmp_path):
    # Patch urlretrieve to just write a dummy gzip file
    dummy_path = tmp_path / "9606_string.txt.gz"
    def fake_urlretrieve(url, filename):
        with gzip.open(filename, "wt") as f:
            f.write("protein1 protein2 combined_score\n9606.A 9606.B 900\n")
    monkeypatch.setattr(ppi.urllib.request, "urlretrieve", fake_urlretrieve)

    # Should not raise
    path = ppi.download_string("9606", cache_dir=tmp_path)
    assert path.endswith(".gz")

    # Now load it
    edges = ppi.load_string_ppi("9606", min_score=800, cache_dir=tmp_path)
    assert list(edges.columns) == ["protein1", "protein2", "score"]
    assert edges.iloc[0]["protein1"] == "A"


def test_build_graph_from_ppi_and_laplacian():
    edges = pd.DataFrame({"protein1": ["A"], "protein2": ["B"], "score": [1.0]})
    G = ppi.build_graph_from_ppi(edges)
    assert G.has_edge("A", "B")

    # Dense Laplacian
    L = ppi.graph_to_laplacian(G, ["A", "B"], sparse=False)
    assert isinstance(L, torch.Tensor)
    assert L.shape == (2, 2)

    # Sparse Laplacian
    Ls = ppi.graph_to_laplacian(G, ["A", "B"], sparse=True)
    assert Ls.is_sparse


def test_load_ppi_laplacian(monkeypatch):
    fake_edges = pd.DataFrame({"protein1": ["A"], "protein2": ["B"], "score": [1.0]})
    monkeypatch.setattr(ppi, "load_string_ppi", lambda *a, **k: fake_edges)
    L, G = ppi.load_ppi_laplacian(["A", "B"])
    assert isinstance(L, torch.Tensor)
    assert G.number_of_nodes() == 2
