import pytest
import warnings

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from bsvae.networks.module_extraction import format_module_feedback, load_adjacency


def test_format_module_feedback_includes_resolution():
    modules = pd.Series([0] * 80 + [1] * 120, index=[f"g{i}" for i in range(200)])
    message = format_module_feedback("Leiden", modules, resolution=1.0)

    assert message.startswith("Leiden resolution=1.0")
    assert "produced 2 modules" in message
    assert "median size=100 genes" in message


def test_format_module_feedback_includes_n_clusters():
    modules = pd.Series([0, 1, 2, 2], index=["a", "b", "c", "d"])
    message = format_module_feedback("Spectral", modules, n_clusters=3)

    assert message.startswith("Spectral n_clusters=3")
    assert "produced 3 modules" in message
    assert "median size=1 genes" in message


def test_load_adjacency_handles_unnamed_index(tmp_path):
    genes = ["g1", "g2"]
    adjacency = pd.DataFrame([[0.0, 0.3], [0.3, 0.0]], index=genes, columns=genes)
    path = tmp_path / "adjacency.csv"
    adjacency.to_csv(path)

    loaded_adj, loaded_genes = load_adjacency(path.as_posix())

    assert loaded_genes == genes
    assert np.allclose(loaded_adj, adjacency.values)


def test_load_adjacency_parquet(tmp_path):
    """Test loading adjacency from Parquet format."""
    from bsvae.networks.extract_networks import save_edge_list_parquet

    genes = ["gene_a", "gene_b", "gene_c"]
    adjacency = np.array([
        [0.0, 0.5, 0.2],
        [0.5, 0.0, 0.8],
        [0.2, 0.8, 0.0],
    ], dtype=np.float32)

    path = tmp_path / "edges.parquet"
    save_edge_list_parquet(adjacency, path.as_posix(), genes)

    loaded_adj, loaded_genes = load_adjacency(path.as_posix())

    assert loaded_genes == genes
    assert np.allclose(loaded_adj, adjacency, atol=1e-6)


def test_parquet_roundtrip_with_threshold(tmp_path):
    """Test Parquet save/load with threshold filtering."""
    from bsvae.networks.extract_networks import save_edge_list_parquet, load_edge_list_parquet

    genes = ["g1", "g2", "g3", "g4"]
    adjacency = np.array([
        [0.0, 0.9, 0.1, 0.3],
        [0.9, 0.0, 0.2, 0.05],
        [0.1, 0.2, 0.0, 0.8],
        [0.3, 0.05, 0.8, 0.0],
    ], dtype=np.float32)

    path = tmp_path / "edges_filtered.parquet"
    save_edge_list_parquet(adjacency, path.as_posix(), genes, threshold=0.25)

    loaded_adj, loaded_genes = load_edge_list_parquet(path.as_posix())

    assert loaded_genes == genes
    # Only edges >= 0.25 should be kept
    assert loaded_adj[0, 1] == pytest.approx(0.9, abs=1e-6)
    assert loaded_adj[0, 3] == pytest.approx(0.3, abs=1e-6)
    assert loaded_adj[2, 3] == pytest.approx(0.8, abs=1e-6)
    # Edges below threshold should be zero
    assert loaded_adj[0, 2] == 0.0
    assert loaded_adj[1, 3] == 0.0


def test_parquet_compression_options(tmp_path):
    """Test different Parquet compression options."""
    from bsvae.networks.extract_networks import save_edge_list_parquet, load_edge_list_parquet

    genes = ["x", "y", "z"]
    adjacency = np.array([
        [0.0, 0.5, 0.3],
        [0.5, 0.0, 0.7],
        [0.3, 0.7, 0.0],
    ], dtype=np.float32)

    for compression in ["zstd", "snappy", "gzip", None]:
        path = tmp_path / f"edges_{compression}.parquet"
        save_edge_list_parquet(adjacency, path.as_posix(), genes, compression=compression)

        loaded_adj, loaded_genes = load_edge_list_parquet(path.as_posix())
        assert loaded_genes == genes
        assert np.allclose(loaded_adj, adjacency, atol=1e-6)


def test_deprecated_gz_format_warning(tmp_path):
    """Test that loading gzipped format emits deprecation warning."""
    import gzip

    genes = ["a", "b"]
    path = tmp_path / "edges.csv.gz"

    # Create a gzipped edge list
    with gzip.open(path, "wt") as f:
        f.write("source_idx,target_idx,weight\n")
        f.write("0,1,0.5\n")

    # Create gene lookup
    gene_path = tmp_path / "edges_genes.txt"
    with open(gene_path, "w") as f:
        f.write("a\nb\n")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded_adj, loaded_genes = load_adjacency(path.as_posix())

        assert len(w) >= 1
        assert any("deprecated" in str(warning.message).lower() for warning in w)
