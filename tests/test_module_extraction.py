import pytest

pd = pytest.importorskip("pandas")

from bsvae.networks.module_extraction import format_module_feedback


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
