from __future__ import annotations

import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as _pkg_version

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

project = "BSVAE"
author = "Kynon J Benjamin"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

try:
    release = _pkg_version("bsvae")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

myst_url_schemes = ("http", "https", "mailto")
myst_heading_anchors = 3

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files: list[str] = []

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Ensure API documentation can resolve package metadata without optional dependencies.
autodoc_mock_imports = []
