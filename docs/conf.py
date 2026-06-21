"""Sphinx configuration for karpos-downscaling documentation.

See https://www.sphinx-doc.org/en/master/usage/configuration.html for a
reference of all available options.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path so autodoc can import the package.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "karpos-downscaling"
author = "Loïc Maurin"
copyright = f"{datetime.now().year}, {author}"

# The full version, including alpha/beta/rc tags.
# Kept here as a static string; CI/release process can override at build time.
release = "0.3.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",                      # Markdown support (CommonMark + MyST extensions)
    "sphinx.ext.autodoc",               # API reference from docstrings
    "sphinx.ext.napoleon",              # NumPy and Google style docstrings
    "sphinx.ext.viewcode",              # "View source" links in API pages
    "sphinx.ext.intersphinx",           # Cross-references to other projects
    "sphinx.ext.mathjax",               # Math rendering
    "sphinx_autodoc_typehints",         # Render type hints in docs
    "sphinxcontrib.bibtex",             # BibTeX bibliographies
]

# MyST extensions: math, definition lists, frontmatter HTML, etc.
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "fieldlist",
    "html_image",
    "linkify",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3  # generate anchors for h1..h3

# Source files: both Markdown and reStructuredText accepted.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# List of patterns to ignore when looking for source files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".venv",
    "wandb",
]

# BibTeX configuration
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"

# Autodoc: include both signature in the signature and the docstring.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "both"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Intersphinx mapping — cross-link to common scientific libraries.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "titles_only": False,
}

html_static_path = ["_static"]
html_show_sourcelink = True
html_show_sphinx = False

# Title shown in the browser tab.
html_title = f"{project} v{release}"
