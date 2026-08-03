"""
Sphinx configuration for gridfoam.

Docstrings/comments in this file are intentionally kept minimal and organized
in sections for readability.
"""

from __future__ import annotations

import os
import pathlib
import sys

# Disable runtime type checks during documentation imports.
os.environ.setdefault("GRIDFOAM_RUNTIME_TYPE_CHECKS", "0")

# -- Paths -------------------------------------------------------------------
# Repository root (conf.py lives in docs/source/).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS_SOURCE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_SOURCE_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as _version

project = "gridfoam"
author = "RICOS"
copyright = "2026, RICOS"
version = release = _version("gridfoam")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    # Docstring style
    "numpydoc",
    # UX
    "sphinx_copybutton",
    # Diagrams
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

source_suffix = {
    ".rst": "restructuredtext",
}

# Autosummary / Numpydoc -----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
numpydoc_show_class_members = False
autodoc_default_options = {
    "members": False,
    "inherited-members": False,
}

# -- Options for HTML output -------------------------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html
html_theme = "pydata_sphinx_theme"
html_title = "gridfoam"

html_logo = "_static/logo.webp"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# PyData theme options (minimal defaults; expand as needed).
html_theme_options = {
    "navbar_align": "content",
    "logo": {
        "text": "gridfoam",
        "image_light": "_static/logo.webp",
    },
}

# -- Copybutton --------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Mermaid -----------------------------------------------------------------
# Render Mermaid in HTML (JS). For non-HTML builders, use mermaid-cli.
mermaid_output_format = "raw"
