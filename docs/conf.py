"""Sphinx configuration for spherex-photometry."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "spherex-photometry"
author = "Hyeonguk Bahk"
copyright = "2026, Hyeonguk Bahk"

try:
    from spherex_photometry.version import __version__ as release
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

# The engine (tractor-jax / jax) and upstream tractor are heavy optional imports;
# mock them so autodoc can import the backend modules without them installed.
autodoc_mock_imports = ["tractor_jax", "tractor", "jax", "jaxlib",
                        "spherex_retrieval", "dl"]
autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

html_theme = "pydata_sphinx_theme"
html_title = f"spherex-photometry {version}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

exclude_patterns = ["_build"]
