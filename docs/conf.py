"""Sphinx configuration for tractorjax-spherex."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "tractorjax-spherex"
author = "Hyeonguk Bahk"
copyright = "2026, Hyeonguk Bahk"

try:
    from tractorjax_spherex.version import __version__ as release
except Exception:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # myst_nb is a superset of myst_parser: it renders the stored outputs of
    # the example notebooks (docs/cluster_example.ipynb) as documentation.
    "myst_nb",
    "sphinx_copybutton",
]

# The notebooks need IRSA, Data Lab and a GPU; the docs build must never try
# to run them. Their committed outputs are the documentation.
nb_execution_mode = "off"
nb_merge_streams = True

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
html_title = f"tractorjax-spherex {version}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

exclude_patterns = ["_build"]

html_extra_path = ["googlee20a25095441ea75.html"]