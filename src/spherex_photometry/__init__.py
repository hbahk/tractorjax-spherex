"""spherex-photometry: forced photometry and spectrophotometry from SPHEREx L2.

Typical use::

    from spherex_photometry import PhotometryConfig, run_photometry, build_spectra

    cfg = PhotometryConfig(solver="eigfloor")          # blind-production default
    phot = run_photometry("cutouts/", "catalog.parquet", cfg, output="phot.parquet")
    spectra = build_spectra(phot)                      # id -> per-source spectrum

The heavy engine (``tractor_jax``) and the cutout downloader
(``spherex_retrieval``) are separate packages; see the docs for installation.
"""

from __future__ import annotations

from .config import ConfigError, PhotometryConfig
from .pipeline import run_photometry
from .spectra import bin_spectrum, build_spectra, to_ab_mag
from .version import __version__

__all__ = [
    "PhotometryConfig", "ConfigError",
    "run_photometry",
    "build_spectra", "bin_spectrum", "to_ab_mag",
    "fetch_ls_dr10", "retrieve",
    "__version__",
]


def fetch_ls_dr10(*args, **kwargs):
    """Lazy wrapper for :func:`spherex_photometry.catalog.fetch_ls.fetch_ls_dr10`."""
    from .catalog.fetch_ls import fetch_ls_dr10 as _f
    return _f(*args, **kwargs)


def retrieve(*args, **kwargs):
    """Thin passthrough to ``spherex_retrieval.retrieve`` (download L2 cutouts)."""
    try:
        from spherex_retrieval import retrieve as _r
    except ImportError as exc:  # pragma: no cover - optional-ish dep
        raise ImportError(
            "Downloading cutouts needs the spherex-retrieval package "
            "(not on PyPI yet):\n"
            "    pip install git+https://github.com/hbahk/spherex-retrieval"
        ) from exc
    return _r(*args, **kwargs)
