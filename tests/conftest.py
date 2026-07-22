"""Shared pytest fixtures. Forces the JAX CPU backend before anything imports jax."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import warnings

import pytest

from fixtures.synth import make_synth_catalog, make_synth_field
from spherex_photometry.io.cutouts import read_cutout

warnings.filterwarnings("ignore")

# Isolated, well-separated point sources -> tight, unambiguous flux recovery.
POINT_SOURCES = [
    {"x": 10.0, "y": 10.0, "flux_mjy": 5.0},
    {"x": 30.0, "y": 30.0, "flux_mjy": 2.0},
]


@pytest.fixture
def synth_field(tmp_path):
    """A 2-cutout synthetic field + matching catalog. Returns a dict of paths."""
    cut = tmp_path / "cut"
    make_synth_field(cut, n_cutouts=2, seed=3, sources=POINT_SOURCES)
    c0 = read_cutout(sorted(cut.glob("cutout_*.fits"))[0])
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, POINT_SOURCES, c0.wcs)
    return {"cutouts_dir": cut, "catalog": cat, "sources": POINT_SOURCES,
            "wcs": c0.wcs}


@pytest.fixture
def one_cutout(tmp_path):
    """A single cutout MEF path + its injected sources."""
    from fixtures.synth import make_synth_cutout
    p = tmp_path / "cutout_0000_SYNTH0000_D1.fits"
    srcs, wcs = make_synth_cutout(p, sources=POINT_SOURCES, seed=0)
    return {"path": p, "sources": srcs, "wcs": wcs}
