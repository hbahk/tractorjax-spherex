"""The vectorised PSF-zone lookup must reproduce the scalar helpers exactly.

``zone_planes_and_weights`` replaces one ``select_zone_plane`` and one
``zone_bilinear_weights`` call per tile with a single call for all tiles.  It is
a pure re-expression — same comparisons, same tie-breaks — so these tests assert
element-by-element equality against the scalar helpers, including on the cases
where a tie-break decides: exact zone centres, lattice edges (clamped, not
extrapolated), points far outside the lattice, and a zone subset with a corner
missing.
"""

import numpy as np
import pytest
from astropy.table import Table

from spherex_photometry.backends import jax_backend as JB
from spherex_photometry.config import PhotometryConfig
from spherex_photometry.pipeline import run_photometry
from spherex_photometry.prepare import (
    select_zone_plane,
    zone_bilinear_weights,
    zone_planes_and_weights,
)

PITCH = 185.4545454545  # the SPHEREx zone pitch seen in the L2 PSF headers
FIRST = 93.22727273


def zone_table(nx, ny, *, drop=()):
    """A zone table on the real SPHEREx lattice; ``drop`` removes rows."""
    xs, ys, planes = [], [], []
    k = 0
    for iy in range(ny):
        for ix in range(nx):
            if k not in drop:
                xs.append(FIRST + ix * PITCH)
                ys.append(FIRST + iy * PITCH)
                planes.append(k)
            k += 1
    return Table({"zone_id": np.arange(len(xs)),
                  "x": np.array(xs), "y": np.array(ys),
                  "plane_idx": np.array(planes)})


def scalar_reference(tab, xs, ys):
    planes = np.array([select_zone_plane(tab, x, y) for x, y in zip(xs, ys)])
    weights = np.array([zone_bilinear_weights(tab, x, y)
                        for x, y in zip(xs, ys)])
    return planes, weights


def probe_points(tab, n=200, seed=0):
    """Random points plus every case where a tie-break decides."""
    rng = np.random.default_rng(seed)
    zx = np.asarray(tab["x"], float)
    zy = np.asarray(tab["y"], float)
    lo = min(zx.min(), zy.min()) - 300.0
    hi = max(zx.max(), zy.max()) + 300.0
    xs = [rng.uniform(lo, hi, n), zx, zx + PITCH / 2, zx - 1e-9,
          np.full(zx.size, lo), np.full(zx.size, hi)]
    ys = [rng.uniform(lo, hi, n), zy, zy + PITCH / 2, zy - 1e-9,
          np.full(zy.size, lo), np.full(zy.size, hi)]
    return np.concatenate(xs), np.concatenate(ys)


@pytest.mark.parametrize("nx,ny,drop", [
    (11, 11, ()),        # the full 121-zone lattice
    (3, 3, ()),          # a delivered subset
    (2, 2, ()),          # the smallest lattice with a real bilinear blend
    (1, 1, ()),          # single zone: weights degenerate to one-hot
    (1, 3, ()),          # degenerate in x only
    (3, 3, (4,)),        # a corner absent from the subset -> nearest-zone fallback
])
def test_matches_scalar_helpers(nx, ny, drop):
    tab = zone_table(nx, ny, drop=drop)
    xs, ys = probe_points(tab)
    planes, rows, weights = zone_planes_and_weights(tab, xs, ys)
    ref_planes, ref_weights = scalar_reference(tab, xs, ys)

    assert np.array_equal(planes, ref_planes)
    assert np.array_equal(weights, ref_weights)          # bit-for-bit
    # normalised the same way the scalar helper does (w / w.sum()), so equal to
    # float round-off, not exactly
    assert np.allclose(weights.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12)
    assert (weights >= 0).all()
    assert np.array_equal(np.asarray(tab["plane_idx"])[rows], planes)


def test_empty_input_is_handled():
    tab = zone_table(3, 3)
    planes, rows, weights = zone_planes_and_weights(tab, [], [])
    assert planes.shape == (0,) and rows.shape == (0,)
    assert weights.shape == (0, len(tab))


def test_scalar_and_vector_agree_through_the_pipeline(synth_field, monkeypatch):
    """End to end: VECTOR_ZONES on and off must give identical photometry."""
    def run(vector):
        monkeypatch.setattr(JB, "VECTOR_ZONES", vector)
        cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                               solver="linear")
        out = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                             cfg, progress=False)
        monkeypatch.undo()
        return out

    fast, scalar = run(True), run(False)
    assert len(fast) == len(scalar) > 0
    for col in ("flux", "flux_err", "central_wavelength", "id"):
        a, b = np.asarray(fast[col]), np.asarray(scalar[col])
        if np.issubdtype(a.dtype, np.floating):
            assert np.array_equal(a, b, equal_nan=True), f"{col} differs"
        else:
            assert np.array_equal(a, b), f"{col} differs"


def test_stamp_cache_returns_one_object_per_plane(synth_field, monkeypatch):
    """Tiles in the same zone must share the downsampled kernel object.

    The engine keys its Fourier-transform cache on kernel identity, so handing
    out a fresh array per tile would silently defeat it.
    """
    monkeypatch.setattr(JB, "VECTOR_ZONES", True)
    seen = []
    original = JB.build_cutout_tiles
    monkeypatch.setattr(JB, "build_cutout_tiles",
                        lambda *a, **k: seen.append(original(*a, **k)) or seen[-1])
    cfg = PhotometryConfig(backend="jax", device="cpu", solver="linear")
    run_photometry(synth_field["cutouts_dir"], synth_field["catalog"], cfg,
                   progress=False)
    for tiles in seen:
        ids = {id(t["psf"]) for t in tiles}
        planes = {t["psf"].tobytes() for t in tiles}
        assert len(ids) == len(planes), \
            "a distinct array object per tile — the engine's FFT cache will miss"
