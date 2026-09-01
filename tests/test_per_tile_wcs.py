"""Dropping the per-tile WCS must not move a single flux.

``build_cutout_tiles`` used to call ``WCS.slice`` once per tile — a deep copy —
although the only thing ever read back from it was the CD matrix, which
``shift_wcs`` leaves untouched.  The default now computes one cutout-level
``cd_inv`` instead.  These tests pin the three claims that make it safe:

1. a shifted WCS has the same CD matrix as the cutout's, for every shift;
2. the tile records carry that one shared matrix and no WCS; and
3. the photometry is identical with ``PER_TILE_WCS`` True and False.
"""

import numpy as np
import pytest

from spherex_photometry.backends import jax_backend as JB
from spherex_photometry.config import PhotometryConfig
from spherex_photometry.io.cutouts import read_cutout
from spherex_photometry.pipeline import run_photometry
from spherex_photometry.tiling import cd_inv_from_wcs, shift_wcs

SHIFTS = [(0, 0), (-3, -3), (12, 42), (87, 87), (900, 900), (-3, 60)]
COMPARED = ("flux", "flux_err", "central_wavelength", "bandwidth", "id")


def test_shift_wcs_preserves_cd_inv(one_cutout):
    """The claim the whole change rests on."""
    wcs = read_cutout(one_cutout["path"]).wcs
    ref = cd_inv_from_wcs(wcs)
    assert np.isfinite(ref).all()
    assert ref.dtype == np.float32
    for xs, ys in SHIFTS:
        assert np.array_equal(ref, cd_inv_from_wcs(shift_wcs(wcs, xs, ys))), \
            f"cd_inv changed under shift {(xs, ys)}"


def test_cd_inv_falls_back_instead_of_raising():
    """A degenerate WCS is a bad fit, not an exception in the batch builder."""
    class _Singular:
        class wcs:            # mimics astropy's attribute layout
            pass
        pixel_scale_matrix = np.zeros((2, 2))

    assert np.array_equal(cd_inv_from_wcs(_Singular()),
                          np.eye(2, dtype=np.float32))

    class _Unusable:
        wcs = None

    assert np.isfinite(cd_inv_from_wcs(_Unusable())).all()


def _run(cutouts_dir, catalog, per_tile_wcs, monkeypatch):
    """One photometry pass with PER_TILE_WCS forced, plus the tiles it built."""
    monkeypatch.setattr(JB, "PER_TILE_WCS", per_tile_wcs)
    seen = []
    original = JB.build_cutout_tiles

    def spy(*a, **kw):
        tiles = original(*a, **kw)
        seen.append(tiles)
        return tiles

    monkeypatch.setattr(JB, "build_cutout_tiles", spy)
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           solver="linear")
    table = run_photometry(cutouts_dir, catalog, cfg, progress=False)
    monkeypatch.undo()
    return table, seen


def test_tile_records_carry_one_shared_cd_inv(synth_field, monkeypatch):
    _, seen = _run(synth_field["cutouts_dir"], synth_field["catalog"],
                   False, monkeypatch)
    assert seen, "build_cutout_tiles was never called"
    for tiles in seen:
        assert all(t["wcs"] is None for t in tiles)
        cd = tiles[0]["cd_inv"]
        assert cd.dtype == np.float32 and np.isfinite(cd).all()
        # one object, not one per tile -- that is the allocation being saved
        assert all(t["cd_inv"] is cd for t in tiles)


def test_legacy_path_still_builds_a_wcs_with_the_same_cd(synth_field,
                                                         monkeypatch):
    _, seen = _run(synth_field["cutouts_dir"], synth_field["catalog"],
                   True, monkeypatch)
    for tiles in seen:
        assert all(t["wcs"] is not None for t in tiles)
        assert np.array_equal(tiles[0]["cd_inv"],
                              cd_inv_from_wcs(tiles[0]["wcs"]))


def test_batch_builder_accepts_records_without_cd_inv(synth_field, monkeypatch):
    """tile_records assembled by an outside caller (WCS, no cd_inv) still work."""
    _, seen = _run(synth_field["cutouts_dir"], synth_field["catalog"],
                   True, monkeypatch)
    tiles = seen[0]
    expected = tiles[0]["cd_inv"]
    for t in tiles:
        del t["cd_inv"]
    assert np.array_equal(cd_inv_from_wcs(tiles[0]["wcs"]), expected)
    assert tiles[0].get("cd_inv") is None      # the fallback branch is reached


def test_photometry_bit_identical_both_ways(synth_field, monkeypatch):
    fast, _ = _run(synth_field["cutouts_dir"], synth_field["catalog"],
                   False, monkeypatch)
    legacy, _ = _run(synth_field["cutouts_dir"], synth_field["catalog"],
                     True, monkeypatch)
    assert len(fast) == len(legacy) > 0
    for col in COMPARED:
        a = np.asarray(fast[col])
        b = np.asarray(legacy[col])
        if np.issubdtype(a.dtype, np.floating):
            assert np.array_equal(a, b, equal_nan=True), f"{col} differs"
        else:
            assert np.array_equal(a, b), f"{col} differs"


@pytest.mark.realdata
def test_photometry_bit_identical_on_real_cutouts(monkeypatch, request):
    """Same assertion on real SPHEREx cutouts, if a field is available.

    Point at one with ``--realdata-dir`` / the ``SPHEREX_TEST_CUTOUTS`` env var
    as the other realdata tests do; skipped otherwise.
    """
    import os
    from pathlib import Path

    d = os.environ.get("SPHEREX_TEST_CUTOUTS")
    cat = os.environ.get("SPHEREX_TEST_CATALOG")
    if not d or not cat:
        pytest.skip("set SPHEREX_TEST_CUTOUTS and SPHEREX_TEST_CATALOG")
    fast, _ = _run(Path(d), Path(cat), False, monkeypatch)
    legacy, _ = _run(Path(d), Path(cat), True, monkeypatch)
    for col in COMPARED:
        a, b = np.asarray(fast[col]), np.asarray(legacy[col])
        assert np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)), \
            f"{col} differs"
