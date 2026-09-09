"""The fitsio fast reader must be indistinguishable from the astropy one.

Nothing in :mod:`tractorjax_spherex.io.fast` is allowed to change a number, so
these tests compare the two readers field by field on real cutout MEFs: arrays
with dtype, the PSF-zone table, every scalar, every header keyword value, and
the WCS through both projections (which is what the fast path rebuilds by hand).
"""

import numpy as np
import pytest
from astropy.io import fits

from tractorjax_spherex.io import cutouts as C
from tractorjax_spherex.io import fast as F

pytestmark = pytest.mark.skipif(not F.have_fitsio(),
                                reason="fitsio not installed")

ARRAY_FIELDS = ("image", "flags", "variance", "zodi", "psf_cube",
                "cwave_map", "cband_map", "sapm")
SCALAR_FIELDS = ("crpix1a", "crpix2a", "psf_oversamp", "detector",
                 "cwave_center")


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.floating):
        return np.array_equal(a, b, equal_nan=True)   # masked pixels are NaN
    return np.array_equal(a, b)


def assert_readers_agree(path):
    ref = C.read_cutout(path, fast=False)
    new = C.read_cutout(path, fast=True)

    for key in ARRAY_FIELDS:
        assert _same(ref[key], new[key]), f"{key} differs"
    for key in SCALAR_FIELDS:
        assert ref[key] == new[key], f"{key}: {ref[key]!r} != {new[key]!r}"

    zr, zn = ref.psf_zones, new.psf_zones
    assert zr.colnames == zn.colnames
    for c in zr.colnames:
        assert np.array_equal(np.asarray(zr[c]), np.asarray(zn[c])), f"zones {c}"

    # The rebuilt WCS carries only the projection keywords (no DATE/MJD cards),
    # so compare what callers use: both projections on a grid spanning the
    # cutout with margin, and the pixel area.
    ny, nx = ref.image.shape
    xx, yy = np.meshgrid(np.linspace(-5.0, nx + 5.0, 12),
                         np.linspace(-5.0, ny + 5.0, 12))
    w0 = np.asarray(ref.wcs.pixel_to_world_values(xx, yy))
    w1 = np.asarray(new.wcs.pixel_to_world_values(xx, yy))
    assert np.array_equal(w0, w1), "wcs pixel->world"
    assert np.array_equal(
        np.asarray(ref.wcs.world_to_pixel_values(*w0)),
        np.asarray(new.wcs.world_to_pixel_values(*w0))), "wcs world->pixel"
    assert ref.wcs.proj_plane_pixel_area() == new.wcs.proj_plane_pixel_area()

    # Headers: the fast path hands back keyword -> value mappings. Every
    # keyword the astropy Header holds must be present with an equal value.
    for key in ("image_header", "primary_header"):
        rh, nh = ref[key], new[key]
        for k in rh:
            if not k or k in ("COMMENT", "HISTORY", "CONTINUE"):
                continue
            assert rh[k] == nh.get(k), f"{key}[{k}]: {rh[k]!r} != {nh.get(k)!r}"


def test_readers_agree_on_one_cutout(one_cutout):
    assert_readers_agree(one_cutout["path"])


def test_readers_agree_across_a_field(synth_field):
    for p in sorted(synth_field["cutouts_dir"].glob("cutout_*.fits")):
        assert_readers_agree(p)


def test_header_dict_converts_back_to_astropy(one_cutout):
    c = C.read_cutout(one_cutout["path"], fast=True)
    assert isinstance(c.primary_header, F.HeaderDict)
    hdr = c.primary_header.to_astropy()
    assert isinstance(hdr, fits.Header)
    for k, v in c.primary_header.items():
        assert hdr[k] == v


def test_psf_cube_cache_returns_equal_cubes(synth_field):
    """A cache hit must hand back the same values the file holds."""
    F.clear_caches()
    paths = sorted(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    first = C.read_cutout(paths[0], fast=True).psf_cube
    for p in paths:                       # later reads may hit the cache
        cached = C.read_cutout(p, fast=True).psf_cube
        direct = C.read_cutout(p, fast=False).psf_cube
        assert _same(cached, direct)
    F.clear_caches()
    assert _same(first, C.read_cutout(paths[0], fast=True).psf_cube)


def test_fast_psf_cube_disabled_still_agrees(one_cutout, monkeypatch):
    monkeypatch.setattr(F, "FAST_PSF_CUBE", False)
    F.clear_caches()
    assert_readers_agree(one_cutout["path"])


def test_fast_wcs_disabled_still_agrees(one_cutout, monkeypatch):
    monkeypatch.setattr(F, "FAST_WCS", False)
    monkeypatch.setattr(F, "FAST_HEADERS", True)
    assert_readers_agree(one_cutout["path"])


def test_fast_headers_disabled_gives_astropy_headers(one_cutout, monkeypatch):
    monkeypatch.setattr(F, "FAST_HEADERS", False)
    c = C.read_cutout(one_cutout["path"], fast=True)
    assert isinstance(c.primary_header, fits.Header)
    assert_readers_agree(one_cutout["path"])


def test_occupancy_geometry_matches_astropy(one_cutout):
    from astropy.wcs import WCS

    H, W, wcs = F.read_image_geometry(one_cutout["path"])
    with fits.open(one_cutout["path"], memmap=False) as hdul:
        hdr = hdul["IMAGE"].header
        assert (H, W) == (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]))
        ref = WCS(hdr).celestial
    xx, yy = np.meshgrid(np.linspace(0, W, 8), np.linspace(0, H, 8))
    assert np.array_equal(np.asarray(ref.pixel_to_world_values(xx, yy)),
                          np.asarray(wcs.pixel_to_world_values(xx, yy)))


def test_fast_true_without_fitsio_raises(one_cutout, monkeypatch):
    monkeypatch.setattr(F, "have_fitsio", lambda: False)
    with pytest.raises(ImportError, match="fitsio"):
        C.read_cutout(one_cutout["path"], fast=True)
