"""R7 effective-PSF bundles (``PSFKIND = 'EPSF'``) through the SPHEREx layer.

A QR3/DR1 bundle carries the ePSF (5x, pixel response included) where a QR2
bundle carries the 10x optical cube. The layer must read the kind, hand the
ePSF planes to the engine untouched (no 2x downsample, no core shift) and ask
it to point-sample them (``pixel_integration="point"``); with that, the
injected fluxes of a synthetic field are recovered as well through the ePSF as
through the optical cube of the same Gaussian, on both backends, and the wrong
pairing (window on the ePSF) is visibly biased.
"""
import numpy as np
import pytest

from fixtures.synth import make_synth_catalog, make_synth_cutout, make_synth_field
from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.config import ConfigError
from tractorjax_spherex.io.cutouts import psf_kind_from_header, read_cutout
from tractorjax_spherex.prepare import (
    core_shift_applies,
    pixel_integration_for,
    psf_stamp_5x,
)
from tractorjax_spherex.psf_cache import cube_signature

SOURCES = [
    {"x": 10.0, "y": 10.0, "flux_mjy": 5.0},
    {"x": 30.0, "y": 30.0, "flux_mjy": 2.0},
]


@pytest.fixture(params=["optical", "effective"])
def field(tmp_path, request):
    cut = tmp_path / "cut"
    make_synth_field(cut, n_cutouts=2, seed=3, sources=SOURCES, psf_kind=request.param)
    c0 = read_cutout(min(cut.glob("cutout_*.fits")))
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, SOURCES, c0.wcs)
    return {"cutouts_dir": cut, "catalog": cat, "kind": request.param}


def _flux_by_id(res, sid):
    return float(np.mean(res["flux"][res["id"] == sid]))


# The optical synthetic path carries a ~2 % bias of its own (the toy image is
# integrated on a 10x sub-grid, the QR2-style kernel goes through the
# centre-preserving 10x->5x downsample; the existing pipeline tests allow 5 %).
# The ePSF path is exact for the same Gaussian, so it is held to 1 %.
TOL = {"optical": 0.03, "effective": 0.01}


def _jax_cfg(**kw):
    # psf_core_shift=False: the synthetic Gaussian has no QR2 core offset, and
    # the measured table would shift the optical kernel by ~0.05 px (a ~1.7 %
    # flux error on these point sources), which is not what is under test here
    kw.setdefault("psf_core_shift", False)
    return PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                            prefetch="sync", solver="linear", pad_bucket=0, **kw)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fast", [False, "auto"])
def test_readers_report_the_kind(tmp_path, fast):
    p = tmp_path / "e.fits"
    make_synth_cutout(p, sources=SOURCES, psf_kind="effective")
    c = read_cutout(p, fast=fast)
    assert c.psf_kind == "effective" and c.psf_oversamp == 5
    assert c.psf_cube.shape == (1, 51, 51) and abs(c.psf_cube[0].sum() - 1.0) < 1e-6
    assert c.primary_header.get("EPSFCAL") and str(c.primary_header["PSFKIND"]) == "EPSF"
    assert "neff" in c.psf_zones.colnames
    q = tmp_path / "o.fits"
    make_synth_cutout(q, sources=SOURCES, psf_kind="optical")
    o = read_cutout(q, fast=fast)
    assert o.psf_kind == "optical" and o.psf_oversamp == 10


def test_kind_keyword_parsing():
    assert psf_kind_from_header({}) == "optical"
    assert psf_kind_from_header({"PSFKIND": "EPSF"}) == "effective"
    assert psf_kind_from_header({"PSFKIND": "OPTICAL"}) == "optical"
    with pytest.raises(ValueError):
        psf_kind_from_header({"PSFKIND": "something"})


def test_effective_stamp_is_the_plane_itself(tmp_path):
    p = tmp_path / "e.fits"
    make_synth_cutout(p, sources=SOURCES, psf_kind="effective")
    c = read_cutout(p)
    s = psf_stamp_5x(c, 0)
    assert s.shape == (51, 51) and np.array_equal(s, c.psf_cube[0])
    assert pixel_integration_for(c) == "point"
    q = tmp_path / "o.fits"
    make_synth_cutout(q, sources=SOURCES, psf_kind="optical")
    o = read_cutout(q)
    assert psf_stamp_5x(o, 0).shape == (51, 51) and pixel_integration_for(o) == "window"


def test_core_shift_resolution(tmp_path):
    e = read_cutout(make_synth_cutout(tmp_path / "e.fits", sources=SOURCES, psf_kind="effective") and tmp_path / "e.fits")
    o = read_cutout(make_synth_cutout(tmp_path / "o.fits", sources=SOURCES, psf_kind="optical") and tmp_path / "o.fits")
    auto = PhotometryConfig()
    assert core_shift_applies(auto, o) is True and core_shift_applies(auto, e) is False
    off = PhotometryConfig(psf_core_shift=False)
    assert core_shift_applies(off, o) is False and core_shift_applies(off, e) is False
    on = PhotometryConfig(psf_core_shift=True)
    assert core_shift_applies(on, o) is True
    with pytest.raises(ValueError, match="effective"):
        core_shift_applies(on, e)
    with pytest.raises(ConfigError):
        PhotometryConfig(psf_core_shift="never")


def test_cache_signature_separates_epsf_libraries(tmp_path):
    a = read_cutout(make_synth_cutout(tmp_path / "a.fits", sources=SOURCES, psf_kind="effective",
                                      fwhm_native=2.5) and tmp_path / "a.fits")
    b = read_cutout(make_synth_cutout(tmp_path / "b.fits", sources=SOURCES, psf_kind="effective",
                                      fwhm_native=2.9) and tmp_path / "b.fits")
    # both planes sum to 1.0 and share detector, shape and zone table
    assert abs(a.psf_cube[0].sum() - b.psf_cube[0].sum()) < 1e-9
    assert cube_signature(a) != cube_signature(b)
    assert cube_signature(a) == cube_signature(read_cutout(tmp_path / "a.fits"))


# --------------------------------------------------------------------------- #
# photometry
# --------------------------------------------------------------------------- #
def test_jax_backend_recovers_fluxes_for_both_kinds(field):
    res = run_photometry(field["cutouts_dir"], field["catalog"], _jax_cfg(), progress=False)
    for i, s in enumerate(SOURCES, start=1):
        assert _flux_by_id(res, i) == pytest.approx(s["flux_mjy"], rel=TOL[field["kind"]]), field["kind"]


def test_cpu_backend_recovers_fluxes_for_both_kinds(field):
    pytest.importorskip("tractor")
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear", psf_core_shift=False)
    res = run_photometry(field["cutouts_dir"], field["catalog"], cfg, progress=False)
    for i, s in enumerate(SOURCES, start=1):
        assert _flux_by_id(res, i) == pytest.approx(s["flux_mjy"], rel=TOL[field["kind"]]), field["kind"]


def test_window_on_the_epsf_is_biased(tmp_path, monkeypatch):
    """Forcing the QR2 rendering on an ePSF bundle applies the pixel window
    twice and biases the fluxes; the automatic dispatch must not do that."""
    from tractorjax_spherex.backends import jax_backend as JB
    cut = tmp_path / "cut"
    make_synth_field(cut, n_cutouts=1, seed=3, sources=SOURCES, psf_kind="effective")
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, SOURCES, read_cutout(min(cut.glob("cutout_*.fits"))).wcs)
    good = run_photometry(cut, cat, _jax_cfg(), progress=False)
    monkeypatch.setattr(JB, "pixel_integration_for", lambda cutout: "window")
    bad = run_photometry(cut, cat, _jax_cfg(), progress=False)
    for i, s in enumerate(SOURCES, start=1):
        assert _flux_by_id(good, i) == pytest.approx(s["flux_mjy"], rel=0.01)
        assert abs(_flux_by_id(bad, i) / s["flux_mjy"] - 1.0) > 0.01


def test_eigfloor_default_config_runs_on_epsf(tmp_path):
    """The configuration of record (eigfloor, zone interp, core shift 'auto')
    accepts an R7 bundle without any flag."""
    cut = tmp_path / "cut"
    make_synth_field(cut, n_cutouts=2, seed=3, sources=SOURCES, psf_kind="effective")
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, SOURCES, read_cutout(min(cut.glob("cutout_*.fits"))).wcs)
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64", prefetch="sync",
                           pad_bucket=0)
    assert cfg.solver == "eigfloor" and cfg.psf_core_shift == "auto"
    res = run_photometry(cut, cat, cfg, progress=False)
    for i, s in enumerate(SOURCES, start=1):
        assert _flux_by_id(res, i) == pytest.approx(s["flux_mjy"], rel=0.02)
