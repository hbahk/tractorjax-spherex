import numpy as np
import pytest

pytest.importorskip("tractor")
pytest.importorskip("tractor_jax")   # cross-backend comparison needs both

from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.config import ConfigError


def _flux_by_id(res, sid):
    return float(np.mean(res["flux"][res["id"] == sid]))


def test_cpu_flux_recovery(synth_field):
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear")
    res = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         cfg, progress=False)
    for i, s in enumerate(synth_field["sources"], start=1):
        assert _flux_by_id(res, i) == pytest.approx(s["flux_mjy"], rel=0.05)


def test_cross_backend_agreement(synth_field):
    jax = run_photometry(
        synth_field["cutouts_dir"], synth_field["catalog"],
        PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                         prefetch="sync", solver="linear", pad_bucket=0),
        progress=False)
    cpu = run_photometry(
        synth_field["cutouts_dir"], synth_field["catalog"],
        PhotometryConfig(backend="cpu-tractor", solver="linear"),
        progress=False)
    for sid in (1, 2):
        fj = _flux_by_id(jax, sid)
        fc = _flux_by_id(cpu, sid)
        # independent rendering paths agree to well under 1% for point sources
        assert fj == pytest.approx(fc, rel=0.01)


def test_cpu_rejects_eigfloor():
    with pytest.raises(ConfigError):
        PhotometryConfig(backend="cpu-tractor", solver="eigfloor")


def test_cpu_handles_galaxy_and_nan_shape(tmp_path):
    # a galaxy (non-round ellipticity) renders, and a NaN shape_r is treated as a
    # point source instead of raising KeyError and dropping the whole cutout.
    from fixtures.synth import make_synth_catalog, make_synth_field
    from tractorjax_spherex.io.cutouts import read_cutout

    srcs = [{"x": 10.0, "y": 10.0, "flux_mjy": 5.0},
            {"x": 28.0, "y": 22.0, "flux_mjy": 2.0, "shape_r": 1.5, "sersic": 1.0},
            {"x": 20.0, "y": 31.0, "flux_mjy": 1.5}]
    d = tmp_path / "cut"
    make_synth_field(d, n_cutouts=1, sources=srcs)
    c0 = read_cutout(min(d.glob("cutout_*.fits")))
    cat = make_synth_catalog(None, srcs, c0.wcs)
    cat["shape_e1"][1] = 0.3      # give the galaxy a real orientation
    cat["shape_e2"][1] = 0.15
    cat["shape_r"][2] = np.nan    # NaN shape must not crash the build
    p = tmp_path / "cat.parquet"
    cat.write(str(p), overwrite=True)

    res = run_photometry(d, p, PhotometryConfig(backend="cpu-tractor",
                                               solver="linear"), progress=False)
    assert set(res["id"]) == {1, 2, 3}          # cutout not dropped
    assert np.all(np.isfinite(res["flux"]))


def test_cross_backend_agreement_for_galaxies(tmp_path):
    """The CPU Fourier (galaxy) path block-integrates the optical PSF like the
    point-source path does, so galaxy fluxes agree with the JAX backend. Until
    0.3.1 that path point-sampled the stamp and every galaxy came out 2.5-2.8 %
    low relative to JAX on this field while point sources matched exactly."""
    from fixtures.synth import make_synth_catalog, make_synth_field
    from tractorjax_spherex.io.cutouts import read_cutout

    srcs = [{"x": 10.0, "y": 10.0, "flux_mjy": 5.0},
            {"x": 30.0, "y": 12.0, "flux_mjy": 3.0, "shape_r": 0.8, "sersic": 1.0},
            {"x": 12.0, "y": 30.0, "flux_mjy": 3.0, "shape_r": 1.5, "sersic": 4.0},
            {"x": 30.0, "y": 30.0, "flux_mjy": 3.0, "shape_r": 3.0, "sersic": 1.0}]
    d = tmp_path / "cut"
    make_synth_field(d, n_cutouts=1, seed=3, sources=srcs)
    c0 = read_cutout(min(d.glob("cutout_*.fits")))
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, srcs, c0.wcs)
    jax = run_photometry(d, cat, PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                                                  prefetch="sync", solver="linear", pad_bucket=0,
                                                  psf_core_shift=False), progress=False)
    cpu = run_photometry(d, cat, PhotometryConfig(backend="cpu-tractor", solver="linear",
                                                  psf_core_shift=False), progress=False)
    for sid in range(1, 5):
        assert _flux_by_id(cpu, sid) == pytest.approx(_flux_by_id(jax, sid), rel=1e-3)
