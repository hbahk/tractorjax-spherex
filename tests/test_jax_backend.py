import numpy as np
import pytest

pytest.importorskip("tractor_jax")

from tractorjax_spherex import PhotometryConfig, run_photometry


def _run(field, **overrides):
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           prefetch="sync", **overrides)
    return run_photometry(field["cutouts_dir"], field["catalog"], cfg,
                          progress=False)


def _flux_by_id(res, sid):
    return float(np.mean(res["flux"][res["id"] == sid]))


@pytest.mark.parametrize("solver", ["linear", "eigfloor"])
def test_flux_recovery(synth_field, solver):
    res = _run(synth_field, solver=solver, pad_bucket=0)
    for i, s in enumerate(synth_field["sources"], start=1):
        assert _flux_by_id(res, i) == pytest.approx(s["flux_mjy"], rel=0.05)


def test_tile_chunk_output_identical(synth_field):
    a = _run(synth_field, solver="linear", pad_bucket=0, tile_chunk=0)
    b = _run(synth_field, solver="linear", pad_bucket=0, tile_chunk=2)
    a.sort(["cutout_index", "id"])
    b.sort(["cutout_index", "id"])
    assert np.allclose(a["flux"], b["flux"], rtol=1e-9, atol=1e-9)


def test_pad_bucket_matches_natural(synth_field):
    a = _run(synth_field, solver="eigfloor", pad_bucket=0)
    b = _run(synth_field, solver="eigfloor", pad_bucket=32)
    a.sort(["cutout_index", "id"])
    b.sort(["cutout_index", "id"])
    assert np.allclose(a["flux"], b["flux"], rtol=1e-4, atol=1e-6)


def test_eigfloor_prior_smoke(synth_field):
    # eigfloor_prior needs SED band columns (present in the synth catalog)
    res = _run(synth_field, solver="eigfloor_prior", pad_bucket=0,
               protect_zmag_max=20.0)
    assert len(res) > 0 and np.all(np.isfinite(res["flux"]))
