"""Per-visit fit diagnostics and quality flags (on by default on the jax backend).

Needs an engine whose solvers return diagnostics; skipped otherwise.
"""
import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("tractor_jax")

from astropy.table import Table, vstack

from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.backends.jax_backend import engine_has_diagnostics
from tractorjax_spherex.config import ConfigError
from tractorjax_spherex.constants import QUALITY_BAD_FIT, QUALITY_NO_DATA
from tractorjax_spherex.quality import quality_flags
from tractorjax_spherex.spectra import build_spectra

pytestmark = pytest.mark.skipif(not engine_has_diagnostics(),
                                reason="engine without return_diagnostics")


def _run(field, **overrides):
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           prefetch="sync", pad_bucket=0, **overrides)
    res = run_photometry(field["cutouts_dir"], field["catalog"], cfg, progress=False)
    res.sort(["cutout_index", "id"])
    return res


def test_on_by_default_where_available():
    assert PhotometryConfig().diagnostics_on()
    assert not PhotometryConfig(solver="lasso").diagnostics_on()
    assert not PhotometryConfig(backend="cpu-tractor", solver="linear").diagnostics_on()
    assert not PhotometryConfig(visit_diagnostics=False).diagnostics_on()


def test_columns_and_unchanged_fluxes(synth_field):
    off = _run(synth_field, visit_diagnostics=False)
    on = _run(synth_field)
    assert "fit_chi2" not in off.colnames and "quality_flag" not in off.colnames
    assert on["fit_chi2"].dtype == np.float32 and on["mask_frac"].dtype == np.float32
    assert on["quality_flag"].dtype == np.int16 and np.all(on["quality_flag"] == 0)
    assert np.array_equal(np.asarray(off["id"]), np.asarray(on["id"]))
    assert np.allclose(off["flux"], on["flux"], rtol=1e-9, atol=1e-12)
    assert np.allclose(off["flux_err"], on["flux_err"], rtol=1e-9, atol=1e-12)
    assert np.all(np.isfinite(on["fit_chi2"])) and np.all(on["fit_chi2"] > 0)
    assert np.all(np.asarray(on["mask_frac"]) < 0.05)


def test_unflagged_bad_pixel_raises_its_visit(synth_field):
    base = _run(synth_field)
    # a pixel next to source 1 (x=10, y=10) that lost the source light and has a
    # small variance, not flagged: the SPHEREx failure the diagnostic is for
    path = min(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    with fits.open(path, mode="update") as h:
        h["IMAGE"].data[10, 11] = np.median(h["IMAGE"].data)
        h["VARIANCE"].data[10, 11] /= 8.0
    bad = _run(synth_field)
    first = int(np.min(bad["cutout_index"]))

    def chi2(tab, ci, sid):
        m = (np.asarray(tab["cutout_index"]) == ci) & (np.asarray(tab["id"]) == sid)
        return float(tab["fit_chi2"][m][0])

    other = int(np.max(bad["cutout_index"]))
    assert chi2(bad, first, 1) > 10 * chi2(base, first, 1)
    assert chi2(bad, first, 1) > 10 * chi2(bad, other, 1)
    assert chi2(bad, first, 2) == pytest.approx(chi2(base, first, 2), rel=0.2)


def test_quality_flags_bits():
    # source 1: four ordinary visits and one 30x its median; source 2: one fully masked visit
    t = Table({"id": [1, 1, 1, 1, 1, 2, 2],
               "fit_chi2": np.array([0.5, 0.6, 0.4, 0.5, 15.0, 0.7, np.nan], dtype=np.float32)})
    f = quality_flags(t, rel_max=10.0)
    assert list(f) == [0, 0, 0, 0, QUALITY_BAD_FIT, 0, QUALITY_NO_DATA]
    assert list(quality_flags(t, rel_max=None)) == [0, 0, 0, 0, 0, 0, QUALITY_NO_DATA]
    assert list(quality_flags(Table({"id": [1]}))) == [0]


def test_flags_default_and_spectra_drop_them(synth_field):
    path = min(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    with fits.open(path, mode="update") as h:
        h["IMAGE"].data[10, 11] = np.median(h["IMAGE"].data)
        h["VARIANCE"].data[10, 11] /= 8.0
    # with two visits a median cannot single out one of them: give source 1 a
    # third, ordinary visit by duplicating the other cutout's rows
    res = _run(synth_field)
    first = int(np.min(res["cutout_index"]))
    extra = res[np.asarray(res["cutout_index"]) != first].copy()
    extra["cutout_index"] = extra["cutout_index"] + 100
    res = vstack([res, extra])
    res["quality_flag"] = quality_flags(res, rel_max=10.0)
    m = (np.asarray(res["cutout_index"]) == first) & (np.asarray(res["id"]) == 1)
    assert int(res["quality_flag"][m][0]) == QUALITY_BAD_FIT
    kept = build_spectra(res, ids=[1])[1]
    assert first not in set(np.asarray(kept["cutout_index"]).tolist())
    everything = build_spectra(res, ids=[1], drop_flagged=False)[1]
    assert len(everything) == len(kept) + 1


def test_masked_pixels_raise_mask_frac(synth_field):
    path = min(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    with fits.open(path, mode="update") as h:
        h["VARIANCE"].data[9:12, 9:12] = np.nan      # invalid variance -> masked
    res = _run(synth_field)
    first = int(np.min(res["cutout_index"]))
    m = (np.asarray(res["cutout_index"]) == first) & (np.asarray(res["id"]) == 1)
    assert float(res["mask_frac"][m][0]) > 0.3


@pytest.mark.parametrize("kw", [dict(visit_diagnostics=True, solver="lasso"),
                                dict(visit_diagnostics=True, backend="cpu-tractor", solver="linear"),
                                dict(visit_diagnostics="yes"), dict(visit_chi2_rel_max=0.5)])
def test_config_rejects_unsupported(kw):
    with pytest.raises(ConfigError):
        PhotometryConfig(**kw)
