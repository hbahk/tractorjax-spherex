"""Per-visit fit diagnostics (``visit_diagnostics=True``): fit_chi2 and mask_frac.

Needs an engine whose solvers return diagnostics; skipped otherwise.
"""
import numpy as np
import pytest
from astropy.io import fits

pytest.importorskip("tractor_jax")

from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.backends.jax_backend import engine_has_diagnostics
from tractorjax_spherex.config import ConfigError

pytestmark = pytest.mark.skipif(not engine_has_diagnostics(),
                                reason="engine without return_diagnostics")


def _run(field, **overrides):
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           prefetch="sync", pad_bucket=0, **overrides)
    res = run_photometry(field["cutouts_dir"], field["catalog"], cfg, progress=False)
    res.sort(["cutout_index", "id"])
    return res


def test_columns_and_unchanged_fluxes(synth_field):
    off = _run(synth_field)
    on = _run(synth_field, visit_diagnostics=True)
    assert "fit_chi2" not in off.colnames
    assert on["fit_chi2"].dtype == np.float32 and on["mask_frac"].dtype == np.float32
    assert np.array_equal(np.asarray(off["id"]), np.asarray(on["id"]))
    assert np.allclose(off["flux"], on["flux"], rtol=1e-9, atol=1e-12)
    assert np.allclose(off["flux_err"], on["flux_err"], rtol=1e-9, atol=1e-12)
    assert np.all(np.isfinite(on["fit_chi2"])) and np.all(on["fit_chi2"] > 0)
    assert np.all(np.asarray(on["mask_frac"]) < 0.05)


def test_unflagged_bad_pixel_raises_its_visit(synth_field):
    base = _run(synth_field, visit_diagnostics=True)
    # a pixel next to source 1 (x=10, y=10) that lost the source light and has a
    # small variance, not flagged: the SPHEREx failure the diagnostic is for
    path = min(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    with fits.open(path, mode="update") as h:
        h["IMAGE"].data[10, 11] = np.median(h["IMAGE"].data)
        h["VARIANCE"].data[10, 11] /= 8.0
    bad = _run(synth_field, visit_diagnostics=True)
    first = int(np.min(bad["cutout_index"]))

    def chi2(tab, ci, sid):
        m = (np.asarray(tab["cutout_index"]) == ci) & (np.asarray(tab["id"]) == sid)
        return float(tab["fit_chi2"][m][0])

    other = int(np.max(bad["cutout_index"]))
    assert chi2(bad, first, 1) > 10 * chi2(base, first, 1)
    assert chi2(bad, first, 1) > 10 * chi2(bad, other, 1)
    assert chi2(bad, first, 2) == pytest.approx(chi2(base, first, 2), rel=0.2)


def test_masked_pixels_raise_mask_frac(synth_field):
    path = min(synth_field["cutouts_dir"].glob("cutout_*.fits"))
    with fits.open(path, mode="update") as h:
        h["VARIANCE"].data[9:12, 9:12] = np.nan      # invalid variance -> masked
    res = _run(synth_field, visit_diagnostics=True)
    first = int(np.min(res["cutout_index"]))
    m = (np.asarray(res["cutout_index"]) == first) & (np.asarray(res["id"]) == 1)
    assert float(res["mask_frac"][m][0]) > 0.3


@pytest.mark.parametrize("kw", [dict(solver="lasso"), dict(backend="cpu-tractor")])
def test_config_rejects_unsupported(kw):
    with pytest.raises(ConfigError):
        PhotometryConfig(visit_diagnostics=True, **kw)
