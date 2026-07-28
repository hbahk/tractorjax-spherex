import numpy as np
import pytest

from spherex_photometry.background import build_background_mask, fit_background_plane
from spherex_photometry.models import ls_shapes_to_ab_phi
from spherex_photometry.priors import catalog_band_fluxes_ujy, predict_flux_ujy


def test_ls_shapes_round_source():
    ab, phi = ls_shapes_to_ab_phi(0.0, 0.0)
    assert ab == pytest.approx(1.0) and 0.0 <= phi < 180.0


def test_ls_shapes_ab_monotone():
    # larger ellipticity magnitude -> smaller axis ratio
    ab1, _ = ls_shapes_to_ab_phi(0.1, 0.0)
    ab2, _ = ls_shapes_to_ab_phi(0.5, 0.0)
    assert ab2 < ab1 < 1.0


def test_ls_shapes_pa_sign_convention():
    # phi = -0.5*atan2(e2, e1) folded to [0,180)
    _, phi = ls_shapes_to_ab_phi(1.0, 1.0)   # atan2=45deg -> -22.5 -> 157.5
    assert phi == pytest.approx(157.5, abs=1e-6)


def test_predict_flux_interpolation():
    from spherex_photometry.priors import BAND_LAM
    # flat SED in f_nu -> prediction equals the flat value anywhere in range
    bf = np.full((1, 6), 10.0)  # uJy already? no: catalog_band uses nmgy*3.631
    pred, nb = predict_flux_ujy(bf, np.array([BAND_LAM[2]]))
    assert nb[0] == 6 and pred[0] == pytest.approx(10.0, rel=1e-6)


def test_predict_flux_needs_two_bands():
    bf = np.full((1, 6), np.nan)
    bf[0, 0] = 5.0
    pred, nb = predict_flux_ujy(bf, np.array([1.0]))
    assert nb[0] == 1 and np.isnan(pred[0])


def test_predict_flux_extrapolation_clipped():
    bf = np.full((1, 6), np.nan)
    bf[0, 4], bf[0, 5] = 10.0, 1.0   # steep W1-W2 slope
    # far red of W2 -> slope clipped, stays finite and positive
    pred, _ = predict_flux_ujy(bf, np.array([10.0]))
    assert np.isfinite(pred[0]) and pred[0] > 0


def test_catalog_band_fluxes_prefers_dered():
    from astropy.table import Table
    t = Table({"dered_flux_z": [2.0], "flux_z": [1.0]})
    bf = catalog_band_fluxes_ujy(t)
    from spherex_photometry.priors import NMGY_TO_UJY
    assert bf[0, 3] == pytest.approx(2.0 * NMGY_TO_UJY)


def test_background_plane_recovers_tilt():
    ny, nx = 30, 30
    yy, xx = np.indices((ny, nx))
    truth = 0.5 + 0.03 * xx + 0.02 * yy
    img = truth.copy()
    flg = np.zeros((ny, nx), dtype=np.int32)
    var = np.ones((ny, nx))
    bkg = fit_background_plane(img, np.zeros_like(img), flg, var)
    assert np.allclose(bkg, truth, atol=1e-6)


def test_background_mask_flags_and_variance():
    flg = np.zeros((3, 3), dtype=np.int32)
    var = np.ones((3, 3))
    var[0, 0] = -1.0            # bad variance
    from spherex_photometry.constants import SOURCE_BIT
    flg[1, 1] = SOURCE_BIT     # source pixel
    mask = build_background_mask(flg, var)
    assert mask[0, 0] and mask[1, 1] and not mask[2, 2]
