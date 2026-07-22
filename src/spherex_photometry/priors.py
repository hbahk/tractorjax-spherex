"""SED flux predictor for the ``eigfloor_prior`` solver.

Lifted from ``analysis/deconfusion.py`` (only the SED-prediction half; the
faint-source image-model rendering is deferred to a later release). Predicts a
per-source flux at an arbitrary wavelength by log-log interpolation over the
Legacy Survey bands g/r/i/z/W1/W2, which becomes the Gaussian prior mean for the
penalized (faint) sources in the ``eigfloor_prior`` estimator.
"""

from __future__ import annotations

import numpy as np

NMGY_TO_UJY = 3.631
# LS band effective wavelengths [um] (g r i z from DECam, W1/W2 from WISE).
LS_BANDS = (("g", 0.481), ("r", 0.641), ("i", 0.783),
            ("z", 0.917), ("w1", 3.368), ("w2", 4.618))
BAND_LAM = np.array([lam for _, lam in LS_BANDS])
SLOPE_MAX = 2.0  # |dlnF/dlnlam| clip for end-slope extrapolation


def catalog_band_fluxes_ujy(df):
    """``(N, 6)`` band fluxes in uJy; ``dered_flux_*`` preferred, ``flux_*``
    fallback; non-positive/missing -> NaN."""
    n = len(df)
    cols = []
    names = set(getattr(df, "columns", getattr(df, "colnames", [])))
    for band, _lam in LS_BANDS:
        v = np.full(n, np.nan)
        for col in (f"dered_flux_{band}", f"flux_{band}"):
            if col in names:
                cand = np.asarray(df[col], dtype=np.float64)
                fill = ~(np.isfinite(v) & (v > 0)) & np.isfinite(cand) & (cand > 0)
                v[fill] = cand[fill]
        v[~(np.isfinite(v) & (v > 0))] = np.nan
        cols.append(v * NMGY_TO_UJY)
    return np.stack(cols, axis=1)


def predict_flux_ujy(band_flux_ujy, lam_um, slope_max=SLOPE_MAX):
    """Vectorized per-source log-log SED evaluation at ``lam_um``.

    Parameters
    ----------
    band_flux_ujy : (N, 6) uJy, NaN for missing (see :func:`catalog_band_fluxes_ujy`)
    lam_um : (N,) evaluation wavelength per source (the source's own CWAVE)

    Returns
    -------
    pred_ujy : (N,) predicted f_nu [uJy]; NaN where <2 positive bands or
        ``lam_um`` is not finite.
    n_bands : (N,) number of usable bands per source.

    Within coverage: linear interpolation of ln F vs ln lambda between the
    bracketing usable bands. Outside coverage: the end-segment slope, clipped to
    ``|slope| <= slope_max`` so a noisy W1-W2 colour cannot explode the
    extrapolation.
    """
    F = np.asarray(band_flux_ujy, dtype=np.float64)
    N, B = F.shape
    good = np.isfinite(F) & (F > 0)
    n_bands = good.sum(axis=1)
    lnF = np.where(good, np.log(np.where(good, F, 1.0)), np.nan)
    lnlam_b = np.log(BAND_LAM)
    lnlam = np.log(np.asarray(lam_um, dtype=np.float64))

    pred = np.full(N, np.nan)
    usable = (n_bands >= 2) & np.isfinite(lnlam)
    if not usable.any():
        return pred, n_bands

    lam_row = np.broadcast_to(lnlam_b, (N, B))
    le = good & (lam_row <= lnlam[:, None] + 1e-12)   # usable bands blueward
    ge = good & (lam_row >= lnlam[:, None] - 1e-12)   # usable bands redward
    has_lo, has_hi = le.any(axis=1), ge.any(axis=1)
    ilo = B - 1 - np.argmax(le[:, ::-1], axis=1)      # reddest blueward band
    ihi = np.argmax(ge, axis=1)                       # bluest redward band
    rows = np.arange(N)

    def second_extreme(mask, extreme_idx, red):
        m = mask.copy()
        m[rows, extreme_idx] = False
        if red:
            return B - 1 - np.argmax(m[:, ::-1], axis=1), m.any(axis=1)
        return np.argmax(m, axis=1), m.any(axis=1)

    # 1) interpolation: bracketing bands on both sides
    both = usable & has_lo & has_hi
    same = both & (ilo == ihi)
    pred[same] = np.exp(lnF[rows[same], ilo[same]])
    interp = both & (ilo != ihi)
    if interp.any():
        r = rows[interp]
        i0, i1 = ilo[interp], ihi[interp]
        s = (lnF[r, i1] - lnF[r, i0]) / (lnlam_b[i1] - lnlam_b[i0])
        pred[interp] = np.exp(lnF[r, i0] + s * (lnlam[interp] - lnlam_b[i0]))

    # 2) red extrapolation: lam beyond the reddest usable band
    red = usable & has_lo & ~has_hi
    if red.any():
        i2 = ilo
        i1, _ = second_extreme(good, i2, red=True)
        r = rows[red]
        a, b = i1[red], i2[red]
        s = (lnF[r, b] - lnF[r, a]) / (lnlam_b[b] - lnlam_b[a])
        s = np.clip(s, -slope_max, slope_max)
        pred[red] = np.exp(lnF[r, b] + s * (lnlam[red] - lnlam_b[b]))

    # 3) blue extrapolation: lam below the bluest usable band
    blue = usable & ~has_lo & has_hi
    if blue.any():
        i2 = ihi
        i1, _ = second_extreme(good, i2, red=False)
        r = rows[blue]
        a, b = i1[blue], i2[blue]
        s = (lnF[r, b] - lnF[r, a]) / (lnlam_b[b] - lnlam_b[a])
        s = np.clip(s, -slope_max, slope_max)
        pred[blue] = np.exp(lnF[r, b] + s * (lnlam[blue] - lnlam_b[b]))

    return pred, n_bands


def make_prior_context(catalog, config):
    """Build the per-run SED-prior context for the ``eigfloor_prior`` solver.

    Returns ``None`` unless ``config.solver == "eigfloor_prior"``. The context
    carries the ``(N, 6)`` band fluxes, the predictor, and the sigma policy
    (``sigma_prior = sigma_frac * f_pred``, floored at ``sigma_min``), all in the
    fit's flux units (mJy). Evaluated per cutout at each source's own CWAVE.
    """
    if config.solver != "eigfloor_prior":
        return None
    return dict(
        band_flux_ujy=catalog_band_fluxes_ujy(catalog),
        predict_fn=predict_flux_ujy,
        sigma_frac=float(config.prior_sigma_frac),
        sigma_min_mjy=float(config.prior_sigma_min_ujy) * 1e-3,
    )
