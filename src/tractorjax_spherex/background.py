"""Background refinement models (run once per cutout, on top of the ZODI HDU).

Lifted from the production driver. Four models, selected by
``config.bkg_model``:

* ``photutils``       — ``Background2D`` on the ZODI-subtracted residual (default)
* ``cwave+photutils`` — first removes a smooth 1D ``B(CWAVE)`` airglow profile
                        (He I 1.083 um geocoronal line) then ``Background2D``
* ``plane``           — weighted least-squares tilted plane
* ``none``            — use the ZODI HDU as-is
"""

from __future__ import annotations

import numpy as np
from photutils.background import Background2D
from scipy.interpolate import PchipInterpolator

from .constants import MASKBITS, SOURCE_BIT


def build_background_mask(flg, var, maskbits=MASKBITS, source_bit=SOURCE_BIT):
    """Pixels to exclude from a background fit: bad flags, sources, bad variance."""
    bad = (flg & maskbits) != 0
    source = (flg & source_bit) != 0
    valid_var = np.isfinite(var) & (var > 0)
    return bad | source | (~valid_var)


def fit_background_plane(img, bkg, flg, var):
    """Add a weighted-least-squares tilted plane to ``bkg``."""
    if img.size == 0:
        return bkg
    mask = build_background_mask(flg, var)
    valid = ~mask
    if not np.any(valid):
        return bkg
    z = (img - bkg)[valid].ravel()
    w = (1.0 / var[valid]).ravel()
    yy, xx = np.indices(img.shape)
    x = xx[valid].ravel().astype(np.float64)
    y = yy[valid].ravel().astype(np.float64)
    A = np.stack([np.ones_like(x), x, y], axis=1)
    Aw = A * w[:, None]
    AtAw = A.T @ Aw
    AtAz = A.T @ (z * w)
    try:
        a, b, c = np.linalg.solve(AtAw, AtAz)
    except np.linalg.LinAlgError:
        return bkg
    plane = (a + b * xx + c * yy).astype(bkg.dtype, copy=False)
    return bkg + plane


def fit_background_photutils(img, bkg, flg, var, box_size=10, filter_size=3):
    """Add a ``Background2D`` estimate of the ZODI-subtracted residual to ``bkg``."""
    if img.size == 0 or img.shape[0] < box_size or img.shape[1] < box_size:
        return bkg
    mask = build_background_mask(flg, var)
    residual = img - bkg
    try:
        bkg2d = Background2D(
            residual,
            box_size=(box_size, box_size),
            filter_size=(filter_size, filter_size),
            mask=mask,
        )
    except Exception:  # noqa: BLE001 - a failed Background2D just means "no 2-D term"
        return bkg
    return bkg + bkg2d.background.astype(bkg.dtype, copy=False)


def fit_background_cwave(img, bkg, flg, var, cwave_map,
                         nbins=48, min_per_bin=20,
                         box_size=10, filter_size=3):
    """Remove a smooth 1D ``B(CWAVE)`` profile then run ``Background2D``.

    Fits a monotone PCHIP profile to ``(img - bkg)`` binned by the per-pixel
    central wavelength (removing fixed-wavelength airglow lines such as the He I
    1.083 um geocoronal line, which the LVF maps onto iso-wavelength detector
    stripes), then runs the standard 2D background on the remainder. Falls back
    to :func:`fit_background_photutils` when no CWAVE map is available.
    """
    if cwave_map is None or img.size == 0:
        return fit_background_photutils(img, bkg, flg, var, box_size, filter_size)
    mask = build_background_mask(flg, var)
    residual = img - bkg
    ok = (~mask) & np.isfinite(residual) & np.isfinite(cwave_map)
    wl = np.asarray(cwave_map)[ok]
    dv = residual[ok]
    if wl.size < max(4 * min_per_bin, 100):
        return fit_background_photutils(img, bkg, flg, var, box_size, filter_size)
    edges = np.unique(np.quantile(wl, np.linspace(0.0, 1.0, nbins + 1)))
    if len(edges) < 3:
        bcw = np.full_like(residual, np.median(dv))
    else:
        bi = np.clip(np.digitize(wl, edges) - 1, 0, len(edges) - 2)
        xb, yb = [], []
        for b in range(len(edges) - 1):
            sel = bi == b
            if sel.sum() < min_per_bin:
                continue
            xb.append(np.median(wl[sel]))
            yb.append(np.median(dv[sel]))
        if len(xb) < 2:
            bcw = np.full_like(residual, np.median(dv))
        else:
            itp = PchipInterpolator(np.asarray(xb), np.asarray(yb),
                                    extrapolate=False)
            bcw = itp(np.clip(cwave_map, xb[0], xb[-1]))
            bcw = np.where(np.isfinite(bcw), bcw, np.median(dv))
    bkg_cw = bkg + bcw.astype(bkg.dtype, copy=False)
    return fit_background_photutils(img, bkg_cw, flg, var, box_size, filter_size)


def fit_background(img, zodi, flg, var, cwave_map, config):
    """Dispatch to the background model named by ``config.bkg_model``.

    Returns the refined background image (the ZODI HDU refined in place is the
    base for all models). ``config.bkg_model == "none"`` returns ``zodi``.
    """
    model = config.bkg_model
    if model == "cwave+photutils":
        return fit_background_cwave(
            img, zodi, flg, var, cwave_map,
            nbins=config.bkg_cwave_nbins, min_per_bin=config.bkg_cwave_min_per_bin,
            box_size=config.bkg_box_size, filter_size=config.bkg_filter_size)
    if model == "photutils":
        return fit_background_photutils(
            img, zodi, flg, var, config.bkg_box_size, config.bkg_filter_size)
    if model == "plane":
        return fit_background_plane(img, zodi, flg, var)
    return zodi
