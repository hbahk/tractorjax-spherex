"""Backend-neutral per-cutout preparation shared by both backends.

Everything here is pure numpy/astropy: background refinement, MJy/sr -> mJy/pixel
scaling, inverse-variance masking, source projection, and PSF-zone selection.
The JAX and CPU backends consume identical prepared pixels, which is what makes
the cross-backend consistency test meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .background import fit_background
from .constants import IMG_SCALE, MASKBITS
from .io.cutouts import Cutout, cutout_pixel_area_sr


@dataclass
class PreparedPixels:
    """Background-subtracted, unit-scaled data and inverse variance (mJy/pixel)."""

    data: np.ndarray
    invvar: np.ndarray
    omega_sr: np.ndarray
    background: np.ndarray


def cutout_to_orig(x_cut, y_cut, *, crpix1a, crpix2a):
    """Map a 0-based cutout pixel to a 0-based original detector pixel.

    ``CRPIX*A`` are the 1-based detector positions of the cutout's (0,0) pixel
    (matching the ``spherex_retrieval`` convention).
    """
    return (1.0 + (x_cut - crpix1a), 1.0 + (y_cut - crpix2a))


def select_zone_plane(psf_zones_tab, x_orig, y_orig):
    """Local PSF-cube plane index nearest ``(x_orig, y_orig)`` in detector pixels."""
    dx = psf_zones_tab["x"] - x_orig
    dy = psf_zones_tab["y"] - y_orig
    return int(psf_zones_tab["plane_idx"][np.argmin(dx * dx + dy * dy)])


def downsample_psf_oversample2(psf):
    """Downsample 2x while preserving center and total sum (10x -> 5x oversample).

    Lifted verbatim from the production driver so the 5x PSF stamp handed to both
    backends is bit-identical to the reference pipeline.

    The strided quadrant slices only line up when each side length is
    ``== 1 (mod 4)`` (true for the standard 101x101 SPHEREx PSF cube plane ->
    51x51); guard other sizes with an actionable error instead of a bare NumPy
    broadcast failure.
    """
    h, w = psf.shape
    if h % 4 != 1 or w % 4 != 1:
        raise ValueError(
            f"downsample_psf_oversample2 needs each PSF side length == 1 (mod 4) "
            f"(e.g. 101); got shape {psf.shape}. The delivered SPHEREx PSF cube "
            f"planes are 101x101; a different size is unexpected.")
    cy, cx = h // 2, w // 2
    oh, ow = h // 2 + 1, w // 2 + 1
    ocy, ocx = oh // 2, ow // 2

    out = np.zeros((oh, ow), dtype=psf.dtype)
    out[0:ocy, 0:ocx] = 0.25 * (
        psf[0:cy:2, 0:cx:2] + psf[1:cy:2, 0:cx:2]
        + psf[0:cy:2, 1:cx:2] + psf[1:cy:2, 1:cx:2]
    )
    out[0:ocy, ocx + 1:ow] = 0.25 * (
        psf[0:cy:2, cx + 1:w:2] + psf[1:cy:2, cx + 1:w:2]
        + psf[0:cy:2, cx + 2:w:2] + psf[1:cy:2, cx + 2:w:2]
    )
    out[ocy + 1:oh, 0:ocx] = 0.25 * (
        psf[cy + 1:h:2, 0:cx:2] + psf[cy + 2:h:2, 0:cx:2]
        + psf[cy + 1:h:2, 1:cx:2] + psf[cy + 2:h:2, 1:cx:2]
    )
    out[ocy + 1:oh, ocx + 1:ow] = 0.25 * (
        psf[cy + 1:h:2, cx + 1:w:2] + psf[cy + 2:h:2, cx + 1:w:2]
        + psf[cy + 1:h:2, cx + 2:w:2] + psf[cy + 2:h:2, cx + 2:w:2]
    )
    out[ocy, 0:ocx] = 0.5 * (psf[cy, 0:cx:2] + psf[cy, 1:cx:2])
    out[ocy, ocx + 1:ow] = 0.5 * (psf[cy, cx + 1:w:2] + psf[cy, cx + 2:w:2])
    out[0:ocy, ocx] = 0.5 * (psf[0:cy:2, cx] + psf[1:cy:2, cx])
    out[ocy + 1:oh, ocx] = 0.5 * (psf[cy + 1:h:2, cx] + psf[cy + 2:h:2, cx])
    out[ocy, ocx] = psf[cy, cx]

    total = psf.sum()
    out_sum = out.sum()
    if out_sum != 0:
        out *= total / out_sum
    return out


def prepare_pixels(cutout: Cutout, config) -> PreparedPixels:
    """Background-fit, unit-scale, and mask one cutout into mJy/pixel space.

    Reproduces the production driver's pixel stage: ZODI-based background refined
    by ``config.bkg_model``; ``data = (img - bkg) * omega_sr * IMG_SCALE``;
    inverse variance zeroed where flags hit ``MASKBITS`` or variance is invalid.
    """
    img = cutout["image"]
    flg = cutout["flags"]
    var = cutout["variance"]

    bkg = fit_background(img, cutout["zodi"], flg, var, cutout.get("cwave_map"), config)

    omega_sr = cutout_pixel_area_sr(cutout).astype(img.dtype, copy=False)
    img_scaled = img * omega_sr * IMG_SCALE
    bkg_scaled = bkg * omega_sr * IMG_SCALE
    var_scaled = var * (omega_sr ** 2) * (IMG_SCALE ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        invvar = 1.0 / var_scaled
    invvar[~np.isfinite(invvar)] = 0.0
    invvar[(flg & MASKBITS) != 0] = 0.0

    data = img_scaled - bkg_scaled
    data[~np.isfinite(data)] = 0.0
    return PreparedPixels(data=data, invvar=invvar, omega_sr=omega_sr,
                          background=bkg_scaled)


def project_sources(cutout: Cutout, sco_all):
    """Project a catalog SkyCoord array to cutout pixel positions ``(sx, sy)``."""
    pxs, pys = cutout["wcs"].world_to_pixel(sco_all)
    return np.asarray(pxs, dtype=np.float64), np.asarray(pys, dtype=np.float64)


def select_psf_native(cutout: Cutout, x_ref, y_ref) -> np.ndarray:
    """Pick the PSF-cube plane nearest a reference pixel and 2x-downsample it.

    Returns the 5x-oversampled native PSF stamp (10x cube -> 5x). ``x_ref/y_ref``
    are cutout pixel coordinates.

    The SPHEREx PSF varies across the focal plane and the L2 cube ships one
    plane per PSF zone (~185 detector px pitch), so this must be called per
    TILE, not once per cutout: any cutout wider than the zone pitch spans
    several zones. Use :func:`zone_psf_selector`, which caches the downsample.
    """
    x_orig, y_orig = cutout_to_orig(x_ref, y_ref,
                                    crpix1a=cutout["crpix1a"],
                                    crpix2a=cutout["crpix2a"])
    plane = select_zone_plane(cutout["psf_zones"], x_orig, y_orig)
    return downsample_psf_oversample2(cutout["psf_cube"][plane])


def zone_psf_selector(cutout: Cutout):
    """Return ``f(x_cut, y_cut) -> native PSF stamp`` for this cutout.

    Each distinct zone plane is downsampled at most once, so a 40-tile cutout
    spanning 4 zones pays 4 downsamples rather than 40. Single-zone cutouts
    (the PSF cube has one plane) return the same array for every tile, which is
    exactly the previous behaviour.
    """
    zones = cutout["psf_zones"]
    cube = cutout["psf_cube"]
    crpix1a = cutout["crpix1a"]
    crpix2a = cutout["crpix2a"]
    cache: dict[int, np.ndarray] = {}

    def select(x_cut, y_cut):
        x_orig, y_orig = cutout_to_orig(x_cut, y_cut,
                                        crpix1a=crpix1a, crpix2a=crpix2a)
        plane = select_zone_plane(zones, x_orig, y_orig)
        stamp = cache.get(plane)
        if stamp is None:
            stamp = downsample_psf_oversample2(cube[plane])
            cache[plane] = stamp
        return stamp

    return select
