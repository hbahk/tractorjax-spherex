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


def zone_bilinear_weights(psf_zones_tab, x_orig, y_orig) -> np.ndarray:
    """Bilinear weights over the PSF-zone lattice at detector (x_orig, y_orig).

    Follows the SPHEREx Sky Simulator's own convention
    (``SPHEREx_InstrumentSimulator.psf.get_dist_weight``): plain bilinear
    between the four bracketing zone centres, CLAMPED at the lattice edge
    rather than extrapolated. Returns weights summing to 1, aligned with the
    rows of ``psf_zones_tab``, and degenerates to one-hot — hence identical to
    :func:`select_zone_plane` — on a single-zone cutout.
    """
    zx = np.asarray(psf_zones_tab["x"], dtype=np.float64)
    zy = np.asarray(psf_zones_tab["y"], dtype=np.float64)
    ux = np.unique(np.round(zx, 6))
    uy = np.unique(np.round(zy, 6))

    def _bracket(v, u):
        if u.size == 1:
            return u[0], u[0], 1.0
        i = int(np.clip(np.searchsorted(u, v) - 1, 0, u.size - 2))
        lo, hi = u[i], u[i + 1]
        return lo, hi, 1.0 - float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    x_lo, x_hi, wx = _bracket(float(x_orig), ux)
    y_lo, y_hi, wy = _bracket(float(y_orig), uy)
    w = np.zeros(zx.size, dtype=np.float64)
    for xv, wxx in ((x_lo, wx), (x_hi, 1.0 - wx)):
        for yv, wyy in ((y_lo, wy), (y_hi, 1.0 - wy)):
            if wxx * wyy == 0.0:
                continue
            hit = np.where((np.abs(zx - xv) < 1e-6) & (np.abs(zy - yv) < 1e-6))[0]
            if hit.size:
                w[hit[0]] += wxx * wyy
    if not np.any(w > 0):            # corner absent from the delivered subset
        w[np.argmin((zx - x_orig) ** 2 + (zy - y_orig) ** 2)] = 1.0
    return w / w.sum()


def zone_planes_and_weights(psf_zones_tab, x_orig, y_orig):
    """Vectorised :func:`select_zone_plane` + :func:`zone_bilinear_weights`.

    A tiled cutout calls both of those once per tile, and each call scans the
    whole zone table in Python; a 2040x2040 frame at tile 15 has 18,496 tiles.
    This does the same arithmetic for every tile at once.

    It is a pure re-expression, not an approximation: the same comparisons in
    the same order, including the tie-breaks (``argmin`` takes the FIRST
    minimum, the corner match takes the FIRST hit, an absent corner falls back
    to the nearest zone). ``tests/test_vector_zones.py`` asserts equality
    against the scalar helpers element by element.

    Parameters
    ----------
    psf_zones_tab : table with columns ``x``, ``y``, ``plane_idx``
    x_orig, y_orig : (n,) detector coordinates (see :func:`cutout_to_orig`)

    Returns
    -------
    planes : (n,) int
        Nearest zone's plane index — matches :func:`select_zone_plane`.
    rows : (n,) int
        Nearest zone's ROW in the table (the index ``planes`` was read from).
    weights : (n, K) float
        Bilinear weights aligned with the table rows, each row summing to 1 —
        matches :func:`zone_bilinear_weights`.
    """
    zx = np.asarray(psf_zones_tab["x"], dtype=np.float64)
    zy = np.asarray(psf_zones_tab["y"], dtype=np.float64)
    plane_idx = np.asarray(psf_zones_tab["plane_idx"])
    x = np.asarray(x_orig, dtype=np.float64).reshape(-1)
    y = np.asarray(y_orig, dtype=np.float64).reshape(-1)
    n, K = x.size, zx.size

    dx = zx[None, :] - x[:, None]
    dy = zy[None, :] - y[:, None]
    rows = np.argmin(dx * dx + dy * dy, axis=1)        # first minimum, as argmin
    planes = plane_idx[rows].astype(int)

    ux = np.unique(np.round(zx, 6))
    uy = np.unique(np.round(zy, 6))

    def _bracket(v, u):
        if u.size == 1:
            return np.full_like(v, u[0]), np.full_like(v, u[0]), np.ones_like(v)
        i = np.clip(np.searchsorted(u, v) - 1, 0, u.size - 2)
        lo, hi = u[i], u[i + 1]
        t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)    # clamp, no extrapolation
        return lo, hi, 1.0 - t

    x_lo, x_hi, wx = _bracket(x, ux)
    y_lo, y_hi, wy = _bracket(y, uy)

    weights = np.zeros((n, K), dtype=np.float64)
    for xv, wxx in ((x_lo, wx), (x_hi, 1.0 - wx)):
        for yv, wyy in ((y_lo, wy), (y_hi, 1.0 - wy)):
            w = wxx * wyy
            match = ((np.abs(zx[None, :] - xv[:, None]) < 1e-6)
                     & (np.abs(zy[None, :] - yv[:, None]) < 1e-6))
            has = match.any(axis=1)
            first = np.argmax(match, axis=1)           # hit[0] of the scalar code
            sel = has & (w != 0.0)
            if np.any(sel):
                idx = np.where(sel)[0]
                weights[idx, first[idx]] += w[idx]
    none = ~(weights > 0).any(axis=1)                  # corner absent: nearest
    if np.any(none):
        weights[np.where(none)[0], rows[none]] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return planes, rows, weights


def zone_lookup_vectorized(cutout: Cutout, x_cut, y_cut):
    """:func:`zone_planes_and_weights` at cutout pixel coordinates.

    Convenience wrapper that applies this cutout's ``CRPIX*A`` offset, so a
    caller with an array of tile centres gets planes and weights in one call.
    """
    x_orig, y_orig = cutout_to_orig(np.asarray(x_cut, dtype=np.float64),
                                    np.asarray(y_cut, dtype=np.float64),
                                    crpix1a=cutout["crpix1a"],
                                    crpix2a=cutout["crpix2a"])
    return zone_planes_and_weights(cutout["psf_zones"], x_orig, y_orig)


def zone_psf_basis(cutout: Cutout, cache=None):
    """Return ``(basis, f(x_cut, y_cut) -> weights)`` for a blended zone PSF.

    ``basis`` is the list of downsampled zone kernels — ONE object, because the
    engine keys its Fourier-transform cache on identity and blends in the
    Fourier domain, so the K transforms are shared by every tile instead of one
    transform per tile.

    Pass a :class:`~spherex_photometry.psf_cache.PSFCache` as ``cache`` to
    extend that sharing *across cutouts*: every cutout of one detector ships a
    byte-identical cube, so the same list object is handed back and the engine's
    transforms are reused instead of recomputed per cutout.

    Nearest-zone (:func:`zone_psf_selector`) leaves a tile using a kernel
    sampled up to ~93 detector px away, half the ~185 px zone pitch. Blending
    removes that; measured on A2055 flight data at z<21 it moves bright fluxes
    by p90 1.7% overall and 2.5% at 2-3 um, where the zone-to-zone kernel
    centroid spread is largest (0.08-0.12 px against 0.005-0.014 px at 4.9 um).

    Note the cutout must have been retrieved with a zone margin
    (``spherex_retrieval`` ``zone_margin >= 1``); bundles written before that
    keep only the zones the cutout bbox spans, which for a cutout smaller than
    the zone pitch is a single plane and makes this a no-op.
    """
    zones = cutout["psf_zones"]
    cube = cutout["psf_cube"]
    crpix1a = cutout["crpix1a"]
    crpix2a = cutout["crpix2a"]

    def _build():
        return [downsample_psf_oversample2(cube[int(p)])
                for p in np.asarray(zones["plane_idx"])]

    if cache is not None:
        from .psf_cache import cube_signature
        basis = cache.zone_basis(cube_signature(cutout), _build)
    else:
        basis = _build()

    def weights(x_cut, y_cut):
        x_orig, y_orig = cutout_to_orig(x_cut, y_cut,
                                        crpix1a=crpix1a, crpix2a=crpix2a)
        return zone_bilinear_weights(zones, x_orig, y_orig)

    return basis, weights


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
