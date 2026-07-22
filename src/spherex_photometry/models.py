"""Catalog geometry helpers: ellipticity -> shape, sky->pixel position angle.

Lifted from ``spherex_gpupipe/utils.py``. These are pure numpy/astropy (no JAX);
:func:`get_profile_cached` lazily imports the tractor-jax Sersic mixture only
when the JAX backend actually needs it.
"""

from __future__ import annotations

from functools import lru_cache

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord


def ls_shapes_to_ab_phi(e1, e2):
    """Legacy Survey ellipticity components -> engine galaxy shape columns.

    Returns ``(shape_ab, shape_phi)`` where ``phi = -0.5*atan2(e2, e1)`` in
    degrees folded to ``[0, 180)`` — the sky-frame position angle the batch
    engine's ``get_galaxy_shape_matrix`` expects (the inverse of
    ``EllipseE.fromRAbPhi``). ``+0.5*atan2`` mirror-reflects the galaxy (the
    2026-07-04 PA sign bug); ``-theta`` reproduces the ``EllipseE(e1, e2)``
    covariance to ~1e-7.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    e = np.hypot(e1, e2)
    ab = (1.0 - e) / (1.0 + e)
    phi = (-0.5 * np.rad2deg(np.arctan2(e2, e1)) + 180.0) % 180.0
    return ab, phi


def sky_pa_to_pixel_pa(wcs, ra_deg, dec_deg, pa_sky_deg,
                       d_arcsec=1.0, y_down=False):
    """Convert a sky position angle (East-of-North) to the pixel frame.

    Probes the local WCS Jacobian with 1" steps toward East and North and maps
    the sky direction into the pixel basis. Used by the CPU backend to build
    pixel-frame galaxy position angles for ``tractor``'s ``GalaxyShape``.
    """
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x0, y0 = wcs.world_to_pixel(sc)
    d = (d_arcsec * u.arcsec).to(u.deg).value
    sc_E = SkyCoord(ra=(ra_deg + d / np.cos(np.deg2rad(dec_deg))) * u.deg,
                    dec=dec_deg * u.deg, frame="icrs")
    sc_N = SkyCoord(ra=ra_deg * u.deg, dec=(dec_deg + d) * u.deg, frame="icrs")
    xE, yE = wcs.world_to_pixel(sc_E)
    xN, yN = wcs.world_to_pixel(sc_N)
    vE = np.array([xE - x0, yE - y0])
    vN = np.array([xN - x0, yN - y0])
    if y_down:
        vE[1] *= -1.0
        vN[1] *= -1.0
    th = np.deg2rad(pa_sky_deg)
    d_sky = np.array([np.cos(th), np.sin(th)])
    M = np.column_stack([vE, vN])
    v_pix = M @ d_sky
    return np.rad2deg(np.arctan2(v_pix[1], v_pix[0]))


def sky_pa_to_pixel_pa_batch(wcs, ra_deg, dec_deg, pa_sky_deg,
                             d_arcsec=1.0, y_down=False):
    """Vectorized :func:`sky_pa_to_pixel_pa` (identical math, batched WCS calls)."""
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    pa = np.atleast_1d(np.asarray(pa_sky_deg, dtype=float))
    sc = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    x0, y0 = wcs.world_to_pixel(sc)
    d = (d_arcsec * u.arcsec).to(u.deg).value
    sc_E = SkyCoord(ra=(ra + d / np.cos(np.deg2rad(dec))) * u.deg,
                    dec=dec * u.deg, frame="icrs")
    sc_N = SkyCoord(ra=ra * u.deg, dec=(dec + d) * u.deg, frame="icrs")
    xE, yE = wcs.world_to_pixel(sc_E)
    xN, yN = wcs.world_to_pixel(sc_N)
    vEx, vEy = xE - x0, yE - y0
    vNx, vNy = xN - x0, yN - y0
    if y_down:
        vEy = -vEy
        vNy = -vNy
    th = np.deg2rad(pa)
    ct, st = np.cos(th), np.sin(th)
    v_pix_x = vEx * ct + vNx * st
    v_pix_y = vEy * ct + vNy * st
    return np.rad2deg(np.arctan2(v_pix_y, v_pix_x))


@lru_cache(maxsize=None)
def _profile_for_sersic(sersic_value):
    from tractor_jax.sersic import SersicMixture
    return SersicMixture.getProfile(sersic_value)


def get_profile_cached(sersic):
    """Memoized ``SersicMixture.getProfile`` keyed on the exact float value.

    Safe because callers only read ``prof.amp/mean/var``. Lazily imports
    tractor-jax so this module stays importable without it.
    """
    return _profile_for_sersic(float(sersic))
