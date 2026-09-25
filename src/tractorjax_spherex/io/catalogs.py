"""Reference-catalog loading, normalization, depth cuts, and source matching.

The pipeline does *reference-catalog* forced photometry: source positions and
shapes come from an input catalog and are held fixed; only fluxes are solved.
The canonical internal schema (see the *Reference catalogs* docs page):

- ``id`` (int64, required) — unique source id (``ls_id`` accepted as an alias).
- ``ra``, ``dec`` (required) — ICRS position, degrees.
- ``flux_z`` (nanomaggies) — z-band flux for AB zmag = 22.5 - 2.5 log10(flux_z);
  only needed for depth cuts / lasso protection.
- ``shape_r`` (arcsec) — effective radius; 0 (or missing) => point source.
- ``sersic`` — Sersic index (galaxies only).
- ``shape_e1`` / ``shape_e2`` — ellipticity components (0 => round).
- ``dered_flux_*`` / ``flux_*`` (nanomaggies) — SED bands g/r/i/z/w1/w2, only
  used by the ``eigfloor_prior`` solver.
"""

from __future__ import annotations

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from ..models import ls_shapes_to_ab_phi

logger = logging.getLogger(__name__)


def load_catalog(path_or_table) -> Table:
    """Load a reference catalog from a path (parquet/fits/ecsv/...) or Table."""
    if isinstance(path_or_table, Table):
        return path_or_table.copy()
    return Table.read(path_or_table)


def normalize_catalog(tab: Table) -> Table:
    """Return a copy with the canonical columns filled in.

    Adds a canonical ``id`` (from ``id`` or ``ls_id``), fills missing shape
    columns with zeros, and derives ``shape_ab`` / ``shape_phi`` from the
    ellipticity via :func:`~tractorjax_spherex.models.ls_shapes_to_ab_phi`.
    Original columns (including SED bands) are preserved.
    """
    tab = tab.copy()
    n = len(tab)

    if "id" not in tab.colnames:
        if "ls_id" in tab.colnames:
            tab["id"] = np.asarray(tab["ls_id"]).astype(np.int64)
        else:
            raise ValueError(
                "catalog must have an 'id' or 'ls_id' column (unique source id)")
    if "ls_id" not in tab.colnames:
        tab["ls_id"] = np.asarray(tab["id"]).astype(np.int64)

    for col in ("ra", "dec"):
        if col not in tab.colnames:
            raise ValueError(f"catalog must have a '{col}' column (ICRS degrees)")

    if "flux_z" not in tab.colnames:
        tab["flux_z"] = np.full(n, np.nan)
    for col, fill in (("shape_r", 0.0), ("sersic", 1.0),
                      ("shape_e1", 0.0), ("shape_e2", 0.0)):
        if col not in tab.colnames:
            tab[col] = np.full(n, fill)

    tab["shape_ab"], tab["shape_phi"] = ls_shapes_to_ab_phi(
        tab["shape_e1"], tab["shape_e2"])
    return tab


def zmag_from_flux_z(flux_z):
    """AB z-band magnitude from Legacy Survey nanomaggies (NaN where flux<=0)."""
    fz = np.asarray(flux_z, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 22.5 - 2.5 * np.log10(fz)


def find_nearest_source(tab: Table, ra_deg: float, dec_deg: float):
    """Return ``(index, all_source_skycoord)`` for the catalog row nearest
    ``(ra, dec)``."""
    sco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    sep = sco.separation(target)
    return int(np.argmin(sep)), sco


def nearest_sources(tab: Table, ra_deg, dec_deg) -> np.ndarray:
    """Row index of the catalog source nearest each ``(ra, dec)`` (vectorised
    :func:`find_nearest_source`, for fields holding several targets)."""
    sco = SkyCoord(ra=tab["ra"], dec=tab["dec"], unit="deg")
    pts = SkyCoord(ra=np.atleast_1d(ra_deg) * u.deg, dec=np.atleast_1d(dec_deg) * u.deg)
    idx, _, _ = pts.match_to_catalog_sky(sco)
    return np.asarray(idx, dtype=int)


def apply_depth_cut(tab: Table, fit_zmag_max, keep_indices=()):
    """Prune to sources with z-band AB mag brighter than ``fit_zmag_max``.

    Sources with ``flux_z <= 0`` (undefined z mag) are dropped; indices in
    ``keep_indices`` are kept unconditionally (the pointing target). Returns
    ``(pruned_table, kept_original_indices)``. ``fit_zmag_max`` None or <= 0
    disables the cut (returns the table unchanged).

    A catalog with **no usable** ``flux_z`` at all (column missing, or every
    value NaN / non-positive) also returns unchanged, with a warning: the cut
    is on by default, and silently reducing such a catalog to the single kept
    target would be a far worse product than fitting it at its own depth.
    """
    if fit_zmag_max is None or fit_zmag_max <= 0:
        return tab, np.arange(len(tab))
    zmag = zmag_from_flux_z(tab["flux_z"])
    if not np.any(np.isfinite(zmag)):
        logger.warning(
            "fit_zmag_max=%.2f requested but the catalog has no usable flux_z "
            "(missing, NaN or <= 0 everywhere): fitting all %d sources at the "
            "catalog's own depth. Add a z-band flux in nanomaggies to enable "
            "the depth cut, or set fit_zmag_max=None to silence this.",
            fit_zmag_max, len(tab))
        return tab, np.arange(len(tab))
    keep = np.isfinite(zmag) & (zmag < fit_zmag_max)
    for idx in keep_indices:
        keep[int(idx)] = True
    kept = np.where(keep)[0]
    return tab[keep], kept


def protected_indices(tab: Table, protect_zmag_max, always=()):
    """Set of catalog indices for *protected* (unpenalized) reported targets:
    brighter than ``protect_zmag_max`` in z, plus the ``always`` indices."""
    zmag = zmag_from_flux_z(tab["flux_z"])
    protect = set(np.where(np.isfinite(zmag) & (zmag < protect_zmag_max))[0].tolist())
    for idx in always:
        protect.add(int(idx))
    return protect
