"""Per-visit quality flags from the fit diagnostics.

``run_photometry`` with ``visit_diagnostics`` on writes, per row, ``fit_chi2``
(the template-weighted normalized squared residual over the source's unmasked
pixels) and ``mask_frac``; :func:`quality_flags` turns them into the bitmask
``quality_flag`` (:data:`~tractorjax_spherex.constants.QUALITY_BITS`):

- ``BAD_FIT`` when ``fit_chi2`` exceeds ``rel_max`` times the median
  ``fit_chi2`` of the same source over the table. Relative, because a bright
  source's residuals grow with its flux (PSF mismatch), so one absolute cut
  would drop whole channels of bright sources while missing faint outliers.
- ``NO_DATA`` when ``fit_chi2`` is undefined: no unmasked pixel is left under
  the source's template.

The cut is on the fit, never on the spectrum's shape, so a real emission line
is not clipped. On the 1,456 LSST DP1 QSOs (about 400k visits) ``rel_max=10``
flags 0.5% of the visits and removes a third of the >5 sigma channel spikes of
the binned spectra; an unflagged cold pixel next to a target (value ~0 with a
small variance) came out at 200x its source's median.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .constants import QUALITY_BAD_FIT, QUALITY_NO_DATA


def quality_flags(table: Table, rel_max: float | None = 10.0) -> np.ndarray:
    """``quality_flag`` (int16 bitmask) for every row of a photometry table
    carrying ``id`` and ``fit_chi2``; zeros when there is no ``fit_chi2``."""
    n = len(table)
    flag = np.zeros(n, dtype=np.int16)
    if "fit_chi2" not in table.colnames or n == 0:
        return flag
    chi2 = np.asarray(table["fit_chi2"], dtype=np.float64)
    finite = np.isfinite(chi2)
    flag[~finite] |= QUALITY_NO_DATA
    if rel_max is None:
        return flag
    ids = np.asarray(table["id"])
    order = np.argsort(ids, kind="stable")
    cuts = np.flatnonzero(np.diff(ids[order])) + 1
    med = np.full(n, np.nan)
    for idx in np.split(order, cuts):
        good = idx[finite[idx]]
        if good.size:
            med[idx] = np.median(chi2[good])
    bad = finite & np.isfinite(med) & (chi2 > rel_max * med)
    flag[bad] |= QUALITY_BAD_FIT
    return flag


def add_quality_flags(table: Table, rel_max: float | None = 10.0) -> Table:
    """Set (or refresh) the ``quality_flag`` column in place; returns the table."""
    if "fit_chi2" in table.colnames:
        table["quality_flag"] = quality_flags(table, rel_max)
    return table
