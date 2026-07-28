"""Fetch a Legacy Survey DR10 reference catalog from NOIRLab Data Lab.

Optional feature (``pip install 'spherex-photometry[catalog]'``). Users may
instead supply their own catalog with the columns documented in
:mod:`spherex_photometry.io.catalogs`. Lifted from the project's
``fetch_ls_catalog_newfield.py``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

logger = logging.getLogger(__name__)

SPHEREX_PIXSCALE = 6.15

DEFAULT_COLUMNS = [
    "ls_id", "type", "ra", "dec",
    "flux_g", "flux_ivar_g", "flux_r", "flux_ivar_r",
    "flux_i", "flux_ivar_i", "flux_z", "flux_ivar_z",
    "flux_w1", "flux_ivar_w1", "flux_w2", "flux_ivar_w2",
    "mag_g", "mag_r", "mag_i", "mag_z",
    "sersic", "shape_r", "shape_e1", "shape_e2",
    "dered_flux_g", "dered_flux_r", "dered_flux_i", "dered_flux_z",
    "dered_flux_w1", "dered_flux_w2",
]

_DATALAB_HINT = (
    "fetch_ls_dr10 needs the NOIRLab Data Lab client, which is optional:\n"
    "    pip install 'spherex-photometry[catalog]'\n"
    "Or supply your own reference catalog (see docs/catalogs).")


def fetch_ls_dr10(ra, dec, *, radius_deg=None, cutout_pixels=100,
                  columns=DEFAULT_COLUMNS, name="field", user=None,
                  password=None, out=None, poll_seconds=5.0,
                  timeout=1800.0, drop_dup=True) -> Table:
    """Query ``ls_dr10.tractor`` around ``(ra, dec)`` and return an astropy Table.

    Parameters
    ----------
    ra, dec : float
        Field center in degrees.
    radius_deg : float, optional
        Cone radius. Defaults to the half-diagonal of a ``cutout_pixels`` cutout
        so every source that could land in any cutout is captured.
    columns : list of str
        Columns to select (default: the standard forced-photometry set).
    user, password : str, optional
        Data Lab credentials (or ``DATALAB_USER`` / ``DATALAB_PASSWORD`` env).
    out : path, optional
        If given, write the result parquet there.
    drop_dup : bool
        Drop ``type == "DUP"`` rows (Legacy Survey duplicate entries).
    """
    try:
        from dl import authClient as ac
        from dl import queryClient as qc
        from dl.helpers.utils import convert
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(_DATALAB_HINT) from exc

    if radius_deg is None:
        radius_arcsec = 0.5 * math.sqrt(2) * cutout_pixels * SPHEREX_PIXSCALE
        radius_deg = radius_arcsec / 3600.0

    user = user or os.environ.get("DATALAB_USER")
    password = password or os.environ.get("DATALAB_PASSWORD")
    if not user or not password:
        raise ValueError("Data Lab credentials required (args or "
                         "DATALAB_USER/DATALAB_PASSWORD env)")
    ac.login(user, password)
    logger.info("Logged into Data Lab as %s", ac.whoAmI())

    cols = "*" if columns in ("*", None) else ", ".join(columns)
    sql = (f"SELECT {cols} FROM ls_dr10.tractor "
           f"WHERE q3c_radial_query(ra, dec, {ra}, {dec}, {radius_deg})")
    jobid = qc.query(sql=sql, out=f"vos://tmp/ls_{name}.csv", async_=True)
    logger.info("Submitted Data Lab job %s (radius=%.4f deg)", jobid, radius_deg)

    t0 = time.time()
    while True:
        st = qc.status(jobid)
        if st == "COMPLETED":
            tab = convert(qc.results(jobid), "table")
            break
        if st == "ERROR":
            try:
                err = qc.error(jobid)
            except Exception:  # noqa: BLE001 - best-effort detail for the error below
                err = "<no message>"
            raise RuntimeError(f"Data Lab query failed (job={jobid}): {err}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Data Lab query {jobid} timed out ({timeout:.0f}s)")
        time.sleep(poll_seconds)

    for col in tab.colnames:
        if tab[col].dtype == "float64":
            tab[col] = tab[col].astype("float32")
    if drop_dup and "type" in tab.colnames:
        n0 = len(tab)
        tab = tab[tab["type"] != "DUP"]
        logger.info("Retrieved %d rows (%d after dropping DUP)", n0, len(tab))
    if "ls_id" in tab.colnames and "id" not in tab.colnames:
        tab["id"] = np.asarray(tab["ls_id"]).astype(np.int64)

    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tab.write(str(out), overwrite=True)
        logger.info("Wrote %s", out)
    return tab
