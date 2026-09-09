"""Assemble per-source spectrophotometry from the per-visit photometry table.

The pipeline emits one row per (source, cutout) — a spectrophotometric point at
that visit's wavelength. :func:`build_spectra` groups those into per-source
spectra sorted by wavelength (which transparently handles the within-detector
wavelength reversal), and :func:`bin_spectrum` combines repeat visits with
inverse-variance weighting. This is the user-facing end product.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table


def to_ab_mag(flux_mjy, flux_err_mjy=None):
    """AB magnitude (and optional error) from flux in mJy."""
    flux_mjy = np.asarray(flux_mjy, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(flux_mjy * 1e-3 / 3631.0)
    if flux_err_mjy is None:
        return mag
    flux_err_mjy = np.asarray(flux_err_mjy, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_err = (2.5 / np.log(10.0)) * (flux_err_mjy / flux_mjy)
    return mag, mag_err


def build_spectra(photometry: Table, ids=None, min_snr=None) -> dict[int, Table]:
    """Group a photometry table into per-source spectra sorted by wavelength.

    Parameters
    ----------
    photometry : Table
        Output of :func:`~tractorjax_spherex.pipeline.run_photometry` (columns
        ``id, central_wavelength, flux, flux_err, ...``).
    ids : iterable of int, optional
        Restrict to these source ids (default: all).
    min_snr : float, optional
        Drop points with ``flux / flux_err`` below this threshold.

    Returns
    -------
    dict[int, Table]
        ``id -> spectrum table`` sorted by ``central_wavelength`` (NaN-wavelength
        points are dropped into no group). Each spectrum keeps ``detector``,
        ``obs_id``, ``cutout_index`` for provenance.
    """
    idcol = np.asarray(photometry["id"])
    want = {int(i) for i in ids} if ids is not None else None
    out: dict[int, Table] = {}
    for sid in np.unique(idcol):
        if want is not None and int(sid) not in want:
            continue
        sub = photometry[idcol == sid]
        wl = np.asarray(sub["central_wavelength"], dtype=np.float64)
        good = np.isfinite(wl) & np.isfinite(np.asarray(sub["flux"], dtype=float))
        if min_snr is not None:
            fe = np.asarray(sub["flux_err"], dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = np.asarray(sub["flux"], dtype=float) / fe
            good &= np.isfinite(snr) & (snr >= min_snr)
        sub = sub[good]
        if len(sub) == 0:
            continue
        sub.sort("central_wavelength")
        out[int(sid)] = sub
    return out


def bin_spectrum(spectrum: Table, dlam: float | None = None,
                 edges=None) -> Table:
    """Inverse-variance-weighted binning of a per-source spectrum.

    Parameters
    ----------
    spectrum : Table
        A single source's spectrum (from :func:`build_spectra`).
    dlam : float, optional
        Uniform bin width in micron. If both ``dlam`` and ``edges`` are None the
        spectrum is returned unbinned (one row per visit).
    edges : array, optional
        Explicit bin edges in micron (overrides ``dlam``).

    Returns
    -------
    Table
        Columns ``central_wavelength, flux, flux_err, n`` — the ivar-weighted
        mean flux and its error per bin (bins with no finite-ivar point dropped).
    """
    wl = np.asarray(spectrum["central_wavelength"], dtype=np.float64)
    fl = np.asarray(spectrum["flux"], dtype=np.float64)
    fe = np.asarray(spectrum["flux_err"], dtype=np.float64)
    if edges is None and dlam is None:
        return Table({"central_wavelength": wl, "flux": fl, "flux_err": fe,
                      "n": np.ones(len(wl), dtype=np.int64)})
    if edges is None:
        lo, hi = float(np.nanmin(wl)), float(np.nanmax(wl))
        edges = np.arange(lo, hi + dlam, dlam)
        if len(edges) < 2:  # all points within one bin width
            edges = np.array([lo - dlam / 2.0, hi + dlam / 2.0])
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = 1.0 / fe ** 2
    ok = np.isfinite(wl) & np.isfinite(fl) & np.isfinite(ivar) & (ivar > 0)
    wl, fl, ivar = wl[ok], fl[ok], ivar[ok]
    idx = np.digitize(wl, edges) - 1
    # Close the last bin on the right (np.histogram convention): points landing
    # exactly on (or above) the top edge would otherwise fall into a nonexistent
    # bin and be silently dropped — e.g. the reddest point when the auto edge
    # equals max(wl).
    idx[idx == len(edges) - 1] = len(edges) - 2
    rows = []
    for b in range(len(edges) - 1):
        sel = idx == b
        if not np.any(sel):
            continue
        w = ivar[sel]
        wsum = w.sum()
        fbar = np.sum(w * fl[sel]) / wsum
        lbar = np.sum(w * wl[sel]) / wsum
        rows.append((lbar, fbar, 1.0 / np.sqrt(wsum), int(sel.sum())))
    if not rows:
        return Table(names=("central_wavelength", "flux", "flux_err", "n"),
                     dtype=("f8", "f8", "f8", "i8"))
    arr = np.array(rows)
    return Table({"central_wavelength": arr[:, 0], "flux": arr[:, 1],
                  "flux_err": arr[:, 2], "n": arr[:, 3].astype(np.int64)})


def plot_spectrum(spectrum: Table, ax=None, binned: Table | None = None,
                  label=None, **kw):
    """Plot one source's spectrum (points + optional binned overlay)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.errorbar(spectrum["central_wavelength"], spectrum["flux"],
                yerr=spectrum["flux_err"], fmt=".", alpha=0.5,
                label=label or "per-visit", **kw)
    if binned is not None and len(binned):
        ax.errorbar(binned["central_wavelength"], binned["flux"],
                    yerr=binned["flux_err"], fmt="o-", color="k", label="binned")
    ax.set_xlabel("Central wavelength (μm)")
    ax.set_ylabel("Flux (mJy)")
    ax.legend()
    return ax
