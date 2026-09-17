"""Assemble per-source spectrophotometry from the per-visit photometry table.

The pipeline emits one row per (source, cutout) — a spectrophotometric point at
that visit's wavelength. :func:`build_spectra` groups those into per-source
spectra sorted by wavelength (which transparently handles the within-detector
wavelength reversal); :func:`bin_to_channels` combines repeat visits on the
102 SPHEREx spectral channels and :func:`bin_spectrum` on a uniform wavelength
grid, both with inverse-variance weighting. This is the user-facing end product.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

# The SPHEREx spectral channel definition: each of the six detector bands is
# split into 17 channels of constant resolving power (geometric spacing)
# between its band edges [um] (Crill et al. 2020, the SPHEREx public band
# parameters). Adjacent bands overlap by a few nm, so a visit is assigned to a
# channel by its DETECTOR and wavelength together, never by wavelength alone.
SPHEREX_BANDS = ((1, 0.75, 1.12), (2, 1.10, 1.65), (3, 1.63, 2.44),
                 (4, 2.40, 3.85), (5, 3.81, 4.41), (6, 4.41, 5.01))
CHANNELS_PER_BAND = 17
N_CHANNELS = CHANNELS_PER_BAND * len(SPHEREX_BANDS)   # 102


def spherex_channels() -> Table:
    """The 102 fiducial SPHEREx channels: ``channel`` (1-based, ascending in
    wavelength), ``detector``, ``lambda_min``, ``lambda_max`` and the channel
    centre ``central_wavelength`` [um]."""
    det, lo, hi = [], [], []
    for band, lmin, lmax in SPHEREX_BANDS:
        edges = np.geomspace(lmin, lmax, CHANNELS_PER_BAND + 1)
        det.append(np.full(CHANNELS_PER_BAND, band, dtype=np.int64))
        lo.append(edges[:-1])
        hi.append(edges[1:])
    det, lo, hi = np.concatenate(det), np.concatenate(lo), np.concatenate(hi)
    return Table({"channel": np.arange(1, N_CHANNELS + 1, dtype=np.int64),
                  "detector": det, "lambda_min": lo, "lambda_max": hi,
                  "central_wavelength": 0.5 * (lo + hi)})


def channel_index(central_wavelength, detector) -> np.ndarray:
    """0-based fiducial channel per visit (``-1`` where the visit's wavelength
    falls outside its own detector's band, e.g. a NaN or an edge pixel)."""
    wl = np.asarray(central_wavelength, dtype=np.float64)
    det = np.asarray(detector, dtype=np.int64)
    ch = spherex_channels()
    out = np.full(wl.shape, -1, dtype=np.int64)
    for k in range(N_CHANNELS):
        m = ((det == ch["detector"][k]) & (wl >= ch["lambda_min"][k])
             & (wl < ch["lambda_max"][k]))
        out[m] = k
    return out


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


def bin_to_channels(spectrum: Table) -> Table:
    """Inverse-variance-weighted mean per fiducial SPHEREx channel.

    Each visit is assigned to the channel of *its own detector* whose
    ``[lambda_min, lambda_max)`` contains its ``central_wavelength``
    (:func:`channel_index`); visits outside their detector's band, or with no
    finite flux / error, are dropped.

    Parameters
    ----------
    spectrum : Table
        A single source's spectrum from :func:`build_spectra` (needs the
        ``detector`` column it keeps).

    Returns
    -------
    Table
        One row per populated channel, sorted by wavelength: ``channel``,
        ``detector``, ``central_wavelength`` (the channel centre),
        ``lambda_min``, ``lambda_max``, ``lambda_mean`` (ivar-weighted mean of
        the visits' own wavelengths), ``flux``, ``flux_err``, ``n``.
    """
    ch = spherex_channels()
    wl = np.asarray(spectrum["central_wavelength"], dtype=np.float64)
    fl = np.asarray(spectrum["flux"], dtype=np.float64)
    fe = np.asarray(spectrum["flux_err"], dtype=np.float64)
    idx = channel_index(wl, spectrum["detector"])
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = 1.0 / fe ** 2
    ok = (idx >= 0) & np.isfinite(fl) & np.isfinite(ivar) & (ivar > 0)
    rows = []
    for k in np.unique(idx[ok]):
        sel = ok & (idx == k)
        w = ivar[sel]
        wsum = w.sum()
        rows.append((int(ch["channel"][k]), int(ch["detector"][k]),
                     float(ch["central_wavelength"][k]),
                     float(ch["lambda_min"][k]), float(ch["lambda_max"][k]),
                     float(np.sum(w * wl[sel]) / wsum),
                     float(np.sum(w * fl[sel]) / wsum), float(1.0 / np.sqrt(wsum)),
                     int(sel.sum())))
    names = ("channel", "detector", "central_wavelength", "lambda_min",
             "lambda_max", "lambda_mean", "flux", "flux_err", "n")
    dtype = ("i8", "i8", "f8", "f8", "f8", "f8", "f8", "f8", "i8")
    out = Table(rows=rows, names=names, dtype=dtype) if rows else \
        Table(names=names, dtype=dtype)
    out.sort("central_wavelength")
    return out


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
