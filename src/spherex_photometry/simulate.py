"""Synthetic SPHEREx cutouts + catalogs — try the pipeline with no network/data.

``make_synth_field`` writes a small field of spec-compliant cutout MEFs
(IMAGE/FLAGS/VARIANCE/ZODI/PSF/PSF_ZONES/CWAVE/CBAND/SAPM + ``summary.ecsv``)
with point/galaxy sources of *known* mJy flux injected via a pixel-integrated
Gaussian PSF; ``make_synth_catalog`` builds the matching reference catalog.

This is a **toy** simulator (Gaussian PSF, flat zodi, white noise) meant for
offline demos, tutorials, and the test suite — not a SPHEREx instrument
simulator. See ``examples/03_offline_demo.py`` and the *Worked example* docs
page for an end-to-end run: simulate -> photometer -> spectrum.
"""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .constants import IMG_SCALE, SPHEREX_PIXSCALE as PIXSCALE

OMEGA_SR = ((PIXSCALE * u.arcsec) ** 2).to_value(u.sr)


def gaussian_oversampled(size_over: int, oversamp: int, fwhm_native: float):
    """A normalized oversampled Gaussian PSF plane (sum == 1), centered."""
    sigma_over = (fwhm_native / 2.3548200450309493) * oversamp
    c = (size_over - 1) / 2.0
    yy, xx = np.mgrid[0:size_over, 0:size_over]
    g = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma_over ** 2))
    return g / g.sum()


def _pixel_integrated_gaussian(shape, x0, y0, fwhm_native, oversamp=10):
    """Pixel-integrated native Gaussian centered at (x0, y0), normalized to sum 1.

    Evaluates the analytic Gaussian on an ``oversamp``x sub-grid and sums each
    native pixel's sub-cells, matching the engine's oversampled-render-then-bin
    model (for a Gaussian PSF the point-source model IS the PSF). Point-sampling
    at native centers would badly under-represent the undersampled SPHEREx PSF.
    """
    sigma = fwhm_native / 2.3548200450309493
    ny, nx = shape
    # Sub-cell centers in native coordinates: pixel i spans [i-0.5, i+0.5].
    off = (np.arange(oversamp) + 0.5) / oversamp - 0.5
    xs = (np.arange(nx)[:, None] + off[None, :]).ravel()   # (nx*os,)
    ys = (np.arange(ny)[:, None] + off[None, :]).ravel()   # (ny*os,)
    gx = np.exp(-((xs - x0) ** 2) / (2.0 * sigma ** 2))
    gy = np.exp(-((ys - y0) ** 2) / (2.0 * sigma ** 2))
    gx = gx.reshape(nx, oversamp).sum(axis=1)
    gy = gy.reshape(ny, oversamp).sum(axis=1)
    g = np.outer(gy, gx)
    s = g.sum()
    return g / s if s > 0 else g


def make_synth_cutout(path, *, cutout_index=0, obsid="SYNTH0001", detector=1,
                      nx=40, ny=40, ra0=150.0, dec0=2.0, oversamp=10,
                      psf_size_over=101, fwhm_native=2.5,
                      sources=None, zodi_level=0.05, noise_mjy_sr=1e-4,
                      seed=0, empty_side_hdus=False, cwave_slope=0.02,
                      cwave_base=2.0):
    """Write one synthetic cutout MEF.

    ``sources`` is a list of dicts ``{x, y, flux_mjy, shape_r?}`` (pixel
    positions, mJy). Point sources (``shape_r`` 0/absent) are injected with a
    native-pixel Gaussian matching the PSF; galaxies are injected as a slightly
    broadened Gaussian (approximate — used only to exercise the galaxy path).

    Returns the list of injected sources with their pixel positions/fluxes.
    """
    rng = np.random.default_rng(seed)
    sources = sources or []

    # WCS (TAN), pixel scale PIXSCALE, reference at cutout center.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.crval = [ra0, dec0]
    w.wcs.cd = np.array([[-PIXSCALE / 3600.0, 0.0], [0.0, PIXSCALE / 3600.0]])

    # Model image in internal mJy/pixel units, then convert to MJy/sr.
    model = np.zeros((ny, nx), dtype=np.float64)
    for s in sources:
        x, y, f = float(s["x"]), float(s["y"]), float(s["flux_mjy"])
        fwhm = fwhm_native
        if s.get("shape_r", 0.0) > 0:
            fwhm = np.hypot(fwhm_native, 2.0 * s["shape_r"] / PIXSCALE)
        model += f * _pixel_integrated_gaussian((ny, nx), x, y, fwhm, oversamp)

    image = model / (OMEGA_SR * IMG_SCALE)               # mJy/pixel -> MJy/sr
    image += zodi_level                                  # flat zodi pedestal
    image += rng.normal(0.0, noise_mjy_sr, size=image.shape)

    variance = np.full((ny, nx), noise_mjy_sr ** 2, dtype=np.float64)
    flags = np.zeros((ny, nx), dtype=np.int32)
    zodi = np.full((ny, nx), zodi_level, dtype=np.float64)

    psf_plane = gaussian_oversampled(psf_size_over, oversamp, fwhm_native)
    psf_cube = psf_plane[None, :, :]
    psf_zones = Table({"zone_id": [1], "x": [nx / 2.0], "y": [ny / 2.0],
                       "plane_idx": [0]})

    cwave = cwave_base + cwave_slope * np.arange(nx)[None, :] * np.ones((ny, 1))
    cband = np.full((ny, nx), 0.01, dtype=np.float64)
    sapm = np.full((ny, nx), PIXSCALE ** 2, dtype=np.float64)  # arcsec^2

    ihdr = w.to_header()
    ihdr["CRPIX1A"] = 1.0   # detector origin of cutout (0,0)
    ihdr["CRPIX2A"] = 1.0
    ihdr["DETECTOR"] = detector

    phdr = fits.Header()
    phdr["OBSID"] = obsid
    phdr["DETECTOR"] = detector
    phdr["OVERSAMP"] = oversamp

    hdus = [
        fits.PrimaryHDU(header=phdr),
        fits.ImageHDU(image.astype(np.float32), header=ihdr, name="IMAGE"),
        fits.ImageHDU(flags, name="FLAGS"),
        fits.ImageHDU(variance.astype(np.float32), name="VARIANCE"),
        fits.ImageHDU(zodi.astype(np.float32), name="ZODI"),
        fits.ImageHDU(psf_cube.astype(np.float32), name="PSF"),
        fits.BinTableHDU(psf_zones, name="PSF_ZONES"),
    ]
    if empty_side_hdus:
        hdus += [fits.ImageHDU(np.zeros((0,), np.float32), name="CWAVE"),
                 fits.ImageHDU(np.zeros((0,), np.float32), name="CBAND"),
                 fits.ImageHDU(np.zeros((0,), np.float32), name="SAPM")]
    else:
        hdus += [fits.ImageHDU(cwave.astype(np.float32), name="CWAVE"),
                 fits.ImageHDU(cband.astype(np.float32), name="CBAND"),
                 fits.ImageHDU(sapm.astype(np.float32), name="SAPM")]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return sources, w


def make_synth_field(dirpath, *, n_cutouts=2, nx=40, ny=40, ra0=150.0,
                     dec0=2.0, sources=None, seed=0, noise_mjy_sr=1e-4):
    """Write a small field of cutouts + a summary.ecsv; return injected sources.

    ``noise_mjy_sr`` is the white-noise sigma in image units (MJy/sr); the
    default is nearly noiseless (tests). For demo figures with visible error
    bars use something like ``0.02`` (~0.02 mJy/pixel).
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    if sources is None:
        sources = [
            {"x": 12.0, "y": 12.0, "flux_mjy": 5.0},
            {"x": 27.0, "y": 20.0, "flux_mjy": 2.0},
            {"x": 20.0, "y": 30.0, "flux_mjy": 1.0, "shape_r": 1.5, "sersic": 1.0},
        ]
    summary_rows = []
    for k in range(n_cutouts):
        p = dirpath / f"cutout_{k:04d}_SYNTH{k:04d}_D1.fits"
        # each visit samples a different wavelength band, like real SPHEREx visits
        make_synth_cutout(p, cutout_index=k, obsid=f"SYNTH{k:04d}",
                          nx=nx, ny=ny, ra0=ra0, dec0=dec0,
                          sources=sources, seed=seed + k,
                          noise_mjy_sr=noise_mjy_sr,
                          cwave_base=1.0 + 0.6 * k)
        summary_rows.append((k, "ok"))
    summary = Table(rows=summary_rows, names=("cutout_index", "status"))
    summary.write(dirpath / "summary.ecsv", overwrite=True)
    return sources


def make_synth_catalog(path, sources, wcs, *, extra_bands=True):
    """Write a reference catalog matching injected ``sources`` (via ``wcs``)."""
    xs = np.array([s["x"] for s in sources])
    ys = np.array([s["y"] for s in sources])
    sc = wcs.pixel_to_world(xs, ys)
    n = len(sources)
    tab = Table()
    tab["id"] = np.arange(1, n + 1, dtype=np.int64)
    tab["ra"] = sc.ra.deg
    tab["dec"] = sc.dec.deg
    # flux_z (nanomaggies) roughly tracks the injected mJy so depth cuts work.
    tab["flux_z"] = np.array([max(s["flux_mjy"], 1e-3) * 1000.0 for s in sources])
    tab["shape_r"] = np.array([s.get("shape_r", 0.0) for s in sources])
    tab["sersic"] = np.array([s.get("sersic", 1.0) for s in sources])
    tab["shape_e1"] = np.zeros(n)
    tab["shape_e2"] = np.zeros(n)
    if extra_bands:
        for b in ("g", "r", "i", "z", "w1", "w2"):
            tab[f"flux_{b}"] = tab["flux_z"] * 0.5
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tab.write(str(path), overwrite=True)
    return tab
