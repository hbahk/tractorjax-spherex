"""Fit diagnostics: render the fitted model image and compare it to the data.

After :func:`~spherex_photometry.pipeline.run_photometry`, these helpers let a
user *see* the fit on any cutout: :func:`render_model_image` rebuilds the scene
(every fitted source at its measured flux, rendered through the same
5x-oversampled PSF machinery as the solver) and :func:`plot_fit` shows the
standard data / model / chi triptych. Large residuals localized on a source
usually mean a bad shape or a missing neighbour; structured background residuals
suggest trying another ``bkg_model``.

All fluxes are in the pipeline's internal mJy/pixel scale, so the panels are
directly comparable to the reported ``flux`` column.
"""

from __future__ import annotations

import numpy as np

from .config import PhotometryConfig
from .constants import SPHEREX_PIXSCALE
from .io.catalogs import normalize_catalog
from .models import sky_pa_to_pixel_pa_batch
from .prepare import prepare_pixels, project_sources, select_psf_native


def fluxes_for_cutout(result, cutout_index):
    """Extract ``{source id: fitted flux [mJy]}`` for one cutout from a
    photometry result table."""
    sub = result[np.asarray(result["cutout_index"]) == int(cutout_index)]
    return {int(i): float(f) for i, f in zip(sub["id"], sub["flux"])}


def render_model_image(cutout, catalog, flux_by_id, config=None):
    """Render the fitted model image for one cutout (mJy/pixel).

    Parameters
    ----------
    cutout : Cutout
        From :func:`~spherex_photometry.io.cutouts.read_cutout`.
    catalog : Table
        The reference catalog (normalized or raw; normalized internally).
    flux_by_id : dict
        ``{source id: flux [mJy]}`` — e.g. from :func:`fluxes_for_cutout`.
    config : PhotometryConfig, optional
        Only the background / rendering fields are used.

    Returns
    -------
    (model, prepared) : (ndarray, PreparedPixels)
        The model image and the prepared (background-subtracted, mJy/pixel)
        data/invvar it should be compared against.
    """
    from tractor_jax import (
        ConstantSky,
        Flux,
        Image,
        NullWCS,
        PixPos,
        PointSource,
        Tractor,
    )
    from tractor_jax.galaxy import GalaxyShape
    from tractor_jax.psf import PixelizedPSF
    from tractor_jax.sersic import SersicGalaxy, SersicIndex

    config = config or PhotometryConfig()
    catalog = normalize_catalog(catalog)
    prepared = prepare_pixels(cutout, config)
    H, W = prepared.data.shape

    from astropy.coordinates import SkyCoord
    sco = SkyCoord(ra=catalog["ra"], dec=catalog["dec"], unit="deg")
    sx, sy = project_sources(cutout, sco)

    ids = np.asarray(catalog["id"], dtype=np.int64)
    shape_r = np.asarray(catalog["shape_r"], dtype=np.float64)
    shape_ab = np.asarray(catalog["shape_ab"], dtype=np.float64)
    shape_phi = np.asarray(catalog["shape_phi"], dtype=np.float64)
    sersic = np.asarray(catalog["sersic"], dtype=np.float64)

    keep = np.array([int(i) in flux_by_id for i in ids])
    keep &= np.isfinite(sx) & np.isfinite(sy)
    idxs = np.where(keep)[0]

    is_gal = np.isfinite(shape_r) & (shape_r > 0)
    gal_idxs = idxs[is_gal[idxs]]
    phi_pix = {}
    if len(gal_idxs):
        pp = sky_pa_to_pixel_pa_batch(
            cutout["wcs"],
            np.asarray(catalog["ra"], dtype=np.float64)[gal_idxs],
            np.asarray(catalog["dec"], dtype=np.float64)[gal_idxs],
            shape_phi[gal_idxs])
        phi_pix = {int(k): float(p) for k, p in zip(gal_idxs, np.atleast_1d(pp))}

    psf5x = select_psf_native(cutout, W / 2.0, H / 2.0)
    s = psf5x.sum()
    if s > 0:
        psf5x = psf5x / s
    psf = PixelizedPSF(psf5x.astype(np.float32), sampling=config.psf_sampling)

    srcs = []
    for k in idxs:
        f = flux_by_id[int(ids[k])]
        if not np.isfinite(f):
            continue
        if not is_gal[k]:
            srcs.append(PointSource(PixPos(float(sx[k]), float(sy[k])), Flux(f)))
        else:
            srcs.append(SersicGalaxy(
                PixPos(float(sx[k]), float(sy[k])), Flux(f),
                GalaxyShape(float(shape_r[k]), float(shape_ab[k]), phi_pix[int(k)]),
                SersicIndex(float(sersic[k]))))

    inverr = np.sqrt(np.maximum(prepared.invvar, 0.0)).astype(np.float32)
    tim = Image(data=prepared.data.astype(np.float32), inverr=inverr, psf=psf,
                wcs=NullWCS(pixscale=SPHEREX_PIXSCALE), sky=ConstantSky(0.0))
    model = np.asarray(Tractor([tim], srcs).getModelImage(0), dtype=np.float64)
    return model, prepared


def plot_fit(cutout, catalog, result, cutout_index, config=None,
             mark_sources=True, chi_range=5.0):
    """Data / model / chi triptych for one cutout of a photometry result.

    Returns the matplotlib figure. ``result`` is the table returned by
    :func:`~spherex_photometry.pipeline.run_photometry` (the fitted fluxes for
    ``cutout_index`` are looked up from it).
    """
    import matplotlib.pyplot as plt

    config = config or PhotometryConfig()
    flux_by_id = fluxes_for_cutout(result, cutout_index)
    model, prepared = render_model_image(cutout, catalog, flux_by_id, config)
    data = prepared.data
    with np.errstate(invalid="ignore"):
        chi = (data - model) * np.sqrt(np.maximum(prepared.invvar, 0.0))

    fin = data[np.isfinite(data)]
    vmax = float(np.percentile(fin, 99.5)) if fin.size else 1.0
    vmin = float(np.percentile(fin, 1.0)) if fin.size else 0.0

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, img, title, kw in (
            (axs[0], data, "data - background", dict(vmin=vmin, vmax=vmax)),
            (axs[1], model, "fitted model", dict(vmin=vmin, vmax=vmax)),
            (axs[2], chi, "chi = (data - model) / sigma",
             dict(vmin=-chi_range, vmax=chi_range, cmap="RdBu_r"))):
        m = ax.imshow(img, origin="lower", interpolation="nearest", **kw)
        ax.set_title(title)
        label = "chi" if img is chi else "mJy / pixel"
        fig.colorbar(m, ax=ax, shrink=0.85, label=label)
    if mark_sources:
        catalog = normalize_catalog(catalog)
        from astropy.coordinates import SkyCoord
        sco = SkyCoord(ra=catalog["ra"], dec=catalog["dec"], unit="deg")
        sx, sy = project_sources(cutout, sco)
        H, W = data.shape
        inside = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
        for ax in axs[:2]:
            ax.scatter(sx[inside], sy[inside], s=90, facecolors="none",
                       edgecolors="w", linewidths=0.8)
    return fig
