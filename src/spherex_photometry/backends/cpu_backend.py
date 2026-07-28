"""CPU backend: forced photometry with the upstream (dstndstn) Tractor.

For users without a GPU or who prefer the classic Tractor. It does one weighted
least-squares solve per cutout (``optimize_forced_photometry``) with source
positions/shapes frozen, so only ``solver="linear"`` is supported (the config
rejects the others). The PSF is the retrieved L2 PSF cube (no instrument
simulator), rendered at 5x oversampling via :class:`OversampledPixelizedPSF` so
the PSF x source-shape convolution stays at oversampled resolution — the
accurate low-resolution flux estimate.

The upstream ``tractor`` package is not on PyPI; install it from source
(``pip install git+https://github.com/dstndstn/tractor``). See docs/cpu_backend.

Notes / approximations (v0.1):
  * Extended (resolved) galaxies are convolved at native resolution inside
    Tractor's Fourier path (only the PSF is oversampled), so resolved-galaxy
    photometry is approximate; most SPHEREx sources are unresolved. Use the JAX
    backend for fully oversampled galaxy convolution.
  * Background is the pre-fit ZODI+model (like the JAX backend's per-tile
    background column, but fit once per cutout rather than per tile).
"""

from __future__ import annotations

import logging

import numpy as np

from ..constants import SPHEREX_PIXSCALE
from ..io.cutouts import sample_map_bilinear_vec
from ..models import sky_pa_to_pixel_pa_batch
from ..prepare import prepare_pixels, project_sources, select_psf_native
from .base import FieldContext
from .cpu_psf import OversampledPixelizedPSF

logger = logging.getLogger(__name__)

# neighbour margin (native px): sources within this of the cutout still enter
# the model (their PSF wings matter) but only in-cutout centres are reported.
MODEL_MARGIN = 5.0


class CpuTractorBackend:
    """Whole-cutout WLS forced photometry on the upstream Tractor (linear only)."""

    name = "cpu-tractor"

    def __init__(self, config):
        self.config = config

    # ---- build (pure CPU) ------------------------------------------------
    def build(self, cutout, ctx: FieldContext):
        import tractor
        from tractor import (
            ConstantSky,
            Flux,
            LinearPhotoCal,
            NullWCS,
            PixPos,
            PointSource,
        )
        from tractor.galaxy import GalaxyShape
        from tractor.sersic import SersicGalaxy, SersicIndex

        cfg = self.config
        prepared = prepare_pixels(cutout, cfg)
        data, invvar = prepared.data, prepared.invvar
        H, W = data.shape

        sx_all, sy_all = project_sources(cutout, ctx.sco_all)

        # PSF: 5x-oversampled native stamp, normalized to unit flux.
        psf5x = select_psf_native(cutout, W / 2.0, H / 2.0)
        s = psf5x.sum()
        if s > 0:
            psf5x = psf5x / s
        psf = OversampledPixelizedPSF(psf5x.astype(np.float32),
                                      sampling=cfg.psf_sampling)

        catalog = ctx.catalog
        in_model = ((sx_all > -MODEL_MARGIN) & (sx_all < W + MODEL_MARGIN)
                    & (sy_all > -MODEL_MARGIN) & (sy_all < H + MODEL_MARGIN)
                    & np.isfinite(sx_all) & np.isfinite(sy_all))
        model_ci = np.where(in_model)[0]

        # Use the normalized shape columns (shape_ab, shape_phi) so the CPU
        # backend's galaxy orientation matches the JAX backend exactly: shape_phi
        # is the engine sky-frame PA (phi = -0.5*atan2(e2, e1); ls_shapes_to_ab_phi),
        # converted to the pixel frame here for tractor's GalaxyShape. Using the
        # +0.5 sign (the stale bench convention) mirror-reflects extended galaxies.
        shape_r = np.asarray(catalog["shape_r"], dtype=np.float64)
        shape_ab = np.asarray(catalog["shape_ab"], dtype=np.float64)
        shape_phi = np.asarray(catalog["shape_phi"], dtype=np.float64)
        sersic = np.asarray(catalog["sersic"], dtype=np.float64)
        is_gal = np.isfinite(shape_r) & (shape_r > 0)
        gal_ci = model_ci[is_gal[model_ci]]
        phi_pix = {}
        if len(gal_ci):
            pp = sky_pa_to_pixel_pa_batch(
                cutout["wcs"],
                np.asarray(catalog["ra"], dtype=np.float64)[gal_ci],
                np.asarray(catalog["dec"], dtype=np.float64)[gal_ci],
                shape_phi[gal_ci])
            phi_pix = {int(ci): float(p) for ci, p in zip(gal_ci, np.atleast_1d(pp))}

        srcs = []
        for ci in model_ci:
            x, y = float(sx_all[ci]), float(sy_all[ci])
            if not is_gal[ci]:
                src = PointSource(PixPos(x, y), Flux(0.1))
            else:
                src = SersicGalaxy(
                    PixPos(x, y), Flux(0.1),
                    GalaxyShape(float(shape_r[ci]), float(shape_ab[ci]),
                                phi_pix[int(ci)]),
                    SersicIndex(float(sersic[ci])))
            src.freezeAllRecursive()
            src.thawParam("brightness")
            srcs.append(src)

        tim = tractor.Image(data=data.astype(np.float32),
                            invvar=invvar.astype(np.float32), psf=psf,
                            wcs=NullWCS(pixscale=SPHEREX_PIXSCALE),
                            photocal=LinearPhotoCal(1.0), sky=ConstantSky(0.0))
        tim.freezeAllRecursive()
        trac = tractor.Tractor([tim], srcs)

        report = in_model & (sx_all >= 0) & (sx_all < W) & (sy_all >= 0) & (sy_all < H)
        report_ci = np.where(report)[0]
        # position of each model source in the source list
        pos_in_list = {int(ci): j for j, ci in enumerate(model_ci)}
        report_pos = np.array([pos_in_list[int(ci)] for ci in report_ci], dtype=np.int64)

        return dict(tractor=trac, cutout=cutout, sx_all=sx_all, sy_all=sy_all,
                    report_ci=report_ci, report_pos=report_pos,
                    n_model=len(model_ci))

    # ---- solve (single WLS) ----------------------------------------------
    def solve(self, inputs):
        trac = inputs["tractor"]
        if inputs["n_model"] == 0:
            return np.zeros(0), np.zeros(0)
        res = trac.optimize_forced_photometry(variance=True, shared_params=False)
        fluxes = np.array([src.getBrightness().getValue()
                           for src in trac.getCatalog()], dtype=np.float64)
        iv = getattr(res, "IV", None)
        if iv is None:
            variances = np.full(len(fluxes), np.nan)
        else:
            iv = np.asarray(iv, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                variances = np.where(iv > 0, 1.0 / iv, np.nan)
        return fluxes, variances

    # ---- extract ---------------------------------------------------------
    def extract(self, inputs, fluxes, variances):
        cutout = inputs["cutout"]
        pos = inputs["report_pos"]
        ci = inputs["report_ci"]
        if len(ci) == 0:
            z = np.zeros(0)
            return (ci, z, z, z, z), cutout.get("cwave_center")
        flux = np.asarray(fluxes)[pos]
        fvar = np.asarray(variances)[pos]
        ferr = np.sqrt(np.maximum(fvar, 0.0))
        sx = inputs["sx_all"][ci]
        sy = inputs["sy_all"][ci]
        lam = sample_map_bilinear_vec(cutout.get("cwave_map"), sx, sy)
        band = sample_map_bilinear_vec(cutout.get("cband_map"), sx, sy)
        return (ci, flux, ferr, lam, band), cutout.get("cwave_center")
