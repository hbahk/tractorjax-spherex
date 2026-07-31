"""CPU backend: forced photometry with the upstream (dstndstn) Tractor.

For users without a GPU or who prefer the classic Tractor. Source positions and
shapes are frozen, so only ``solver="linear"`` is supported (the config rejects
the others). The PSF is the retrieved L2 PSF cube (no instrument simulator),
rendered at 5x oversampling via :class:`OversampledPixelizedPSF` so the PSF x
source-shape convolution stays at oversampled resolution — the accurate
low-resolution flux estimate.

Two solve geometries, selected by ``config.cpu_tiling``:

**Tiled** (default). The cutout is split on the SAME 15 px core + 3 px halo grid
the JAX backend uses (:mod:`spherex_photometry.tiling`) and each tile is a small,
self-contained ``tractor.Tractor`` solved with its own
``optimize_forced_photometry``. Each source is *reported* from the one tile whose
core box contains it, so halo overlaps are never double-counted. This is pure
orchestration: the upstream engine is called unmodified, one tile at a time. Tens
of fluxes per tile converge quickly and are well conditioned, where the
whole-cutout system at full catalog depth is both slow and degenerate.

**Whole-cutout** (``cpu_tiling=False``). One joint solve over every source in the
cutout — the "global geometry" configuration, kept as the independent cross-check
of the tiled result. At full LS depth this system is degenerate; keep
``fit_zmag_max`` at a sensible depth (~21) when using it.

The upstream ``tractor`` package is not on PyPI; install it from source
(``pip install git+https://github.com/dstndstn/tractor``). See docs/cpu_backend.

Notes / approximations (v0.1):
  * Extended (resolved) galaxies are convolved at native resolution inside
    Tractor's Fourier path (only the PSF is oversampled), so resolved-galaxy
    photometry is approximate; most SPHEREx sources are unresolved. Use the JAX
    backend for fully oversampled galaxy convolution.
  * Background is the pre-fit ZODI+model, fit once per cutout. The JAX backend
    additionally fits a free constant per tile; set ``cpu_tile_background=True``
    to fit one here too (upstream's ``sky=True``) and complete the match.
"""

from __future__ import annotations

import logging

import numpy as np

from ..constants import SPHEREX_PIXSCALE
from ..io.cutouts import sample_map_bilinear_vec
from ..models import sky_pa_to_pixel_pa_batch
from ..prepare import prepare_pixels, project_sources
from ..tiling import extract_tile_region, iter_tiles
from .base import FieldContext

logger = logging.getLogger(__name__)

# Whole-cutout neighbour margin (native px): sources within this of the cutout
# still enter the model (their PSF wings matter) but only in-cutout centres are
# reported. The tiled path uses ``tile_halo`` instead, because that is the
# margin the JAX backend uses and same-geometry is the point of tiling.
MODEL_MARGIN = 5.0


class _SourceMaker:
    """Builds frozen-geometry tractor sources for one cutout's catalog rows.

    The per-source geometry (point vs Sérsic, the sky->pixel position-angle
    conversion) is resolved ONCE for the whole cutout; ``__call__`` then
    instantiates a fresh source object at a given pixel position. The tiled path
    needs fresh instances per tile — a source that sits in one tile's halo and
    another's core is fitted twice, with two independent fluxes, so the two
    tiles must not share one mutable ``Flux`` object.
    """

    def __init__(self, cutout, catalog, model_ci):
        from tractor import Flux, PixPos, PointSource
        from tractor.galaxy import GalaxyShape
        from tractor.sersic import SersicGalaxy, SersicIndex

        self._Flux, self._PixPos, self._PointSource = Flux, PixPos, PointSource
        self._GalaxyShape = GalaxyShape
        self._SersicGalaxy, self._SersicIndex = SersicGalaxy, SersicIndex

        # Use the normalized shape columns (shape_ab, shape_phi) so the CPU
        # backend's galaxy orientation matches the JAX backend exactly: shape_phi
        # is the engine sky-frame PA (phi = -0.5*atan2(e2, e1); ls_shapes_to_ab_phi),
        # converted to the pixel frame here for tractor's GalaxyShape. Using the
        # +0.5 sign (the stale bench convention) mirror-reflects extended galaxies.
        self._shape_r = np.asarray(catalog["shape_r"], dtype=np.float64)
        self._shape_ab = np.asarray(catalog["shape_ab"], dtype=np.float64)
        self._sersic = np.asarray(catalog["sersic"], dtype=np.float64)
        shape_phi = np.asarray(catalog["shape_phi"], dtype=np.float64)
        self._is_gal = np.isfinite(self._shape_r) & (self._shape_r > 0)

        model_ci = np.asarray(model_ci, dtype=np.int64)
        gal_ci = model_ci[self._is_gal[model_ci]] if model_ci.size else model_ci
        self._phi_pix: dict[int, float] = {}
        if gal_ci.size:
            pp = sky_pa_to_pixel_pa_batch(
                cutout["wcs"],
                np.asarray(catalog["ra"], dtype=np.float64)[gal_ci],
                np.asarray(catalog["dec"], dtype=np.float64)[gal_ci],
                shape_phi[gal_ci])
            self._phi_pix = {int(ci): float(p)
                             for ci, p in zip(gal_ci, np.atleast_1d(pp))}

    def __call__(self, ci, x, y):
        """A frozen-geometry, flux-thawed tractor source for catalog row ``ci``."""
        ci = int(ci)
        if not self._is_gal[ci]:
            src = self._PointSource(self._PixPos(float(x), float(y)),
                                    self._Flux(0.1))
        else:
            src = self._SersicGalaxy(
                self._PixPos(float(x), float(y)), self._Flux(0.1),
                self._GalaxyShape(float(self._shape_r[ci]),
                                  float(self._shape_ab[ci]),
                                  self._phi_pix[ci]),
                self._SersicIndex(float(self._sersic[ci])))
        src.freezeAllRecursive()
        src.thawParam("brightness")
        return src


def _make_image(data, invvar, psf, *, fit_sky=False):
    """One ``tractor.Image`` over already-prepared (mJy/pixel) pixels.

    Fully frozen unless ``fit_sky``, which thaws the constant sky so
    ``optimize_forced_photometry(sky=True)`` fits it jointly with the fluxes.
    Upstream asserts that the sky is then the image's ONLY thawed parameter, so
    the freeze-everything-then-thaw-one order matters.
    """
    import tractor
    from tractor import ConstantSky, LinearPhotoCal, NullWCS

    tim = tractor.Image(data=np.ascontiguousarray(data, dtype=np.float32),
                        invvar=np.ascontiguousarray(invvar, dtype=np.float32),
                        psf=psf, wcs=NullWCS(pixscale=SPHEREX_PIXSCALE),
                        photocal=LinearPhotoCal(1.0), sky=ConstantSky(0.0))
    tim.freezeAllRecursive()
    if fit_sky:
        tim.thawParam("sky")
    return tim


def _forced_solve(trac, *, fit_sky=False):
    """One upstream WLS forced-photometry solve -> ``(fluxes, variances)``.

    Both arrays follow ``trac.getCatalog()`` order. Each source contributes
    exactly one thawed parameter (geometry frozen, brightness thawed), so
    ``res.IV`` — which is ``catalog.numberOfParams()`` long — lines up one-to-one
    with them. ``skyvariance`` is deliberately left off: it is the only thing
    that would prepend sky entries to ``IV`` and shift every source's variance
    by one (and it is broken upstream anyway). A shorter or absent ``IV`` becomes
    NaN variances rather than a misaligned gather.
    """
    n = len(trac.getCatalog())
    if n == 0:
        return np.zeros(0), np.zeros(0)
    res = trac.optimize_forced_photometry(variance=True, shared_params=False,
                                          sky=fit_sky)
    fluxes = np.array([src.getBrightness().getValue()
                       for src in trac.getCatalog()], dtype=np.float64)
    iv = getattr(res, "IV", None)
    variances = np.full(n, np.nan)
    if iv is not None:
        iv = np.asarray(iv, dtype=np.float64).ravel()
        if iv.size >= n:
            with np.errstate(divide="ignore", invalid="ignore"):
                variances = np.where(iv[:n] > 0, 1.0 / iv[:n], np.nan)
        else:
            logger.warning("optimize_forced_photometry returned %d inverse "
                           "variances for %d sources; reporting NaN errors",
                           iv.size, n)
    return fluxes, variances


class CpuTractorBackend:
    """Forced photometry on the upstream Tractor (linear only), tiled by default."""

    name = "cpu-tractor"

    def __init__(self, config):
        self.config = config

    # ---- build (pure CPU) ------------------------------------------------
    def build(self, cutout, ctx: FieldContext):
        if getattr(self.config, "cpu_tiling", True):
            return self._build_tiled(cutout, ctx)
        return self._build_whole(cutout, ctx)

    def _prepared(self, cutout, ctx):
        """Pixels, source positions and the PSF-fix flags, shared by both paths."""
        cfg = self.config
        prepared = prepare_pixels(cutout, cfg)
        sx_all, sy_all = project_sources(cutout, ctx.sco_all)
        return prepared.data, prepared.invvar, sx_all, sy_all

    # -- whole-cutout ("global geometry") ----------------------------------
    def _build_whole(self, cutout, ctx: FieldContext):
        import tractor

        from .. import prepare as _prepare
        from .zone_psf import build_cpu_psf

        data, invvar, sx_all, sy_all = self._prepared(cutout, ctx)
        H, W = data.shape

        # PSF per the config's PSF-fix flags: zone-blended and/or
        # core-registered when asked, the plain centre-zone stamp otherwise.
        psf = build_cpu_psf(cutout, self.config, prepare=_prepare)

        in_model = ((sx_all > -MODEL_MARGIN) & (sx_all < W + MODEL_MARGIN)
                    & (sy_all > -MODEL_MARGIN) & (sy_all < H + MODEL_MARGIN)
                    & np.isfinite(sx_all) & np.isfinite(sy_all))
        model_ci = np.where(in_model)[0]

        make_source = _SourceMaker(cutout, ctx.catalog, model_ci)
        srcs = [make_source(ci, sx_all[ci], sy_all[ci]) for ci in model_ci]
        trac = tractor.Tractor([_make_image(data, invvar, psf)], srcs)

        report = in_model & (sx_all >= 0) & (sx_all < W) & (sy_all >= 0) & (sy_all < H)
        report_ci = np.where(report)[0]
        # position of each model source in the source list
        pos_in_list = {int(ci): j for j, ci in enumerate(model_ci)}
        report_pos = np.array([pos_in_list[int(ci)] for ci in report_ci],
                              dtype=np.int64)

        return dict(tiles=[trac], cutout=cutout, sx_all=sx_all, sy_all=sy_all,
                    report_ci=report_ci, report_pos=report_pos,
                    n_model=len(model_ci), fit_sky=False)

    # -- tiled (the JAX backend's geometry) --------------------------------
    def _build_tiled(self, cutout, ctx: FieldContext):
        import tractor

        from .. import prepare as _prepare
        from .zone_psf import build_cpu_psf_selector

        cfg = self.config
        data, invvar, sx_all, sy_all = self._prepared(cutout, ctx)
        H, W = data.shape
        tile_size, halo = cfg.tile_size, cfg.tile_halo
        fit_sky = bool(getattr(cfg, "cpu_tile_background", False))

        # One constant kernel per tile, blended at the tile's core centre —
        # exactly what build_cutout_tiles hands the JAX engine.
        psf_select = build_cpu_psf_selector(cutout, cfg, prepare=_prepare)

        # Cutout-level pre-filter, matching build_cutout_tiles: the halo boxes of
        # the outermost tiles can reach past W + halo when W is not a multiple of
        # tile_size, and sources out there must not enter any model.
        in_model = ((sx_all > -halo) & (sx_all < W + halo)
                    & (sy_all > -halo) & (sy_all < H + halo)
                    & np.isfinite(sx_all) & np.isfinite(sy_all))
        model_ci = np.where(in_model)[0]
        mx, my = sx_all[model_ci], sy_all[model_ci]

        make_source = _SourceMaker(cutout, ctx.catalog, model_ci)

        tiles: list = []
        report_ci_parts: list = []
        report_pos_parts: list = []
        slot_base = 0
        n_model = 0
        for meta in iter_tiles(H, W, tile_size, halo):
            xs, ys = meta["x_start"], meta["y_start"]
            xe, ye = meta["x_end"], meta["y_end"]
            in_box = (mx >= xs) & (mx < xe) & (my >= ys) & (my < ye)
            idxs = model_ci[in_box]
            if idxs.size == 0:
                continue          # empty tile: no core sources either
            tx, ty = sx_all[idxs] - xs, sy_all[idxs] - ys
            srcs = [make_source(ci, x, y) for ci, x, y in zip(idxs, tx, ty)]
            cx = 0.5 * (meta["core_x0"] + meta["core_x1"])
            cy = 0.5 * (meta["core_y0"] + meta["core_y1"])
            tim = _make_image(
                extract_tile_region(data, xs, ys, xe, ye, fill=0.0),
                extract_tile_region(invvar, xs, ys, xe, ye, fill=0.0),
                psf_select(cx, cy), fit_sky=fit_sky)
            tiles.append(tractor.Tractor([tim], srcs))

            # Read back ONLY the sources whose centres lie in this tile's CORE.
            # Cores tile [0,W)x[0,H) exactly, so every in-cutout source is
            # claimed by exactly one tile: no double counts, no drops.
            core = ((sx_all[idxs] >= meta["core_x0"])
                    & (sx_all[idxs] < meta["core_x1"])
                    & (sy_all[idxs] >= meta["core_y0"])
                    & (sy_all[idxs] < meta["core_y1"]))
            report_ci_parts.append(idxs[core])
            report_pos_parts.append(slot_base + np.where(core)[0])
            slot_base += idxs.size
            n_model += idxs.size

        if report_ci_parts:
            report_ci = np.concatenate(report_ci_parts)
            report_pos = np.concatenate(report_pos_parts)
            order = np.argsort(report_ci, kind="stable")   # ascending ci, as
            report_ci = report_ci[order]                   # the whole-cutout
            report_pos = report_pos[order]                 # path reports
        else:
            report_ci = np.zeros(0, dtype=np.int64)
            report_pos = np.zeros(0, dtype=np.int64)

        return dict(tiles=tiles, cutout=cutout, sx_all=sx_all, sy_all=sy_all,
                    report_ci=report_ci, report_pos=report_pos,
                    n_model=n_model, fit_sky=fit_sky)

    # ---- solve (one WLS per tile; one tile = the whole cutout when untiled) --
    def solve(self, inputs):
        if inputs["n_model"] == 0:
            return np.zeros(0), np.zeros(0)
        fit_sky = inputs.get("fit_sky", False)
        fluxes, variances = [], []
        for trac in inputs["tiles"]:
            f, v = _forced_solve(trac, fit_sky=fit_sky)
            fluxes.append(f)
            variances.append(v)
        return np.concatenate(fluxes), np.concatenate(variances)

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
