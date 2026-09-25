"""GPU/JAX backend: tiled batched forced photometry on the tractor-jax engine.

Lifted from the production driver, de-globalized so every knob comes from a
:class:`~tractorjax_spherex.config.PhotometryConfig`. Imports JAX (via the
tractor-jax engine) at module import, so this module is imported lazily by
:func:`tractorjax_spherex.backends.get_backend` after the device is configured.

The engine renders every source on a 5x-oversampled grid (``psf_sampling=0.2``,
``fixed_max_factor=5``) and sum-bins to native pixels, so the PSF x source-shape
convolution is done at oversampled resolution — the accurate low-resolution flux
estimate. Cutouts are split into ``tile_size`` cores with a halo and all tiles
of a cutout are solved in one ``vmap``.

The tile grid itself lives in :mod:`tractorjax_spherex.tiling` and is shared with
the ``cpu-tractor`` backend (re-exported here for callers that import it from
this module), so a CPU-vs-GPU comparison is a comparison of engines rather than
of geometries.
"""

from __future__ import annotations

import numpy as np

# tractor-jax engine (imports jax).
from tractor_jax.jax import batching as tjb
from tractor_jax.jax.pipeline import prefetch_pipeline  # noqa: F401  (re-exported)

#: Oldest engine this layer runs on: 0.3.0 added the static ``pixel_integration``
#: solver option that R7 effective-PSF bundles need.
TRACTOR_JAX_MIN = (0, 3, 0)


def _check_engine_version() -> None:
    import tractor_jax
    ver = getattr(tractor_jax, "__version__", "0")
    parts = []
    for tok in str(ver).split("."):
        digits = "".join(ch for ch in tok if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    if not parts:
        # no parseable version (e.g. the engine mocked out by autodoc): do not judge
        return
    if tuple(parts[:3]) < TRACTOR_JAX_MIN:
        raise ImportError(
            f"tractorjax-spherex needs tractor-jax >= {'.'.join(map(str, TRACTOR_JAX_MIN))} "
            f"(found {ver}); install the release it is developed against:\n"
            "    pip install git+https://github.com/hbahk/tractor-jax@v0.3.0")


_check_engine_version()

from ..config import CapExceededError, ConfigError
from ..io.cutouts import sample_map_bilinear_vec
from ..prepare import (
    core_shift_applies,
    pixel_integration_for,
    prepare_pixels,
    project_sources,
    psf_stamp_5x,
    zone_lookup_vectorized,
    zone_psf_basis,
    zone_psf_selector,
)
from ..psf_cache import PSFCache
from ..tiling import (
    cd_inv_from_wcs,
    extract_tile_region,
    iter_tiles,
    shift_wcs,
)
from .base import FieldContext

#: Build a sliced :class:`~astropy.wcs.WCS` for every tile.  Only the CD matrix
#: is ever read from it, and :func:`~tractorjax_spherex.tiling.shift_wcs` leaves
#: that bit-identical, so the default hands the batch builder one cutout-level
#: ``cd_inv`` instead and skips ``WCS.slice`` -- a deep copy that cost ~40 ms
#: per cutout in the driver's host profile, and scales with the tile count
#: (a 2040x2040 frame at tile 15 has 18,496 tiles, not the ~44 of a 100x100
#: cutout).  Set True to restore the per-tile WCS when bisecting a difference.
PER_TILE_WCS = False

#: Do the PSF-zone lookup for every tile in ONE vectorised call
#: (:func:`~tractorjax_spherex.prepare.zone_planes_and_weights`) instead of two
#: Python scans of the zone table per tile -- ~15 ms per cutout in the driver's
#: host profile, and it grows with the tile count.  Bit-identical by
#: construction and by test; set False to restore the scalar helpers.
VECTOR_ZONES = True

#: Share PSF kernels, core-shift tables and the engine's Fourier transforms
#: ACROSS cutouts.  Every cutout of one detector ships a byte-identical PSF
#: cube, so downsampling and transforming it per cutout is repeat work (~20 ms
#: per cutout in the driver's host profile, and it does not shrink with the
#: cutout).  The cache lives on the backend instance, i.e. for one run.  Set
#: False to rebuild per cutout.  See :mod:`tractorjax_spherex.psf_cache` for why
#: the kernels and their transforms must be cached and cleared together.
PSF_CACHE_ACROSS_CUTOUTS = True


# --------------------------------------------------------------------------- #
# Batch build
# --------------------------------------------------------------------------- #
def extract_tiled_batches(tile_records, catalog_full, sx_all, sy_all,
                          psf_sampling=0.2, fixed_max_factor=5.0,
                          fit_background=True, profile_lookup_fn=None,
                          max_ps_cap=None, max_gal_cap=None,
                          max_mog_k_cap=None, pad_bucket=None,
                          psf_fft_cache=None):
    """Build vmap-ready padded batches for a cutout's tiles (engine call)."""
    if len(tile_records) == 0:
        raise ValueError("extract_tiled_batches: tile_records is empty")

    # Prefer the cutout-level cd_inv the tile builder computed once (see
    # PER_TILE_WCS); fall back to deriving it from a tile WCS so tile_records
    # assembled by an outside caller still work.
    cd_inv = tile_records[0].get("cd_inv")
    if cd_inv is None:
        cd_inv = cd_inv_from_wcs(tile_records[0]["wcs"])

    views = []
    for t in tile_records:
        v = {
            "data": t["data"], "invvar": t["invvar"], "psf": t["psf"],
            "src_indices": t["src_indices"],
            "origin": (t["tile_meta"]["x_start"], t["tile_meta"]["y_start"]),
        }
        # Zone-interp / core-shift pass-through. Dropping these here was the
        # bug that made psf_zone_interp a silent no-op on this backend: the
        # tile builder attached the basis but the engine never saw it.
        for key in ("psf_basis", "psf_weights", "psf_basis_shifts"):
            if t.get(key) is not None:
                v[key] = t[key]
        views.append(v)

    bundle = tjb.build_padded_batches(
        views, catalog_full, sx_all, sy_all,
        psf_sampling=psf_sampling, fixed_max_factor=fixed_max_factor,
        fit_background=fit_background, profile_lookup_fn=profile_lookup_fn,
        cd_inv=cd_inv, max_ps_cap=max_ps_cap, max_gal_cap=max_gal_cap,
        max_mog_k_cap=max_mog_k_cap, pad_bucket=pad_bucket,
        psf_fft_cache=psf_fft_cache)
    return (bundle.images_data, bundle.batches, bundle.initial_fluxes,
            bundle.meta["src_slot"])


def build_cutout_tiles(cutout, *, sx_all, sy_all, tile_size, halo,
                       data_scaled, invvar_scaled, psf_native=None,
                       psf_select=None, psf_basis=None, psf_weights=None,
                       psf_basis_shifts=None, psf_cache=None):
    """Construct tile records (core + halo boxes) for one cutout.

    ``psf_select(x, y) -> stamp`` (from
    :func:`~tractorjax_spherex.prepare.zone_psf_selector`) gives each tile the
    PSF of the zone containing its own core centre — the SPHEREx PSF varies
    across the focal plane and the zone pitch (~185 detector px) is smaller
    than a typical cutout, so one kernel per cutout mis-renders the tiles that
    fall in a neighbouring zone. ``psf_native`` is the legacy single-kernel
    form, kept for callers that already resolved the PSF themselves.
    """
    if psf_select is None:
        if psf_native is None:
            raise ValueError("build_cutout_tiles needs psf_select or psf_native")
        psf_select = lambda x, y: psf_native
    H, W = data_scaled.shape
    inside = ((sx_all > -halo) & (sx_all < W + halo)
              & (sy_all > -halo) & (sy_all < H + halo)
              & np.isfinite(sx_all) & np.isfinite(sy_all))
    cutout_src_indices = np.where(inside)[0]
    cutout_sx = sx_all[cutout_src_indices]
    cutout_sy = sy_all[cutout_src_indices]

    # One per cutout, not one per tile: shift_wcs only moves the reference
    # pixel, so every tile shares this matrix (see PER_TILE_WCS).
    cd_inv = cd_inv_from_wcs(cutout["wcs"])

    metas = list(iter_tiles(H, W, tile_size, halo))

    # One vectorised zone lookup for every tile core centre instead of two
    # Python scans of the zone table per tile (see VECTOR_ZONES). The plane
    # selection is skipped when the caller supplied a fixed psf_native.
    v_planes = v_weights = None
    if VECTOR_ZONES and cutout.get("psf_zones") is not None:
        cxs = np.array([0.5 * (m["core_x0"] + m["core_x1"]) for m in metas])
        cys = np.array([0.5 * (m["core_y0"] + m["core_y1"]) for m in metas])
        v_planes, _v_rows, v_weights = zone_lookup_vectorized(cutout, cxs, cys)
        if psf_native is not None:
            v_planes = None

    # Kernels are shared objects: within the cutout always, and across cutouts
    # when a PSFCache is given (the engine's transform cache keys on identity).
    _local: dict[int, np.ndarray] = {}
    _sig = None
    if psf_cache is not None and v_planes is not None:
        from ..psf_cache import cube_signature
        _sig = cube_signature(cutout)

    def _stamp(plane):
        def _build():
            return psf_stamp_5x(cutout, plane)
        if _sig is not None:
            return psf_cache.stamp(_sig, plane, _build)
        s = _local.get(plane)
        if s is None:
            s = _build()
            _local[plane] = s
        return s

    tile_records = []
    for ti, meta in enumerate(metas):
        xs, ys = meta["x_start"], meta["y_start"]
        xe, ye = meta["x_end"], meta["y_end"]
        in_box = ((cutout_sx >= xs) & (cutout_sx < xe)
                  & (cutout_sy >= ys) & (cutout_sy < ye))
        idxs = cutout_src_indices[in_box].tolist()
        cx = 0.5 * (meta["core_x0"] + meta["core_x1"])
        cy = 0.5 * (meta["core_y0"] + meta["core_y1"])
        rec = {
            "data": extract_tile_region(data_scaled, xs, ys, xe, ye, fill=0.0),
            "invvar": extract_tile_region(invvar_scaled, xs, ys, xe, ye, fill=0.0),
            "psf": (_stamp(int(v_planes[ti])) if v_planes is not None
                    else psf_select(cx, cy)),
            "wcs": shift_wcs(cutout["wcs"], xs, ys) if PER_TILE_WCS else None,
            "cd_inv": cd_inv,
            "src_indices": idxs,
            "tile_meta": meta,
        }
        # Single-zone cutouts skip the blend (its weights are one-hot, so it
        # would only cost time) -- but NOT when core shifts are present: the
        # shifts ride on this basis, so skipping it made psf_core_shift a
        # silent no-op on any single-zone cutout, which is what a retrieval
        # without a zone margin gives for anything under the ~185 px zone
        # pitch. The cpu-tractor backend shifts the stamp directly and always
        # applied it, so the two backends disagreed by construction there.
        # zone_bilinear_weights degenerates to one-hot on one zone, so the K=1
        # basis path is exact.
        if psf_basis is not None and (len(psf_basis) > 1
                                      or psf_basis_shifts is not None):
            rec["psf_basis"] = psf_basis
            rec["psf_weights"] = (v_weights[ti] if v_weights is not None
                                  else psf_weights(cx, cy))
            if psf_basis_shifts is not None:
                # the SAME (K, 2) array for every tile: the engine memoizes
                # the native->high-res conversion on the table's identity
                rec["psf_basis_shifts"] = psf_basis_shifts
        tile_records.append(rec)
    return tile_records


def _check_caps(tile_records, catalog, max_ps_cap, max_gal_cap,
                cutout_index=None):
    """Raise :class:`CapExceededError` before the engine's bare ValueError.

    The engine already refuses to build an over-wide batch, but it reports a
    string the caller has to parse. Checking here means the failure names the
    width that was needed and how to fix it, and it costs one pass over the
    already-computed tile index lists.
    """
    if max_ps_cap is None and max_gal_cap is None:
        return
    shape_r = np.asarray(catalog["shape_r"], dtype=np.float64)
    need_ps = need_gal = 0
    for t in tile_records:
        src = np.asarray(t["src_indices"], dtype=np.intp)
        if src.size == 0:
            continue
        isgal = shape_r[src] > 0
        need_ps = max(need_ps, int((~isgal).sum()))
        need_gal = max(need_gal, int(isgal.sum()))
    if max_ps_cap is not None and need_ps > max_ps_cap:
        raise CapExceededError("ps", need_ps, max_ps_cap, cutout_index)
    if max_gal_cap is not None and need_gal > max_gal_cap:
        raise CapExceededError("gal", need_gal, max_gal_cap, cutout_index)


def build_extract_index(tile_records, src_slot, sx_all, sy_all, W, H):
    """Precompute the (tile, slot) gather index for record extraction.

    Each in-cutout source is read from the tile whose CORE box contains it, so
    halo overlaps are not double-counted. Returns arrays in ascending-ci order.
    """
    inside = ((sx_all >= 0) & (sx_all < W) & (sy_all >= 0) & (sy_all < H)
              & np.isfinite(sx_all) & np.isfinite(sy_all))
    cis = np.where(inside)[0]
    sx, sy = sx_all[cis], sy_all[cis]

    metas = [t["tile_meta"] for t in tile_records]
    cx0 = np.array([m["core_x0"] for m in metas])
    cy0 = np.array([m["core_y0"] for m in metas])
    x_edges = np.unique(cx0)
    y_edges = np.unique(cy0)
    lut = np.full((len(y_edges), len(x_edges)), -1, dtype=np.int64)
    ixs = np.searchsorted(x_edges, cx0)
    iys = np.searchsorted(y_edges, cy0)
    lut[iys, ixs] = np.arange(len(metas))
    ix = np.searchsorted(x_edges, sx, side="right") - 1
    iy = np.searchsorted(y_edges, sy, side="right") - 1
    ti = lut[iy, ix]

    keep_ci, keep_ti, keep_slot = [], [], []
    for k in range(len(cis)):
        slot = src_slot[int(ti[k])].get(int(cis[k]))
        if slot is not None:
            keep_ci.append(int(cis[k]))
            keep_ti.append(int(ti[k]))
            keep_slot.append(int(slot))
    keep_ci = np.asarray(keep_ci, dtype=np.int64)
    return dict(ci=keep_ci,
                ti=np.asarray(keep_ti, dtype=np.int64),
                slot=np.asarray(keep_slot, dtype=np.int64),
                sx=sx_all[keep_ci], sy=sy_all[keep_ci])


def engine_has_diagnostics() -> bool:
    """Whether the installed engine's solvers can return fit diagnostics."""
    import inspect
    return "return_diagnostics" in inspect.signature(tjb.make_batched_solver).parameters


def make_tiled_solver(spec, batches_in_axes, penalty_weights=None,
                      prior_arrays=None, pixel_integration="window",
                      return_diagnostics=False):
    """Build the vmapped per-tile solver for ``spec`` (thin engine wrapper).

    ``pixel_integration`` is the engine's static rendering option: ``"window"``
    for the QR2 optical PSF, ``"point"`` for the R7 effective PSF (see
    :func:`tractorjax_spherex.prepare.pixel_integration_for`).
    ``return_diagnostics`` makes the solver also return the per-slot fit
    diagnostics ``{"chi2", "mask_frac"}`` (not for lasso).
    """
    kind = spec["kind"]
    static = dict(pixel_integration=pixel_integration)
    if return_diagnostics:
        if kind == "lasso":
            raise ConfigError("visit_diagnostics is not available with solver='lasso'")
        static["return_diagnostics"] = True
    if kind == "linear":
        return tjb.make_batched_solver(
            "linear", in_axes=batches_in_axes, rcond=spec.get("rcond", 1e-12), **static)
    if kind == "eigfloor":
        return tjb.make_batched_solver(
            "eigfloor", in_axes=batches_in_axes, floor=spec.get("floor", 1e-2), **static)
    if kind == "eigfloor_prior":
        jfn = tjb.make_batched_solver(
            "eigfloor_prior", in_axes=batches_in_axes,
            floor=spec.get("floor", 1e-2), **static)
        lam0, fp0 = prior_arrays if prior_arrays is not None else (None, None)
        return lambda init, imgd, bat, lam=lam0, fp=fp0: jfn(init, imgd, bat, lam, fp)
    if kind == "lasso":
        jfn = tjb.make_batched_solver(
            "lasso", in_axes=batches_in_axes,
            alpha=spec.get("alpha", "auto"), penalty_mode="snr",
            nonneg=True, debias=True,
            debias_signfree=spec.get("debias_signfree", "protected"),
            n_iter=spec.get("n_iter", 1000), **static)
        return lambda init, imgd, bat, pw=penalty_weights: jfn(init, imgd, bat, pw)
    raise ValueError(f"unknown solver kind {kind!r}")


# --------------------------------------------------------------------------- #
# Backend
# --------------------------------------------------------------------------- #
class JaxBackend:
    """Tiled batched forced photometry on the tractor-jax GPU/CPU engine."""

    name = "jax"

    def __init__(self, config):
        self.config = config
        if getattr(config, "visit_diagnostics", False) and not engine_has_diagnostics():
            raise ImportError(
                "visit_diagnostics needs a tractor-jax whose make_batched_solver takes "
                "return_diagnostics (the feat/solve-diagnostics engine or later)")
        self._solve_fn_cache: dict = {}
        # Lives for the whole run: the PSF cube is byte-identical across every
        # cutout of one detector, so its kernels and their engine transforms are
        # built once instead of once per cutout (see PSF_CACHE_ACROSS_CUTOUTS).
        self._psf_cache = PSFCache() if PSF_CACHE_ACROSS_CUTOUTS else None

    # ---- build (CPU stage; prefetch-safe) --------------------------------
    def build(self, cutout, ctx: FieldContext):
        cfg = self.config
        if self._psf_cache is not None:
            # the one safe point to evict: nothing of this cutout is cached yet
            self._psf_cache.begin_cutout()
        prepared = prepare_pixels(cutout, cfg)
        data, invvar = prepared.data, prepared.invvar
        H, W = data.shape

        sx_all, sy_all = project_sources(cutout, ctx.sco_all)

        basis = weights = basis_shifts = None
        if getattr(cfg, "psf_zone_interp", True):
            basis, weights = zone_psf_basis(cutout, cache=self._psf_cache)
        if core_shift_applies(cfg, cutout):
            if basis is None:
                raise ConfigError(
                    "psf_core_shift on the JAX backend requires "
                    "psf_zone_interp=True (the shifts ride on the zone "
                    "basis); the cpu-tractor backend supports it standalone.")
            from ..calib import DOWNSAMPLE_GRID_SHIFT_NATIVE, psf_core_shift
            det = int(cutout.detector)
            zone_ids = np.asarray(cutout.psf_zones["zone_id"], dtype=int)

            def _shift_table():
                rows = []
                for z in zone_ids:
                    cs = psf_core_shift(det, int(z))
                    if cs.source != "zone":
                        raise ValueError(
                            f"psf_core_shift(det={det}, zone={int(z)}) fell back "
                            f"to {cs.source!r}; coverage is 726/726, so a "
                            "fallback means the detector or zone_id is wrong")
                    rows.append((cs.dy_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                                 cs.dx_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE))
                return np.asarray(rows, dtype=np.float64)

            # The engine memoizes the native -> high-res conversion on this
            # table's identity, so share ONE object per (detector, zone ids).
            basis_shifts = (self._psf_cache.zone_shifts(det, zone_ids,
                                                        _shift_table)
                            if self._psf_cache is not None else _shift_table())
        tile_records = build_cutout_tiles(
            cutout, sx_all=sx_all, sy_all=sy_all,
            tile_size=cfg.tile_size, halo=cfg.tile_halo,
            data_scaled=data, invvar_scaled=invvar,
            psf_select=zone_psf_selector(cutout),
            psf_basis=basis, psf_weights=weights,
            psf_basis_shifts=basis_shifts, psf_cache=self._psf_cache)

        max_ps, max_gal, max_mog_k = cfg.resolved_caps(ctx.occupancy)
        _check_caps(tile_records, ctx.catalog, max_ps, max_gal,
                    cutout_index=getattr(ctx, "cutout_index", None))
        images_data, batches, initial_fluxes, src_slot = extract_tiled_batches(
            tile_records, ctx.catalog, sx_all, sy_all,
            psf_sampling=cfg.psf_sampling, fixed_max_factor=cfg.fixed_max_factor,
            fit_background=True, profile_lookup_fn=ctx.profile_lookup_fn,
            max_ps_cap=max_ps, max_gal_cap=max_gal, max_mog_k_cap=max_mog_k,
            pad_bucket=cfg.pad_bucket or None,
            psf_fft_cache=(self._psf_cache.fft
                           if self._psf_cache is not None else None))

        batches_in_axes = tjb.batches_in_axes(batches)
        extract_index = build_extract_index(tile_records, src_slot,
                                            sx_all, sy_all, W, H)

        prior_flux = prior_sigma = None
        n_prior_free = 0
        if ctx.prior_ctx is not None:
            lam_all = sample_map_bilinear_vec(
                cutout.get("cwave_map"),
                np.clip(sx_all, 0, W - 1), np.clip(sy_all, 0, H - 1))
            f_pred_ujy, _nb = ctx.prior_ctx["predict_fn"](
                ctx.prior_ctx["band_flux_ujy"], lam_all)
            prior_flux = f_pred_ujy * 1e-3  # uJy -> mJy
            prior_sigma = np.maximum(ctx.prior_ctx["sigma_frac"] * prior_flux,
                                     ctx.prior_ctx["sigma_min_mjy"])
            inside = (sx_all >= 0) & (sx_all < W) & (sy_all >= 0) & (sy_all < H)
            n_prior_free = int(np.sum(inside & ~np.isfinite(prior_flux)))

        return dict(images_data=images_data, batches=batches,
                    initial_fluxes=initial_fluxes, src_slot=src_slot,
                    batches_in_axes=batches_in_axes, tile_records=tile_records,
                    sx_all=sx_all, sy_all=sy_all, W=W, H=H, cutout=cutout,
                    extract_index=extract_index, protect_ci=ctx.protect_ci,
                    prior_flux=prior_flux, prior_sigma=prior_sigma,
                    n_prior_free=n_prior_free,
                    # static engine option: one compiled solver per value, so a
                    # run mixing QR2 (window) and R7 (point) cutouts compiles two
                    pixel_integration=pixel_integration_for(cutout))

    # ---- solve (GPU stage) -----------------------------------------------
    def solve(self, inputs):
        """``(fluxes, variances)`` per (tile, slot); with ``visit_diagnostics``
        the fit diagnostics are left in ``inputs["diagnostics"]`` for
        :meth:`extract_diagnostics`."""
        diag = bool(getattr(self.config, "visit_diagnostics", False))
        out = self._solve(inputs, diag)
        if diag:
            fluxes, variances, d = out
            inputs["diagnostics"] = {k: np.asarray(v) for k, v in d.items()}
            return fluxes, variances
        return out

    def _solve(self, inputs, diag):
        spec = self.config.solver_spec()
        protect_ci = inputs.get("protect_ci")
        batches = inputs["batches"]
        batches_in_axes = inputs["batches_in_axes"]
        initial_fluxes = inputs["initial_fluxes"]
        tile_chunk = self.config.tile_chunk

        solve_fn = None
        lam_d = f_pr = pw = None
        pix = inputs.get("pixel_integration", "window")
        if spec["kind"] == "lasso":
            n_tiles = int(np.asarray(initial_fluxes).shape[0])
            n_flux = int(np.asarray(initial_fluxes).shape[1])
            pw = tjb.penalty_weights_from_slots(
                inputs["src_slot"], n_tiles, n_flux, protect_ci or set())
            solve_fn = make_tiled_solver(spec, batches_in_axes, penalty_weights=pw,
                                         pixel_integration=pix, return_diagnostics=diag)
        elif spec["kind"] == "eigfloor_prior":
            if inputs.get("prior_flux") is None:
                raise ValueError("eigfloor_prior requires prior_ctx (SED priors)")
            n_tiles = int(np.asarray(initial_fluxes).shape[0])
            n_flux = int(np.asarray(initial_fluxes).shape[1])
            lam_d, f_pr = tjb.prior_arrays_from_slots(
                inputs["src_slot"], n_tiles, n_flux,
                inputs["prior_flux"], inputs["prior_sigma"],
                protected=sorted(protect_ci) if protect_ci else [])
            solve_fn = make_tiled_solver(spec, batches_in_axes,
                                         prior_arrays=(lam_d, f_pr),
                                         pixel_integration=pix, return_diagnostics=diag)
        else:
            struct_key = (tuple(sorted(batches.keys())), pix, diag)
            solve_fn = self._solve_fn_cache.get(struct_key)
            if solve_fn is None:
                solve_fn = make_tiled_solver(spec, batches_in_axes,
                                             pixel_integration=pix, return_diagnostics=diag)
                self._solve_fn_cache[struct_key] = solve_fn

        n_tiles_total = int(np.asarray(initial_fluxes).shape[0])
        if not tile_chunk or n_tiles_total <= tile_chunk:
            out = solve_fn(initial_fluxes, inputs["images_data"], batches)
            return tuple(np.asarray(o) if not isinstance(o, dict) else o for o in out)

        return self._solve_chunked(inputs, spec, solve_fn, batches,
                                   batches_in_axes, initial_fluxes,
                                   n_tiles_total, tile_chunk, pw, lam_d, f_pr, diag)

    @staticmethod
    def _solve_chunked(inputs, spec, solve_fn, batches, batches_in_axes,
                       initial_fluxes, n_tiles_total, tile_chunk, pw, lam_d, f_pr,
                       diag=False):
        def _cut(x, start, end, pad):
            if isinstance(x, dict):
                return {k: _cut(v, start, end, pad) for k, v in x.items()}
            x = np.asarray(x)
            sl = x[start:end]
            if pad:
                sl = np.concatenate([sl, np.repeat(sl[-1:], pad, axis=0)], axis=0)
            return sl

        def _cut_tree(tree, axes_tree, start, end, pad):
            if isinstance(axes_tree, dict):
                return {k: _cut_tree(tree[k], axes_tree[k], start, end, pad)
                        for k in axes_tree}
            if axes_tree == 0:
                return _cut(tree, start, end, pad)
            return tree

        outs_f, outs_v, outs_d = [], [], []
        for start in range(0, n_tiles_total, tile_chunk):
            end = min(start + tile_chunk, n_tiles_total)
            pad = tile_chunk - (end - start)
            init_c = _cut(initial_fluxes, start, end, pad)
            imgd_c = {k: _cut(v, start, end, pad)
                      for k, v in inputs["images_data"].items()}
            bat_c = _cut_tree(batches, batches_in_axes, start, end, pad)
            if spec["kind"] == "lasso":
                out = solve_fn(init_c, imgd_c, bat_c, _cut(pw, start, end, pad))
            elif spec["kind"] == "eigfloor_prior":
                out = solve_fn(init_c, imgd_c, bat_c,
                               _cut(lam_d, start, end, pad),
                               _cut(f_pr, start, end, pad))
            else:
                out = solve_fn(init_c, imgd_c, bat_c)
            outs_f.append(np.asarray(out[0])[:end - start])
            outs_v.append(np.asarray(out[1])[:end - start])
            if diag:
                outs_d.append({k: np.asarray(v)[:end - start] for k, v in out[2].items()})
        f, v = np.concatenate(outs_f, axis=0), np.concatenate(outs_v, axis=0)
        if diag:
            return f, v, {k: np.concatenate([d[k] for d in outs_d], axis=0) for k in outs_d[0]}
        return f, v

    # ---- extract ---------------------------------------------------------
    def extract_diagnostics(self, inputs):
        """Per-row fit diagnostics, in the order of :meth:`extract`'s rows."""
        ei = inputs["extract_index"]
        d = inputs["diagnostics"]
        return {"fit_chi2": np.asarray(d["chi2"], dtype=np.float64)[ei["ti"], ei["slot"]],
                "mask_frac": np.asarray(d["mask_frac"], dtype=np.float64)[ei["ti"], ei["slot"]]}

    def extract(self, inputs, fluxes_np, var_np):
        cutout = inputs["cutout"]
        ei = inputs["extract_index"]
        flux = np.asarray(fluxes_np, dtype=np.float64)[ei["ti"], ei["slot"]]
        fvar = np.asarray(var_np, dtype=np.float64)[ei["ti"], ei["slot"]]
        ferr = np.sqrt(np.maximum(fvar, 0.0))
        lam = sample_map_bilinear_vec(cutout.get("cwave_map"), ei["sx"], ei["sy"])
        band = sample_map_bilinear_vec(cutout.get("cband_map"), ei["sx"], ei["sy"])
        return (ei["ci"], flux, ferr, lam, band), cutout.get("cwave_center")
