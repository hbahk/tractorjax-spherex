"""GPU/JAX backend: tiled batched forced photometry on the tractor-jax engine.

Lifted from the production driver, de-globalized so every knob comes from a
:class:`~spherex_photometry.config.PhotometryConfig`. Imports JAX (via the
tractor-jax engine) at module import, so this module is imported lazily by
:func:`spherex_photometry.backends.get_backend` after the device is configured.

The engine renders every source on a 5x-oversampled grid (``psf_sampling=0.2``,
``fixed_max_factor=5``) and sum-bins to native pixels, so the PSF x source-shape
convolution is done at oversampled resolution — the accurate low-resolution flux
estimate. Cutouts are split into ``tile_size`` cores with a halo and all tiles
of a cutout are solved in one ``vmap``.

The tile grid itself lives in :mod:`spherex_photometry.tiling` and is shared with
the ``cpu-tractor`` backend (re-exported here for callers that import it from
this module), so a CPU-vs-GPU comparison is a comparison of engines rather than
of geometries.
"""

from __future__ import annotations

import numpy as np

# tractor-jax engine (imports jax).
from tractor_jax.jax import batching as tjb
from tractor_jax.jax.pipeline import prefetch_pipeline  # noqa: F401  (re-exported)

from ..config import CapExceededError, ConfigError
from ..constants import SPHEREX_PIXSCALE
from ..io.cutouts import sample_map_bilinear_vec
from ..prepare import (
    prepare_pixels,
    project_sources,
    zone_psf_basis,
    zone_psf_selector,
)
from ..tiling import (
    extract_tile_region,
    iter_tiles,
    shift_wcs,
)
from .base import FieldContext


# --------------------------------------------------------------------------- #
# Batch build
# --------------------------------------------------------------------------- #
def extract_tiled_batches(tile_records, catalog_full, sx_all, sy_all,
                          psf_sampling=0.2, fixed_max_factor=5.0,
                          fit_background=True, profile_lookup_fn=None,
                          max_ps_cap=None, max_gal_cap=None,
                          max_mog_k_cap=None, pad_bucket=None):
    """Build vmap-ready padded batches for a cutout's tiles (engine call)."""
    if len(tile_records) == 0:
        raise ValueError("extract_tiled_batches: tile_records is empty")

    wcs0 = tile_records[0]["wcs"]
    try:
        cd_matrix = (np.asarray(wcs0.wcs.cd) if hasattr(wcs0.wcs, "cd")
                     else np.asarray(wcs0.pixel_scale_matrix))
    except Exception:  # noqa: BLE001 - any unusable WCS falls back to the nominal scale
        cd_matrix = np.eye(2) * (SPHEREX_PIXSCALE / 3600.0)
    try:
        cd_inv = np.linalg.inv(cd_matrix).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        cd_inv = np.eye(2, dtype=np.float32)

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
        max_mog_k_cap=max_mog_k_cap, pad_bucket=pad_bucket)
    return (bundle.images_data, bundle.batches, bundle.initial_fluxes,
            bundle.meta["src_slot"])


def build_cutout_tiles(cutout, *, sx_all, sy_all, tile_size, halo,
                       data_scaled, invvar_scaled, psf_native=None,
                       psf_select=None, psf_basis=None, psf_weights=None,
                       psf_basis_shifts=None):
    """Construct tile records (core + halo boxes) for one cutout.

    ``psf_select(x, y) -> stamp`` (from
    :func:`~spherex_photometry.prepare.zone_psf_selector`) gives each tile the
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

    tile_records = []
    for meta in iter_tiles(H, W, tile_size, halo):
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
            "psf": psf_select(cx, cy),
            "wcs": shift_wcs(cutout["wcs"], xs, ys),
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
            rec["psf_weights"] = psf_weights(cx, cy)
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


def make_tiled_solver(spec, batches_in_axes, penalty_weights=None,
                      prior_arrays=None):
    """Build the vmapped per-tile solver for ``spec`` (thin engine wrapper)."""
    kind = spec["kind"]
    if kind == "linear":
        return tjb.make_batched_solver(
            "linear", in_axes=batches_in_axes, rcond=spec.get("rcond", 1e-12))
    if kind == "eigfloor":
        return tjb.make_batched_solver(
            "eigfloor", in_axes=batches_in_axes, floor=spec.get("floor", 1e-2))
    if kind == "eigfloor_prior":
        jfn = tjb.make_batched_solver(
            "eigfloor_prior", in_axes=batches_in_axes,
            floor=spec.get("floor", 1e-2))
        lam0, fp0 = prior_arrays if prior_arrays is not None else (None, None)
        return lambda init, imgd, bat, lam=lam0, fp=fp0: jfn(init, imgd, bat, lam, fp)
    if kind == "lasso":
        jfn = tjb.make_batched_solver(
            "lasso", in_axes=batches_in_axes,
            alpha=spec.get("alpha", "auto"), penalty_mode="snr",
            nonneg=True, debias=True,
            debias_signfree=spec.get("debias_signfree", "protected"),
            n_iter=spec.get("n_iter", 1000))
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
        self._solve_fn_cache: dict = {}

    # ---- build (CPU stage; prefetch-safe) --------------------------------
    def build(self, cutout, ctx: FieldContext):
        cfg = self.config
        prepared = prepare_pixels(cutout, cfg)
        data, invvar = prepared.data, prepared.invvar
        H, W = data.shape

        sx_all, sy_all = project_sources(cutout, ctx.sco_all)

        basis = weights = basis_shifts = None
        if getattr(cfg, "psf_zone_interp", True):
            basis, weights = zone_psf_basis(cutout)
        if getattr(cfg, "psf_core_shift", False):
            if basis is None:
                raise ConfigError(
                    "psf_core_shift on the JAX backend requires "
                    "psf_zone_interp=True (the shifts ride on the zone "
                    "basis); the cpu-tractor backend supports it standalone.")
            from ..calib import DOWNSAMPLE_GRID_SHIFT_NATIVE, psf_core_shift
            det = int(cutout.detector)
            rows = []
            for z in np.asarray(cutout.psf_zones["zone_id"], dtype=int):
                cs = psf_core_shift(det, int(z))
                if cs.source != "zone":
                    raise ValueError(
                        f"psf_core_shift(det={det}, zone={int(z)}) fell back "
                        f"to {cs.source!r}; coverage is 726/726, so a "
                        "fallback means the detector or zone_id is wrong")
                rows.append((cs.dy_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                             cs.dx_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE))
            basis_shifts = np.asarray(rows, dtype=np.float64)
        tile_records = build_cutout_tiles(
            cutout, sx_all=sx_all, sy_all=sy_all,
            tile_size=cfg.tile_size, halo=cfg.tile_halo,
            data_scaled=data, invvar_scaled=invvar,
            psf_select=zone_psf_selector(cutout),
            psf_basis=basis, psf_weights=weights,
            psf_basis_shifts=basis_shifts)

        max_ps, max_gal, max_mog_k = cfg.resolved_caps(ctx.occupancy)
        _check_caps(tile_records, ctx.catalog, max_ps, max_gal,
                    cutout_index=getattr(ctx, "cutout_index", None))
        images_data, batches, initial_fluxes, src_slot = extract_tiled_batches(
            tile_records, ctx.catalog, sx_all, sy_all,
            psf_sampling=cfg.psf_sampling, fixed_max_factor=cfg.fixed_max_factor,
            fit_background=True, profile_lookup_fn=ctx.profile_lookup_fn,
            max_ps_cap=max_ps, max_gal_cap=max_gal, max_mog_k_cap=max_mog_k,
            pad_bucket=cfg.pad_bucket or None)

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
                    n_prior_free=n_prior_free)

    # ---- solve (GPU stage) -----------------------------------------------
    def solve(self, inputs):
        spec = self.config.solver_spec()
        protect_ci = inputs.get("protect_ci")
        batches = inputs["batches"]
        batches_in_axes = inputs["batches_in_axes"]
        initial_fluxes = inputs["initial_fluxes"]
        tile_chunk = self.config.tile_chunk

        solve_fn = None
        lam_d = f_pr = pw = None
        if spec["kind"] == "lasso":
            n_tiles = int(np.asarray(initial_fluxes).shape[0])
            n_flux = int(np.asarray(initial_fluxes).shape[1])
            pw = tjb.penalty_weights_from_slots(
                inputs["src_slot"], n_tiles, n_flux, protect_ci or set())
            solve_fn = make_tiled_solver(spec, batches_in_axes, penalty_weights=pw)
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
                                         prior_arrays=(lam_d, f_pr))
        else:
            struct_key = tuple(sorted(batches.keys()))
            solve_fn = self._solve_fn_cache.get(struct_key)
            if solve_fn is None:
                solve_fn = make_tiled_solver(spec, batches_in_axes)
                self._solve_fn_cache[struct_key] = solve_fn

        n_tiles_total = int(np.asarray(initial_fluxes).shape[0])
        if not tile_chunk or n_tiles_total <= tile_chunk:
            fluxes, variances = solve_fn(
                initial_fluxes, inputs["images_data"], batches)
            return np.asarray(fluxes), np.asarray(variances)

        return self._solve_chunked(inputs, spec, solve_fn, batches,
                                   batches_in_axes, initial_fluxes,
                                   n_tiles_total, tile_chunk, pw, lam_d, f_pr)

    @staticmethod
    def _solve_chunked(inputs, spec, solve_fn, batches, batches_in_axes,
                       initial_fluxes, n_tiles_total, tile_chunk, pw, lam_d, f_pr):
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

        outs_f, outs_v = [], []
        for start in range(0, n_tiles_total, tile_chunk):
            end = min(start + tile_chunk, n_tiles_total)
            pad = tile_chunk - (end - start)
            init_c = _cut(initial_fluxes, start, end, pad)
            imgd_c = {k: _cut(v, start, end, pad)
                      for k, v in inputs["images_data"].items()}
            bat_c = _cut_tree(batches, batches_in_axes, start, end, pad)
            if spec["kind"] == "lasso":
                f, v = solve_fn(init_c, imgd_c, bat_c, _cut(pw, start, end, pad))
            elif spec["kind"] == "eigfloor_prior":
                f, v = solve_fn(init_c, imgd_c, bat_c,
                                _cut(lam_d, start, end, pad),
                                _cut(f_pr, start, end, pad))
            else:
                f, v = solve_fn(init_c, imgd_c, bat_c)
            outs_f.append(np.asarray(f)[:end - start])
            outs_v.append(np.asarray(v)[:end - start])
        return np.concatenate(outs_f, axis=0), np.concatenate(outs_v, axis=0)

    # ---- extract ---------------------------------------------------------
    def extract(self, inputs, fluxes_np, var_np):
        cutout = inputs["cutout"]
        ei = inputs["extract_index"]
        flux = np.asarray(fluxes_np, dtype=np.float64)[ei["ti"], ei["slot"]]
        fvar = np.asarray(var_np, dtype=np.float64)[ei["ti"], ei["slot"]]
        ferr = np.sqrt(np.maximum(fvar, 0.0))
        lam = sample_map_bilinear_vec(cutout.get("cwave_map"), ei["sx"], ei["sy"])
        band = sample_map_bilinear_vec(cutout.get("cband_map"), ei["sx"], ei["sy"])
        return (ei["ci"], flux, ferr, lam, band), cutout.get("cwave_center")
