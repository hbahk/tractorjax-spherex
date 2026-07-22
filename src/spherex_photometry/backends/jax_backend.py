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
"""

from __future__ import annotations

import math

import numpy as np

from ..constants import SPHEREX_PIXSCALE
from ..io.cutouts import sample_map_bilinear_vec
from ..prepare import prepare_pixels, project_sources, select_psf_native
from .base import FieldContext

# tractor-jax engine (imports jax).
from tractor_jax.jax import batching as tjb
from tractor_jax.jax.pipeline import prefetch_pipeline  # noqa: F401  (re-exported)


# --------------------------------------------------------------------------- #
# Tiling helpers
# --------------------------------------------------------------------------- #
def iter_tiles(H, W, tile_size, halo):
    """Yield tile metadata covering an H x W cutout (core box clipped, halo padded)."""
    nx = max(1, math.ceil(W / tile_size))
    ny = max(1, math.ceil(H / tile_size))
    for iy in range(ny):
        for ix in range(nx):
            x0 = ix * tile_size
            y0 = iy * tile_size
            core_x1 = min(x0 + tile_size, W)
            core_y1 = min(y0 + tile_size, H)
            yield {
                "ix": ix, "iy": iy,
                "core_x0": x0, "core_y0": y0,
                "core_x1": core_x1, "core_y1": core_y1,
                "x_start": x0 - halo, "y_start": y0 - halo,
                "x_end": x0 + tile_size + halo, "y_end": y0 + tile_size + halo,
            }


def extract_tile_region(arr, x_start, y_start, x_end, y_end, fill=0.0):
    """Slice ``arr[y_start:y_end, x_start:x_end]``, zero-padding out-of-bounds."""
    H, W = arr.shape
    th = y_end - y_start
    tw = x_end - x_start
    out = np.full((th, tw), fill, dtype=arr.dtype)
    im_x0 = max(0, x_start)
    im_y0 = max(0, y_start)
    im_x1 = min(W, x_end)
    im_y1 = min(H, y_end)
    if im_x1 > im_x0 and im_y1 > im_y0:
        out[im_y0 - y_start: im_y1 - y_start,
            im_x0 - x_start: im_x1 - x_start] = arr[im_y0:im_y1, im_x0:im_x1]
    return out


def shift_wcs(wcs, x_start, y_start):
    """Return a WCS whose pixel (0,0) maps to the original ``(x_start, y_start)``.

    Uses ``WCS.slice`` so both ``wcs.wcs.crpix`` and ``wcs.sip.crpix`` shift
    together (hand-editing crpix alone mis-projects the SIP polynomial).
    """
    return wcs.slice((slice(int(y_start), int(y_start) + 10**6),
                      slice(int(x_start), int(x_start) + 10**6)))


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
    except Exception:
        cd_matrix = np.eye(2) * (SPHEREX_PIXSCALE / 3600.0)
    try:
        cd_inv = np.linalg.inv(cd_matrix).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        cd_inv = np.eye(2, dtype=np.float32)

    views = [{
        "data": t["data"], "invvar": t["invvar"], "psf": t["psf"],
        "src_indices": t["src_indices"],
        "origin": (t["tile_meta"]["x_start"], t["tile_meta"]["y_start"]),
    } for t in tile_records]

    bundle = tjb.build_padded_batches(
        views, catalog_full, sx_all, sy_all,
        psf_sampling=psf_sampling, fixed_max_factor=fixed_max_factor,
        fit_background=fit_background, profile_lookup_fn=profile_lookup_fn,
        cd_inv=cd_inv, max_ps_cap=max_ps_cap, max_gal_cap=max_gal_cap,
        max_mog_k_cap=max_mog_k_cap, pad_bucket=pad_bucket)
    return (bundle.images_data, bundle.batches, bundle.initial_fluxes,
            bundle.meta["src_slot"])


def build_cutout_tiles(cutout, *, sx_all, sy_all, tile_size, halo,
                       data_scaled, invvar_scaled, psf_native):
    """Construct tile records (core + halo boxes) for one cutout."""
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
        tile_records.append({
            "data": extract_tile_region(data_scaled, xs, ys, xe, ye, fill=0.0),
            "invvar": extract_tile_region(invvar_scaled, xs, ys, xe, ye, fill=0.0),
            "psf": psf_native,
            "wcs": shift_wcs(cutout["wcs"], xs, ys),
            "src_indices": idxs,
            "tile_meta": meta,
        })
    return tile_records


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

        sx_main = float(sx_all[ctx.main_idx])
        sy_main = float(sy_all[ctx.main_idx])
        psf_native = select_psf_native(cutout, sx_main, sy_main)

        tile_records = build_cutout_tiles(
            cutout, sx_all=sx_all, sy_all=sy_all,
            tile_size=cfg.tile_size, halo=cfg.tile_halo,
            data_scaled=data, invvar_scaled=invvar, psf_native=psf_native)

        max_ps, max_gal, max_mog_k = cfg.resolved_caps()
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
