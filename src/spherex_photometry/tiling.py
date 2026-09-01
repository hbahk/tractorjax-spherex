"""Backend-neutral tile geometry: the 15 px core + 3 px halo grid.

Both backends split a cutout into the same tiles, so the geometry lives here
rather than inside either one. The JAX backend solves all of a cutout's tiles in
one ``vmap``; the CPU backend runs one upstream-Tractor
``optimize_forced_photometry`` per tile. Sharing this module is what makes a
CPU-vs-GPU comparison a comparison of *engines* rather than of geometries.

The conventions, in one place, because every consumer depends on them:

* Cores tile ``[0, W) x [0, H)`` exactly — the last column/row of tiles is
  clipped at the cutout edge, so **every in-cutout pixel and source belongs to
  exactly one core**. That is what makes halo overlaps impossible to
  double-count and in-cutout sources impossible to drop.
* Halo boxes are *not* clipped: they run from ``-halo`` to ``W + halo`` at the
  edges and are zero-padded by :func:`extract_tile_region`. A tile's model
  therefore includes the neighbours whose PSF wings reach into it.
* A source enters a tile's model when its centre lies in the **halo box**
  (``x_start <= x < x_end``); it is *reported* from the tile whose **core box**
  contains it (``core_x0 <= x < core_x1``).
"""

from __future__ import annotations

import math

import numpy as np

from .constants import SPHEREX_PIXSCALE


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

    Only the reference pixel moves — the CD matrix is untouched — so a tile
    never needs its own WCS just to get the pixel scale; see
    :func:`cd_inv_from_wcs`.
    """
    return wcs.slice((slice(int(y_start), int(y_start) + 10**6),
                      slice(int(x_start), int(x_start) + 10**6)))


def cd_inv_from_wcs(wcs):
    """World-to-pixel linear map (the inverted CD matrix) as ``float32``.

    This is the only thing the engine takes from a WCS, and it is a property of
    the *cutout*: :func:`shift_wcs` moves the reference pixel and leaves the CD
    matrix bit-identical, so every tile of a cutout shares this matrix and it
    can be computed once per cutout rather than once per tile.

    Falls back to the nominal SPHEREx pixel scale for a WCS with no usable
    linear part, and to the identity for a singular one, rather than raising —
    a degenerate WCS should surface as a bad fit, not as an exception inside
    the batch builder.
    """
    try:
        # astropy raises AttributeError from .cd on a PC+CDELT WCS (which is
        # what the SPHEREx cutouts carry), so hasattr is the branch, exactly as
        # the pre-refactor code in the JAX backend had it.
        cd = (np.asarray(wcs.wcs.cd) if hasattr(wcs.wcs, "cd")
              else np.asarray(wcs.pixel_scale_matrix))
    except Exception:  # noqa: BLE001 - any unusable WCS falls back to nominal
        cd = np.eye(2) * (SPHEREX_PIXSCALE / 3600.0)
    try:
        return np.linalg.inv(cd).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        return np.eye(2, dtype=np.float32)


def tile_core_index(tile_metas, sx, sy):
    """Index of the tile whose CORE box contains each ``(sx, sy)``.

    ``tile_metas`` is the list of dicts from :func:`iter_tiles`, in any order.
    Returns ``-1`` where the position falls in no core (i.e. outside the
    cutout). Vectorised over the position arrays.
    """
    cx0 = np.array([m["core_x0"] for m in tile_metas])
    cy0 = np.array([m["core_y0"] for m in tile_metas])
    x_edges = np.unique(cx0)
    y_edges = np.unique(cy0)
    lut = np.full((len(y_edges), len(x_edges)), -1, dtype=np.int64)
    lut[np.searchsorted(y_edges, cy0), np.searchsorted(x_edges, cx0)] = \
        np.arange(len(tile_metas))

    sx = np.asarray(sx, dtype=np.float64)
    sy = np.asarray(sy, dtype=np.float64)
    ix = np.searchsorted(x_edges, sx, side="right") - 1
    iy = np.searchsorted(y_edges, sy, side="right") - 1
    ok = (ix >= 0) & (iy >= 0) & np.isfinite(sx) & np.isfinite(sy)
    out = np.full(sx.shape, -1, dtype=np.int64)
    if np.any(ok):
        out[ok] = lut[iy[ok], ix[ok]]
    # searchsorted alone cannot reject a position past the LAST core's upper
    # edge (x >= W): it just returns the last tile. Check the upper bounds of
    # the tile actually selected.
    hit = out >= 0
    if np.any(hit):
        cx1 = np.array([m["core_x1"] for m in tile_metas])
        cy1 = np.array([m["core_y1"] for m in tile_metas])
        sel = out[hit]
        inside = (sx[hit] < cx1[sel]) & (sy[hit] < cy1[sel])
        out[np.where(hit)[0][~inside]] = -1
    return out
