"""Tile-occupancy pre-scan — how wide the padded batches actually need to be.

The batched solver pads every tile's flux vector to a fixed width so the jitted
solve compiles once per run instead of once per distinct tile shape. That width
is either bucketed (``pad_bucket``) or fixed (``max_ps_cap`` / ``max_gal_cap``),
and a tile needing more slots than a fixed cap takes its whole cutout out of the
product.

Which sources land in which tile is *pure geometry* — catalog positions and
``shape_r`` projected through each cutout's WCS. It needs no pixels, no PSF and
no solve, so the true occupancy can be measured up front from the image headers
alone and the caps sized from it. That is what :func:`measure_occupancy` does,
and what ``max_ps_cap="auto"`` runs before the first solve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


@dataclass
class Occupancy:
    """Densest-tile source counts over a set of cutouts.

    ``max_ps`` / ``max_gal`` are the largest number of point sources / galaxies
    that any single tile must hold; ``per_cutout`` maps cutout index to that
    cutout's own ``(ps, gal)`` maxima, so a caller can report exactly which
    cutouts would overflow a proposed cap.
    """

    max_ps: int
    max_gal: int
    n_cutouts: int
    per_cutout: dict[int, tuple[int, int]]

    def overflowing(self, ps_cap: int | None,
                    gal_cap: int | None) -> list[int]:
        """Cutout indices that would be skipped under these caps."""
        return sorted(
            idx for idx, (ps, gal) in self.per_cutout.items()
            if (ps_cap is not None and ps > ps_cap)
            or (gal_cap is not None and gal > gal_cap)
        )


def _iter_tile_boxes(H, W, tile_size, halo):
    """Yield each tile's (x_start, y_start, x_end, y_end) halo box."""
    nx = max(1, -(-W // tile_size))
    ny = max(1, -(-H // tile_size))
    for iy in range(ny):
        for ix in range(nx):
            x0, y0 = ix * tile_size, iy * tile_size
            yield (x0 - halo, y0 - halo,
                   x0 + tile_size + halo, y0 + tile_size + halo)


def cutout_occupancy(path, sco_all, is_gal, tile_size, halo):
    """Densest-tile ``(n_point_sources, n_galaxies)`` for one cutout.

    Reads only the IMAGE header — the WCS and the array shape — so a full-field
    scan costs a header parse per cutout rather than a pixel read.
    """
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul["IMAGE"].header
        H = int(hdr["NAXIS2"])
        W = int(hdr["NAXIS1"])
        wcs = WCS(hdr).celestial

    px, py = wcs.world_to_pixel(sco_all)
    sx = np.asarray(px, dtype=np.float64)
    sy = np.asarray(py, dtype=np.float64)

    inside = ((sx > -halo) & (sx < W + halo) & (sy > -halo) & (sy < H + halo)
              & np.isfinite(sx) & np.isfinite(sy))
    ci = np.where(inside)[0]
    if ci.size == 0:
        return 0, 0
    cx, cy, cg = sx[ci], sy[ci], is_gal[ci]

    max_ps = max_gal = 0
    for xs, ys, xe, ye in _iter_tile_boxes(H, W, tile_size, halo):
        in_box = (cx >= xs) & (cx < xe) & (cy >= ys) & (cy < ye)
        if not in_box.any():
            continue
        g = cg[in_box]
        max_ps = max(max_ps, int((~g).sum()))
        max_gal = max(max_gal, int(g.sum()))
    return max_ps, max_gal


def measure_occupancy(pairs, catalog, *, tile_size, halo,
                      progress=False) -> Occupancy:
    """Scan ``(cutout_index, path)`` pairs for the densest-tile occupancy.

    ``catalog`` is the prepared reference catalog (needs ``ra``, ``dec`` and
    ``shape_r``); a source with ``shape_r > 0`` is rendered as a galaxy and
    consumes a galaxy slot, everything else a point-source slot.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    sco_all = SkyCoord(np.asarray(catalog["ra"]) * u.deg,
                       np.asarray(catalog["dec"]) * u.deg)
    is_gal = np.asarray(catalog["shape_r"], dtype=np.float64) > 0

    it = pairs
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(pairs, desc="Occupancy scan")
        except ImportError:
            pass

    per_cutout: dict[int, tuple[int, int]] = {}
    for cutout_index, path in it:
        try:
            per_cutout[cutout_index] = cutout_occupancy(
                Path(path), sco_all, is_gal, tile_size, halo)
        except Exception as exc:      # unreadable header: let the solve report it
            logger.warning("Occupancy scan skipped cutout %d: %s",
                           cutout_index, exc)

    max_ps = max((v[0] for v in per_cutout.values()), default=0)
    max_gal = max((v[1] for v in per_cutout.values()), default=0)
    return Occupancy(max_ps=max_ps, max_gal=max_gal,
                     n_cutouts=len(per_cutout), per_cutout=per_cutout)
