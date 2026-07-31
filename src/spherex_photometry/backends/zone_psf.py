"""A spatially-varying, optionally core-registered PSF for the CPU Tractor.

Brings the two PSF fixes of the JAX backend to `backend="cpu-tractor"`:

**Zone interpolation** (``psf_zone_interp``). The SPHEREx PSF varies across the
focal plane; the L2 cube ships one plane per PSF zone (~185 detector px pitch).
A single whole-cutout kernel mis-renders every source that sits in a
neighbouring zone. :class:`ZoneBlendedPSF` blends the delivered zone kernels
bilinearly at each evaluation position (the SPHEREx Sky Simulator convention,
clamped at the lattice edge), matching
:func:`spherex_photometry.prepare.zone_bilinear_weights`.

**Core re-registration** (``psf_core_shift``). The delivered kernel's core sits
~-0.05 native px from its declared fiducial per axis (per detector and zone,
measured; :mod:`spherex_photometry.calib`). Each zone stamp is shifted by its
measured correction — plus the fixed 10x->5x binning grid term — BEFORE
blending, mirroring the JAX engine's per-basis-element phase ramps.

Cost model: positions are quantized to a ``grid``-px cell (default 15, the JAX
backend's tile size, so the two backends see the SAME piecewise-constant PSF
field) and one blended :class:`OversampledPixelizedPSF` delegate is built and
cached per cell — a ~40x40 px cutout costs at most 9 blends, not one per
source. The upstream Tractor dispatches ``getPointSourcePatch`` /
``getFourierTransform`` through the parent's ``sampling != 1`` branch, so only
the two ``_getOversampled*`` hooks are overridden.
"""

from __future__ import annotations

import numpy as np
from tractor.psf import lanczos_shift_image

from .cpu_psf import OversampledPixelizedPSF


def shift_stamp_native(stamp, dy_native, dx_native, sampling):
    """Shift a unit-flux oversampled stamp by (dy, dx) NATIVE pixels.

    ``sampling`` native px per stamp px (0.2 for the standard 5x stamp), so the
    stamp-grid shift is ``d/sampling``. Sum is renormalized: the Lanczos kernel
    loses a little flux off the stamp edge, whereas the engine's Fourier phase
    ramp is exactly flux-preserving (DC term untouched) — without the
    renormalization the two backends would disagree at the ~1e-4 level on every
    flux, which is exactly the kind of silent cross-backend drift this package
    exists to avoid.
    """
    dx = dx_native / sampling
    dy = dy_native / sampling
    # tractor's lanczos_shift_image asserts |fraction| <= 0.5 (its C kernel is
    # sub-pixel only), and a realistic core shift + grid term is ~0.57 stamp
    # px at 5x. Split into an integer roll (zero-filled, exact) and a
    # fractional Lanczos shift.
    ix, iy = int(np.round(dx)), int(np.round(dy))
    out = np.asarray(stamp, dtype=np.float64)
    if ix or iy:
        rolled = np.zeros_like(out)
        h, w = out.shape
        ys0, ys1 = max(0, iy), min(h, h + iy)
        xs0, xs1 = max(0, ix), min(w, w + ix)
        rolled[ys0:ys1, xs0:xs1] = out[ys0 - iy:ys1 - iy, xs0 - ix:xs1 - ix]
        out = rolled
    fx, fy = dx - ix, dy - iy
    if abs(fx) > 1e-12 or abs(fy) > 1e-12:
        out = lanczos_shift_image(out, fx, fy)
    s_in, s_out = float(np.sum(stamp)), float(np.sum(out))
    if s_out > 0:
        out *= s_in / s_out
    return out


class ZoneBlendedPSF:
    """Position-dependent PSF: bilinear zone blend, evaluated per grid cell.

    Parameters
    ----------
    stamps : sequence of (H, W) arrays
        The 5x-oversampled native zone stamps, aligned with ``zones_tab`` rows
        (already unit-normalized; core shifts applied by the caller).
    zones_tab
        The cutout's ``psf_zones`` table (columns ``x``, ``y``, ``plane_idx``).
    pix_to_det : callable
        ``(x_cut, y_cut) -> (x_orig, y_orig)`` cutout-pixel to detector-pixel.
    sampling : float
        Native px per stamp px (0.2 for 5x).
    weights_fn : callable
        ``(zones_tab, x_orig, y_orig) -> (K,) weights`` — pass
        :func:`spherex_photometry.prepare.zone_bilinear_weights` so both
        backends share one convention (and one set of unit tests).
    grid : int
        Quantization cell in cutout px. Matches the JAX tile size by default
        so the piecewise-constant PSF field is identical across backends.
    """

    def __init__(self, stamps, zones_tab, pix_to_det, sampling, weights_fn,
                 grid=15):
        if len(stamps) == 0:
            raise ValueError("ZoneBlendedPSF needs at least one zone stamp")
        self._stamps = [np.asarray(s, dtype=np.float64) for s in stamps]
        self._zones = zones_tab
        self._pix_to_det = pix_to_det
        self._sampling = float(sampling)
        self._weights_fn = weights_fn
        self._grid = int(grid)
        self._delegates = {}
        # Nominal delegate for anything position-independent tractor probes
        # (radius, sizes). All stamps share one shape, so any cell works.
        self._nominal = self._delegate(0.0, 0.0)

    # -- delegate management -------------------------------------------------
    def _cell(self, px, py):
        g = self._grid
        return (int(np.floor(px / g)), int(np.floor(py / g)))

    def _delegate(self, px, py):
        key = self._cell(px, py)
        d = self._delegates.get(key)
        if d is None:
            g = self._grid
            cx, cy = (key[0] + 0.5) * g, (key[1] + 0.5) * g   # cell centre
            x_orig, y_orig = self._pix_to_det(cx, cy)
            w = np.asarray(self._weights_fn(self._zones, x_orig, y_orig),
                           dtype=np.float64)
            if w.shape[0] != len(self._stamps):
                raise ValueError("weights length does not match the zone basis")
            blended = np.tensordot(w, np.stack(self._stamps), axes=(0, 0))
            d = OversampledPixelizedPSF(blended.astype(np.float32),
                                        sampling=self._sampling)
            self._delegates[key] = d
        return d

    # -- the two oversampled entry points the parent dispatches to -----------
    def getPointSourcePatch(self, px, py, **kwargs):
        return self._delegate(px, py).getPointSourcePatch(px, py, **kwargs)

    def getFourierTransform(self, px, py, radius):
        return self._delegate(px, py).getFourierTransform(px, py, radius)

    # -- position-independent probes forwarded to the nominal delegate -------
    def getRadius(self):
        return self._nominal.getRadius()

    def getFourierTransformSize(self, radius):
        return self._nominal.getFourierTransformSize(radius)

    def __getattr__(self, name):
        # Anything else tractor probes (constantPsfAt, hashkey, shape, ...)
        # is position-independent for our stamps; serve it from the nominal
        # delegate rather than enumerating the upstream duck API here.
        return getattr(self._nominal, name)

    def __str__(self):
        return (f"ZoneBlendedPSF(K={len(self._stamps)}, grid={self._grid}, "
                f"{len(self._delegates)} cells built)")


def zone_stamp_provider(cutout, cfg, *, prepare):
    """Return ``f(zone_row) -> stamp``, cached, for one row of ``psf_zones``.

    The stamp is the 5x-oversampled native kernel of that zone: 2x-downsampled
    from the delivered 10x cube plane, normalized to unit flux, and — when
    ``cfg.psf_core_shift`` is set — Lanczos-shifted by that zone's measured core
    offset plus the fixed binning-grid term. Each plane is resolved at most once
    per cutout, so a 49-tile cutout over 12 zones pays 12 downsamples.

    ``prepare`` is the :mod:`spherex_photometry.prepare` module (passed in to
    keep this module import-light for tractor-less environments).
    """
    zones = cutout.psf_zones
    cube = cutout.psf_cube
    core_shift = bool(getattr(cfg, "psf_core_shift", False))
    sampling = cfg.psf_sampling
    det = int(cutout.detector) if core_shift else None
    cache: dict[int, np.ndarray] = {}

    def get(row):
        row = int(row)
        stamp = cache.get(row)
        if stamp is None:
            stamp = prepare.downsample_psf_oversample2(
                cube[int(zones["plane_idx"][row])])
            total = stamp.sum()
            if total > 0:
                stamp = stamp / total          # unit flux before any shift
            if core_shift:
                from ..calib import DOWNSAMPLE_GRID_SHIFT_NATIVE, psf_core_shift
                z = int(zones["zone_id"][row])
                s = psf_core_shift(det, z)
                if s.source != "zone":
                    raise ValueError(
                        f"psf_core_shift(det={det}, zone={z}) fell back to "
                        f"{s.source!r}; coverage is 726/726, so a fallback "
                        "means the detector or zone_id is wrong")
                stamp = shift_stamp_native(
                    stamp,
                    s.dy_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                    s.dx_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                    sampling)
            cache[row] = stamp
        return stamp

    return get


def nearest_zone_row(zones, x_orig, y_orig) -> int:
    """Row of ``psf_zones`` whose centre is nearest detector ``(x_orig, y_orig)``.

    The row index rather than ``plane_idx`` (which
    :func:`spherex_photometry.prepare.select_zone_plane` returns), because the
    core-shift table is keyed on ``zone_id`` and only the row knows both.
    """
    dx = np.asarray(zones["x"], dtype=np.float64) - float(x_orig)
    dy = np.asarray(zones["y"], dtype=np.float64) - float(y_orig)
    return int(np.argmin(dx * dx + dy * dy))


def resolve_zone_stamps(cutout, cfg, *, prepare):
    """Return ``(stamps, interp)`` — the zone kernels the CPU PSF is built from.

    ``stamps`` are unit-flux 5x-oversampled native stamps with the config's core
    shifts already applied. ``interp`` says whether they form a blend basis
    (aligned with ``cutout.psf_zones``) or are the single centre-zone kernel of
    the pre-fix behaviour.
    """
    zones = cutout.psf_zones
    interp = bool(getattr(cfg, "psf_zone_interp", True)) and len(zones) > 1
    get = zone_stamp_provider(cutout, cfg, prepare=prepare)

    if interp:
        return [get(r) for r in range(len(zones))], True

    # centre-zone kernel, the pre-fix behaviour
    H, W = cutout.image.shape
    xo, yo = prepare.cutout_to_orig(W / 2.0, H / 2.0,
                                    crpix1a=cutout.crpix1a,
                                    crpix2a=cutout.crpix2a)
    return [get(nearest_zone_row(zones, xo, yo))], False


def build_cpu_psf(cutout, cfg, *, prepare):
    """Resolve the CPU backend's PSF per the config's two PSF-fix flags.

    Returns a PSF object for ``tractor.Image``: an
    :class:`OversampledPixelizedPSF` when the cutout is single-zone or
    interpolation is off, a :class:`ZoneBlendedPSF` otherwise. Core shifts,
    when enabled, are applied to the zone stamps in either case.

    This is the WHOLE-CUTOUT form, where one ``tractor.Image`` covers every
    source and the PSF must therefore vary with position. The tiled path wants
    :func:`build_cpu_psf_selector` instead: one constant kernel per tile.
    """
    stamps, interp = resolve_zone_stamps(cutout, cfg, prepare=prepare)

    if not interp:
        return OversampledPixelizedPSF(stamps[0].astype(np.float32),
                                       sampling=cfg.psf_sampling)

    crpix1a, crpix2a = cutout.crpix1a, cutout.crpix2a

    def pix_to_det(x_cut, y_cut):
        return prepare.cutout_to_orig(x_cut, y_cut,
                                      crpix1a=crpix1a, crpix2a=crpix2a)

    return ZoneBlendedPSF(stamps, cutout.psf_zones, pix_to_det, cfg.psf_sampling,
                          prepare.zone_bilinear_weights,
                          grid=getattr(cfg, "tile_size", 15))


def build_cpu_psf_selector(cutout, cfg, *, prepare):
    """Return ``f(x_cut, y_cut) -> PSF``: one CONSTANT kernel per tile.

    The tiled CPU solve gives each tile its own small ``tractor.Image``, so the
    PSF does not have to vary inside it — and must not, since the tile image
    carries tile-local pixel coordinates that a position-dependent
    :class:`ZoneBlendedPSF` would misread. This mirrors the JAX backend, which
    resolves the kernel once at each tile's CORE CENTRE and renders the whole
    tile (halo neighbours included) with it, in both branches:

    * ``psf_zone_interp=True`` -> the bilinear zone blend at that position;
    * ``psf_zone_interp=False`` -> the nearest zone's kernel at that position
      (:func:`spherex_photometry.prepare.zone_psf_selector` on the JAX side) —
      NOT the whole-cutout centre zone, which is what the untiled CPU path uses
      and which would put every off-centre tile on the wrong kernel.

    Results are cached, so a whole cutout costs at most one blend per tile (and
    one kernel per zone when interpolation is off).
    """
    zones = cutout.psf_zones
    interp = bool(getattr(cfg, "psf_zone_interp", True)) and len(zones) > 1
    get_stamp = zone_stamp_provider(cutout, cfg, prepare=prepare)
    crpix1a, crpix2a = cutout.crpix1a, cutout.crpix2a
    sampling = cfg.psf_sampling
    basis = None
    cache: dict = {}

    def _psf(key, make_stamp):
        psf = cache.get(key)
        if psf is None:
            psf = OversampledPixelizedPSF(
                np.asarray(make_stamp(), dtype=np.float32), sampling=sampling)
            cache[key] = psf
        return psf

    def select(x_cut, y_cut):
        x_orig, y_orig = prepare.cutout_to_orig(
            x_cut, y_cut, crpix1a=crpix1a, crpix2a=crpix2a)
        if not interp:
            # keyed on the zone row: single-zone cutouts build exactly one PSF
            row = nearest_zone_row(zones, x_orig, y_orig)
            return _psf(row, lambda r=row: get_stamp(r))

        def blend():
            nonlocal basis
            if basis is None:
                basis = np.stack([get_stamp(r) for r in range(len(zones))])
            w = np.asarray(prepare.zone_bilinear_weights(zones, x_orig, y_orig),
                           dtype=np.float64)
            return np.tensordot(w, basis, axes=(0, 0))

        return _psf((round(float(x_cut), 6), round(float(y_cut), 6)), blend)

    return select
