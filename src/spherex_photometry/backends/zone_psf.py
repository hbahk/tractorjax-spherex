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


def build_cpu_psf(cutout, cfg, *, prepare):
    """Resolve the CPU backend's PSF per the config's two PSF-fix flags.

    ``prepare`` is the :mod:`spherex_photometry.prepare` module (passed in to
    keep this module import-light for tractor-less environments).

    Returns a PSF object for ``tractor.Image``: an
    :class:`OversampledPixelizedPSF` when the cutout is single-zone or
    interpolation is off, a :class:`ZoneBlendedPSF` otherwise. Core shifts,
    when enabled, are applied to the zone stamps in either case.
    """
    zones = cutout.psf_zones
    K = len(zones)
    interp = bool(getattr(cfg, "psf_zone_interp", True)) and K > 1
    H, W = cutout.image.shape

    if interp:
        stamps = [prepare.downsample_psf_oversample2(cutout.psf_cube[int(p)])
                  for p in np.asarray(zones["plane_idx"])]
        zids = np.asarray(zones["zone_id"], dtype=int)
    else:
        # centre-zone kernel, the pre-fix behaviour
        stamps = [prepare.select_psf_native(cutout, W / 2.0, H / 2.0)]
        xo, yo = prepare.cutout_to_orig(W / 2.0, H / 2.0,
                                        crpix1a=cutout.crpix1a,
                                        crpix2a=cutout.crpix2a)
        dxz = np.asarray(zones["x"]) - xo
        dyz = np.asarray(zones["y"]) - yo
        zids = np.asarray([zones["zone_id"][int(np.argmin(
            dxz * dxz + dyz * dyz))]], dtype=int)

    # unit flux before any shift
    stamps = [s / s.sum() if s.sum() > 0 else s for s in stamps]

    if bool(getattr(cfg, "psf_core_shift", False)):
        from ..calib import DOWNSAMPLE_GRID_SHIFT_NATIVE, psf_core_shift
        det = int(cutout.detector)
        shifted = []
        for stamp, z in zip(stamps, np.asarray(zids, dtype=int)):
            s = psf_core_shift(det, int(z))
            if s.source != "zone":
                raise ValueError(
                    f"psf_core_shift(det={det}, zone={int(z)}) fell back to "
                    f"{s.source!r}; coverage is 726/726, so a fallback means "
                    "the detector or zone_id is wrong")
            shifted.append(shift_stamp_native(
                stamp,
                s.dy_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                s.dx_apply + DOWNSAMPLE_GRID_SHIFT_NATIVE,
                cfg.psf_sampling))
        stamps = shifted

    if not interp:
        return OversampledPixelizedPSF(stamps[0].astype(np.float32),
                                       sampling=cfg.psf_sampling)

    crpix1a, crpix2a = cutout.crpix1a, cutout.crpix2a

    def pix_to_det(x_cut, y_cut):
        return prepare.cutout_to_orig(x_cut, y_cut,
                                      crpix1a=crpix1a, crpix2a=crpix2a)

    return ZoneBlendedPSF(stamps, zones, pix_to_det, cfg.psf_sampling,
                          prepare.zone_bilinear_weights,
                          grid=getattr(cfg, "tile_size", 15))
