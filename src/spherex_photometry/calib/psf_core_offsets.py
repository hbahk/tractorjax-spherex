"""Loader for the static SPHEREx L2 PSF core-offset table.

The problem this solves
-----------------------
A delivered L2 PSF plane is 101x101 at ``OVERSAMP = 10`` with
``CRPIX1 = CRPIX2 = 51.0``, so its *declared* fiducial is the 0-based array
index ``(50.0, 50.0)`` -- the exact geometric centre of the odd-sized array.
The renderer puts that array centre on the source's projected catalog position.
But the PSF **core** is not at the array centre: it sits about ``-0.05`` native
px away on *both* axes.  Every rendered source therefore lands ~0.05 native px
(~0.3 arcsec) off unless the kernel is shifted first.

Sign convention -- read this once and you will not get it wrong
--------------------------------------------------------------
There are two different numbers and they differ by a sign:

``psf_core_offset(det, zone)``  ->  ``CoreOffset(dy_core, dx_core)``
    Where the core **IS**, measured relative to the declared fiducial.
    Both components are *negative* (~ -0.05 native px).

``psf_core_shift(det, zone)``   ->  ``CoreShift(dy_apply, dx_apply)``
    What you **DO** to the kernel: ``= -offset``, so both components are
    *positive* (~ +0.05 native px).  Shift the kernel by this and the core
    lands on the declared fiducial, hence on the source position.

Consumers should call :func:`psf_core_shift`.  It is already negated; do not
negate it again.  The returned object is a ``NamedTuple`` whose fields are
literally named ``dy_apply`` / ``dx_apply``, so a mis-read is visible at the
call site::

    from spherex_gpupipe.calib import psf_core_shift
    s = psf_core_shift(detector, zone_id)
    # NATIVE px. tractor_jax's build_padded_batches takes native px and converts
    # internally, so pass these straight through:
    rec["psf_basis_shifts"] = np.array([tuple(psf_core_shift(det, z))
                                        for z in zones["zone_id"]])
    # If you instead shift the OVERSAMPLED plane yourself, scale by OVERSAMP --
    # feeding native px to a 10x-oversampled kernel under-shifts by 10x and
    # leaves the core at (-0.037, -0.057) instead of ~0.

Tuple unpacking works too and yields exactly two values ordered ``(dy, dx)``,
matching numpy axis order (axis 0 = y = rows)::

    dy_apply, dx_apply = psf_core_shift(detector, zone_id)

The provenance lives on ``.source`` and is deliberately kept *out* of the
unpacked pair so it can never be mistaken for a third coordinate.

What is deliberately *not* here
-------------------------------
The full-array flux centroid is **not** the correction.  It is ~ -0.10 native
px in x because it mixes the rigid core translation (-0.05) with a genuine
asymmetric optical wing (-0.03..-0.04).  Correcting by the centroid would
over-shift the core in x.  The wing is real optics and must be left alone.
The centroid is carried in the table as a diagnostic column only.

Fallbacks
---------
Coverage is limited to the ``(detector, zone_id)`` pairs the available cutout
bundles reach.  Lookups never raise for an in-range key: the chain is

    1. the exact ``(detector, zone_id)`` row;
    2. the median over covered zones of that detector;
    3. the global median over all covered zones.

``.source`` on the returned value records which rung was used
(``"zone"`` / ``"detector_median"`` / ``"global_median"``), so a driver can log
or count fallbacks instead of silently assuming full coverage.

As shipped, the table covers all 6 x 121 = 726 pairs, so the fallbacks are
insurance, not routine behaviour.

Do not "improve" the fallback into a nearest-zone interpolation.  It was tested
and it is *worse*: leave-one-out over the 726 measured zones gives dy rms
0.0226 native px for nearest-zone against 0.0166 for the detector median,
because ``dy`` alternates between adjacent lattice rows, so the spatially
nearest zone frequently sits in the opposite phase.  (For ``dx``, which is
flat, nearest-zone is marginally better: 0.0042 vs 0.0053.  Not worth
special-casing.)

Why this is a per-zone table and not one scalar
-----------------------------------------------
``dx`` is nearly constant (-0.053 +/- 0.007).  ``dy`` is not: it varies mainly
*between* lattice rows -- detector-Y bands -- from about +0.01 to -0.11 native
px, nearly constant along each row, alternating row to row.  Collapsing the
table to a single global pair leaves a dy residual of rms 0.017 and up to 0.067
native px (0.41 arcsec), i.e. larger than the ~0.048 px correction itself.
Both independent peak estimators reproduce that structure (correlation ~0.96
with the symmetry fit across all 726 zones), so it is a real property of the
delivered kernels.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from astropy.table import Table

__all__ = [
    "NATIVE_PX_ARCSEC",
    "OVERSAMP",
    "TABLE_PATH",
    "CoreOffset",
    "CoreShift",
    "coverage",
    "load_core_offset_table",
    "psf_core_offset",
    "psf_core_shift",
    "psf_core_shift_arcsec",
    "psf_core_shift_batch",
    "psf_core_shift_for_kernel",
    "psf_core_shift_oversampled",
]

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "psf_core_offsets.ecsv")

OVERSAMP = 10          # oversampled px per native px

# Pair-aligned 2x2 binning (10x cube -> 5x working stamp, the package's
# ``downsample_psf_oversample2``) puts the 5x grid's origin half an INPUT pixel
# from the 10x one, so every feature sits 0.05 native px more negative in the
# working kernel than in the delivered plane. This term must be ADDED to
# ``psf_core_shift(...)`` (per axis) when correcting a 5x working stamp; it is
# already zero if you correct the 10x plane directly.
DOWNSAMPLE_GRID_SHIFT_NATIVE = 0.05
NATIVE_PX_ARCSEC = 6.15  # 1 native detector px, arcsec
N_ZONES = 121
N_DETECTORS = 6


class _Pair:
    """A named ``(dy, dx)`` pair that also unpacks as a plain 2-tuple.

    ``dy, dx = value`` works, ``value[0]`` works, and ``value.source`` carries
    the fallback provenance without ever landing in the unpacked pair.
    """

    __slots__ = ("_dx", "_dy", "source")

    def __init__(self, dy, dx, source="zone"):
        object.__setattr__(self, "_dy", float(dy))
        object.__setattr__(self, "_dx", float(dx))
        object.__setattr__(self, "source", str(source))

    def __iter__(self):
        yield self._dy
        yield self._dx

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return (self._dy, self._dx)[i]

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def __hash__(self):
        return hash(tuple(self))

    def as_array(self):
        """``np.array([dy, dx])`` -- axis order matches numpy indexing."""
        return np.array([self._dy, self._dx])


class CoreOffset(_Pair):
    """Where the PSF core IS, native px, relative to the declared fiducial.

    Negative on both axes.  This is the *measurement*, not the correction --
    do not hand it to a shift routine.  Use :func:`psf_core_shift`.
    """

    __slots__ = ()

    @property
    def dy_core(self):
        return self._dy

    @property
    def dx_core(self):
        return self._dx

    def __repr__(self):
        return (f"CoreOffset(dy_core={self._dy:+.6f}, dx_core={self._dx:+.6f}, "
                f"source={self.source!r})  # MEASURED position, not a shift")


class CoreShift(_Pair):
    """What to APPLY to the kernel, native px.  Already ``-offset``.

    Positive on both axes.  Shifting the kernel by ``(dy_apply, dx_apply)``
    moves its core onto the declared fiducial.  Do not negate this again.
    """

    __slots__ = ()

    @property
    def dy_apply(self):
        return self._dy

    @property
    def dx_apply(self):
        return self._dx

    def __repr__(self):
        return (f"CoreShift(dy_apply={self._dy:+.6f}, dx_apply={self._dx:+.6f}, "
                f"source={self.source!r})  # APPLY this; already -offset")


# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_core_offset_table(path: str | None = None) -> Table:
    """Read (and cache) the committed core-offset ECSV.

    The table is static: it is measured once by
    ``analysis/derive_psf_core_offsets.py`` from the delivered PSF planes,
    which are byte-identical for a given ``(detector, zone_id)`` across every
    cutout and pipeline version.  Never recompute it per run.
    """
    p = path or TABLE_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"PSF core-offset table not found at {p}. Regenerate it with:\n"
            "  python analysis/derive_psf_core_offsets.py --build"
        )
    return Table.read(p, format="ascii.ecsv")


@lru_cache(maxsize=4)
def _index(path: str | None = None):
    """Build the O(1) lookup structures and the fallback medians."""
    tab = load_core_offset_table(path)
    by_key = {}
    by_md5 = {}
    for row in tab:
        key = (int(row["detector"]), int(row["zone_id"]))
        by_key[key] = (float(row["dy_core"]), float(row["dx_core"]))
        md5 = str(row["md5"]) if "md5" in tab.colnames else None
        if md5:
            by_md5.setdefault(md5, (float(row["dy_core"]), float(row["dx_core"])))
    det_median = {}
    for det in sorted({int(d) for d in tab["detector"]}):
        sel = tab["detector"] == det
        det_median[det] = (
            float(np.median(tab["dy_core"][sel])),
            float(np.median(tab["dx_core"][sel])),
        )
    global_median = (
        float(np.median(tab["dy_core"])),
        float(np.median(tab["dx_core"])),
    )
    return by_key, by_md5, det_median, global_median


def _lookup(detector: int, zone_id: int, path=None):
    """-> (dy_core, dx_core, source).  Never raises for integral inputs."""
    by_key, _, det_median, global_median = _index(path)
    det, zone = int(detector), int(zone_id)
    hit = by_key.get((det, zone))
    if hit is not None:
        return hit[0], hit[1], "zone"
    hit = det_median.get(det)
    if hit is not None:
        return hit[0], hit[1], "detector_median"
    return global_median[0], global_median[1], "global_median"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def psf_core_offset(detector, zone_id, table=None, path=None) -> CoreOffset:
    """MEASURED core position in native px (negative).  See module docstring.

    This is *not* the correction.  For the shift to apply, call
    :func:`psf_core_shift`.
    """
    del table  # accepted for symmetry with psf_core_shift; index is cached
    dy, dx, src = _lookup(detector, zone_id, path)
    return CoreOffset(dy, dx, src)


def psf_core_shift(detector, zone_id, table=None, path=None) -> CoreShift:
    """Shift to **APPLY** to the delivered kernel, in native px.

    Returns ``CoreShift(dy_apply, dx_apply, source)`` with
    ``dy_apply = -dy_core`` and ``dx_apply = -dx_core`` -- i.e. the negative of
    the measured core offset, which is what puts the core on the source
    position.  Both components come back *positive* (~ +0.05).  **Do not negate
    the result.**

    ``(dy, dx)`` is numpy axis order: ``dy`` along axis 0 (rows), ``dx`` along
    axis 1 (columns), positive toward increasing index.

    Never raises for an integral ``(detector, zone_id)``; falls back to the
    detector median then the global median and says so in ``.source``.

    Parameters
    ----------
    detector : int
        ``DETECTOR`` from the cutout's primary header (1..6).
    zone_id : int
        ``zone_id`` from that cutout's ``PSF_ZONES`` row for the plane in use.
    table, path : optional
        ``table`` is accepted and ignored (the index is process-cached);
        ``path`` overrides the committed ECSV location, for tests.
    """
    del table
    dy, dx, src = _lookup(detector, zone_id, path)
    return CoreShift(-dy, -dx, src)


def psf_core_shift_arcsec(detector, zone_id, path=None) -> CoreShift:
    """:func:`psf_core_shift` converted to arcsec (1 native px = 6.15")."""
    s = psf_core_shift(detector, zone_id, path=path)
    return CoreShift(s.dy_apply * NATIVE_PX_ARCSEC,
                     s.dx_apply * NATIVE_PX_ARCSEC, s.source)


def psf_core_shift_oversampled(detector, zone_id, path=None) -> CoreShift:
    """:func:`psf_core_shift` in oversampled (delivered-grid) px."""
    s = psf_core_shift(detector, zone_id, path=path)
    return CoreShift(s.dy_apply * OVERSAMP, s.dx_apply * OVERSAMP, s.source)


def psf_core_shift_for_kernel(plane, path=None) -> CoreShift:
    """Label-free lookup: identify the kernel by its bytes, not its zone_id.

    Some retrieval bundles attach a geometrically wrong ``zone_id`` to a cutout
    (the kernel bytes themselves are always consistent with whatever label they
    carry).  If you would rather not trust the label, hand the float32 plane
    straight to this function.  Falls back to the global median with
    ``source="global_median"`` if the array is not in the table.
    """
    import hashlib

    _, by_md5, _, global_median = _index(path)
    arr = np.ascontiguousarray(np.asarray(plane, dtype=np.float32))
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    hit = by_md5.get(digest)
    if hit is None:
        return CoreShift(-global_median[0], -global_median[1], "global_median")
    return CoreShift(-hit[0], -hit[1], "md5")


def psf_core_shift_batch(detectors, zone_ids, path=None):
    """Vectorised :func:`psf_core_shift` for a driver looping over many tiles.

    Returns ``(dy_apply, dx_apply, source)`` as three arrays of shape
    ``(n,)``, same units and sign convention as :func:`psf_core_shift`.
    """
    dets = np.atleast_1d(np.asarray(detectors, dtype=int)).ravel()
    zones = np.atleast_1d(np.asarray(zone_ids, dtype=int)).ravel()
    if dets.size != zones.size:
        raise ValueError(
            f"detectors and zone_ids must match in length "
            f"({dets.size} vs {zones.size})"
        )
    dy = np.empty(dets.size)
    dx = np.empty(dets.size)
    src = np.empty(dets.size, dtype=object)
    for i, (d, z) in enumerate(zip(dets, zones)):
        s = psf_core_shift(d, z, path=path)
        dy[i], dx[i], src[i] = s.dy_apply, s.dx_apply, s.source
    return dy, dx, src


def coverage(path=None) -> dict:
    """Which ``(detector, zone_id)`` pairs are measured, and which fall back."""
    by_key, _, det_median, _ = _index(path)
    per_det = {}
    for det in range(1, N_DETECTORS + 1):
        have = sorted(z for (d, z) in by_key if d == det)
        per_det[det] = dict(
            n_zones=len(have),
            missing=sorted(set(range(1, N_ZONES + 1)) - set(have)),
            has_detector_median=det in det_median,
        )
    return dict(
        n_pairs=len(by_key),
        n_possible=N_DETECTORS * N_ZONES,
        per_detector=per_det,
    )
