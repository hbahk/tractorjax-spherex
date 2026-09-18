"""Cross-cutout PSF kernel and Fourier-transform cache.

Every cutout of one detector ships a byte-identical PSF cube, so downsampling
its zone planes and transforming them once per cutout is pure repeat work —
~20 ms per cutout in the production driver's host profile, and it does not
shrink with the cutout: a whole-archive scan pays it on every frame.

Two things have to be cached together, and that is the whole reason this is a
class rather than two dicts:

* the **kernels**, so a cutout reuses the arrays instead of rebuilding them; and
* the engine's **PSF FFTs**, which ``tractor_jax`` keys on ``id(kernel)``.

The engine's contract is explicit that *"the caller must keep the PSF arrays
alive for the cache's lifetime (id reuse after gc would alias)"*.  A kernel that
is dropped while its transform stays cached is a latent wrong answer: a later
array allocated at the same address would hit the stale entry.  Holding both
here, and clearing both in :meth:`PSFCache.clear`, makes that impossible to get
wrong by accident.

Keying
------
On a **fingerprint of the cube**, not on the detector.  Detector alone is not
enough: on the SPHEREx production archive detectors 2, 4, 5 and 6 each ship two
distinct cubes, and detector 4 ships both of them under the *same* processing
version — so ``(detector, procver)`` would alias two different PSFs.  The
signature below mixes the detector, the cube shape, three plane sums and the
zone table, which separates them.
"""

from __future__ import annotations

import numpy as np

#: Distinct cubes to hold before dropping everything.  A field has a handful of
#: keys (one per detector x zone subset); the cap only guards a pathological mix
#: of products in one process.
MAX_CUBES = 64


def cube_signature(cutout) -> tuple:
    """Cheap identity of a cutout's PSF cube.

    Detector + PSF kind + cube shape + calibration source file + three plane
    sums + a hash of the middle plane + the zone table.  The sums and the hash
    are what make this a fingerprint rather than a label: a changed product
    with the same detector and zone layout will not collide (R7 ePSF planes all
    sum to 1.0, so the hash carries that case).
    """
    import hashlib
    cube = np.asarray(cutout["psf_cube"])
    zones = cutout["psf_zones"]
    try:
        primary = cutout["primary_header"]
        kind = cutout["psf_kind"]
    except (KeyError, AttributeError, TypeError):
        primary, kind = {}, "optical"
    mid = np.ascontiguousarray(cube[cube.shape[0] // 2])
    # Plane sums alone do not separate ePSF products (every R7 plane sums to
    # 1.0), hence the kind, the calibration source file and a byte hash of the
    # middle plane.
    return (int(cutout["detector"]), str(kind), tuple(cube.shape),
            str(primary.get("EPSFCAL", "")),
            float(cube[0].sum()), float(cube[-1].sum()), float(mid.sum()),
            hashlib.blake2b(mid.tobytes(), digest_size=16).hexdigest(),
            tuple(int(z) for z in np.asarray(zones["zone_id"])),
            tuple(int(p) for p in np.asarray(zones["plane_idx"])))


class PSFCache:
    """Kernels, core-shift tables and engine FFTs, shared across cutouts.

    One instance lives for as long as a photometry run (the backend owns it).
    Everything it hands out is a **shared object**: two cutouts of the same
    detector get the identical list, the identical arrays. That identity is
    load-bearing — it is what makes the engine's transform cache hit.
    """

    __slots__ = ("basis", "fft", "max_cubes", "shifts", "stamps")

    def __init__(self, max_cubes: int = MAX_CUBES):
        self.basis: dict[tuple, list] = {}      # signature -> list[kernel]
        self.stamps: dict[tuple, np.ndarray] = {}   # (signature, plane) -> kernel
        self.shifts: dict[tuple, np.ndarray] = {}   # (detector, zone ids) -> (K,2)
        self.fft: dict = {}                     # engine-owned, keyed on id(kernel)
        self.max_cubes = max_cubes

    def clear(self) -> None:
        """Drop everything.

        The FFTs go with the kernels, never separately: an entry keyed on the
        id of a freed array would alias whatever is allocated there next.
        """
        self.basis.clear()
        self.stamps.clear()
        self.shifts.clear()
        self.fft.clear()

    def _evict_if_full(self) -> None:
        if len(self.basis) >= self.max_cubes or len(self.stamps) >= self.max_cubes * 16:
            self.clear()

    def zone_basis(self, signature, build_fn) -> list:
        """The downsampled zone kernels for this cube — ONE list per cube.

        ``build_fn()`` is called only on a miss.  The list object itself is
        shared, not just its contents, because the engine memoizes on the
        kernels' identity.
        """
        got = self.basis.get(signature)
        if got is None:
            self._evict_if_full()
            got = build_fn()
            self.basis[signature] = got
        return got

    def stamp(self, signature, plane: int, build_fn) -> np.ndarray:
        """One downsampled kernel, for the nearest-zone (non-blended) path."""
        key = (signature, int(plane))
        got = self.stamps.get(key)
        if got is None:
            self._evict_if_full()
            got = build_fn()
            self.stamps[key] = got
        return got

    def zone_shifts(self, detector: int, zone_ids, build_fn) -> np.ndarray:
        """The ``(K, 2)`` per-zone core-shift table for this detector.

        Also identity-sensitive: the engine memoizes the native -> high-res
        conversion on this array's identity, so every cutout must get the same
        object.
        """
        key = (int(detector), tuple(int(z) for z in zone_ids))
        got = self.shifts.get(key)
        if got is None:
            got = build_fn()
            self.shifts[key] = got
        return got

    def stats(self) -> dict:
        """Entry counts — for logging and for tests that assert reuse."""
        return {"cubes": len(self.basis), "stamps": len(self.stamps),
                "shift_tables": len(self.shifts), "ffts": len(self.fft)}
