"""Read one SPHEREx L2 cutout MEF (as written by ``spherex_retrieval.retrieve``).

Lifted from the production driver (``read_cutout`` / ``discover_cutouts`` /
``sample_map_bilinear*`` / ``cutout_pixel_area_sr``). The parsed arrays are
returned as a :class:`Cutout` — a dataclass that also supports ``cutout["key"]``
and ``cutout.get("key")`` so downstream lifted code reads unchanged.

MEF layout::

    PRIMARY  IMAGE  FLAGS  VARIANCE  ZODI  PSF  PSF_ZONES  [CWAVE] [CBAND] [SAPM]

Two readers produce the identical :class:`Cutout`: the astropy one below, and
the ``fitsio`` fast path in :mod:`tractorjax_spherex.io.fast` (~4x faster per
cutout).  :func:`read_cutout` picks the fast one when ``fitsio`` is installed;
set ``fast=False`` (or the module flag :data:`FAST_IO`) to force astropy.
"""

from __future__ import annotations

import os

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ..constants import ARCSEC2_TO_SR

CUTOUT_RE = re.compile(r"cutout_(\d{4})_(.+)_D(\d)\.fits$")

#: Default reader. ``"auto"`` uses fitsio when importable, else astropy;
#: ``True`` requires fitsio (raises if missing); ``False`` forces astropy.
FAST_IO: bool | str = "auto"


@dataclass
class Cutout:
    """Parsed arrays and metadata for one SPHEREx cutout.

    Supports both attribute access (``cutout.image``) and mapping access
    (``cutout["image"]`` / ``cutout.get("cwave_map")``).
    """

    image: np.ndarray
    flags: np.ndarray
    variance: np.ndarray
    zodi: np.ndarray
    psf_cube: np.ndarray
    psf_zones: Table
    wcs: WCS
    # An astropy Header from the astropy reader, a HeaderDict from the fitsio
    # one; both support ``[]`` / ``.get()`` / ``in``, which is all callers use.
    image_header: fits.Header | Mapping
    primary_header: fits.Header | Mapping
    crpix1a: float
    crpix2a: float
    psf_oversamp: int
    detector: int
    cwave_center: float | None
    cwave_map: np.ndarray | None
    cband_map: np.ndarray | None
    sapm: np.ndarray | None
    # "optical": the QR2 10x cube, integrate over native pixels when rendering;
    # "effective": the R7 ePSF (5x, pixel response included), point-sample it.
    # From the bundle's PSFKIND keyword; bundles written before it default to
    # optical, which is what they hold.
    psf_kind: str = "optical"

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default=None):
        return getattr(self, key, default)


def discover_cutouts(cutouts_dir: str | Path) -> list[tuple[int, Path]]:
    """Return ``(cutout_index, path)`` pairs sorted by cutout index."""
    cutouts_dir = Path(cutouts_dir)
    pairs = []
    for p in cutouts_dir.glob("cutout_*.fits"):
        m = CUTOUT_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort()
    return pairs


def filter_ok(pairs: list[tuple[int, Path]],
              summary_path: str | Path) -> list[tuple[int, Path]]:
    """Keep only cutouts marked ``status == "ok"`` in ``summary.ecsv``.

    If the summary file is absent, all pairs are returned unchanged.
    """
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return pairs
    summary = Table.read(summary_path)
    ok = {int(i) for i in summary[summary["status"] == "ok"]["cutout_index"]}
    return [(idx, p) for idx, p in pairs if idx in ok]


def read_cutout(path: str | Path, *, fast: bool | str | None = None) -> Cutout:
    """Open one cutout MEF and return a :class:`Cutout`.

    ``path`` may also be the MEF's bytes or a binary file object (a bundle
    handed on in memory); those are read with astropy, which returns the same
    :class:`Cutout` as the fitsio reader.

    Present-but-empty CWAVE/CBAND/SAPM HDUs (shape ``(0,)``) are guarded: a
    missing wavelength map yields ``cwave_center=None`` / ``cwave_map=None`` (the
    source is still photometered, just labelled NaN wavelength), and a missing
    SAPM falls back to the WCS pixel area in :func:`cutout_pixel_area_sr`.

    Parameters
    ----------
    fast : bool or ``"auto"``, optional
        Reader selection, overriding the module default :data:`FAST_IO`.
        ``"auto"`` uses the fitsio fast path when importable; ``True`` requires
        it; ``False`` forces astropy.  Both readers return the identical
        :class:`Cutout` (asserted in ``tests/test_io_fast.py``).
    """
    if isinstance(path, (bytes, bytearray, memoryview)):
        # a bundle handed on in memory (spherex_retrieval.bundle.bundle_bytes)
        import io
        return _read_cutout_astropy(io.BytesIO(bytes(path)))
    if hasattr(path, "read"):
        return _read_cutout_astropy(path)
    if _use_fast(FAST_IO if fast is None else fast):
        from .fast import read_cutout_fields
        return Cutout(**read_cutout_fields(path))
    return _read_cutout_astropy(path)


def _use_fast(fast: bool | str) -> bool:
    from .fast import have_fitsio

    if fast == "auto":
        return have_fitsio()
    if fast:
        if not have_fitsio():
            raise ImportError(
                "fast=True requires fitsio (`pip install fitsio`, or "
                "`conda install -c conda-forge fitsio`); pass fast=False to "
                "use the astropy reader")
        return True
    return False


def _read_cutout_astropy(path: str | Path) -> Cutout:
    """Reference reader: astropy only. See :func:`read_cutout`."""
    if isinstance(path, (str, os.PathLike)):
        path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header.copy()
        img = np.array(hdul["IMAGE"].data, dtype=np.float64, copy=True)
        flg = np.array(hdul["FLAGS"].data, copy=True)
        var = np.array(hdul["VARIANCE"].data, dtype=np.float64, copy=True)
        zodi = np.array(hdul["ZODI"].data, dtype=np.float64, copy=True)
        psf_cube = np.array(hdul["PSF"].data, dtype=np.float64, copy=True)
        psf_zones = Table(hdul["PSF_ZONES"].data)
        img_hdr = hdul["IMAGE"].header.copy()

        cwave_map = None
        cwave_center = None
        if "CWAVE" in hdul:
            _cwave = np.asarray(hdul["CWAVE"].data, dtype=np.float64)
            if _cwave.ndim == 2 and _cwave.size > 0:
                cwave_map = _cwave
                cy, cx = _cwave.shape[0] // 2, _cwave.shape[1] // 2
                cwave_center = float(_cwave[cy, cx])
        cband_map = None
        if "CBAND" in hdul:
            _cband = np.asarray(hdul["CBAND"].data, dtype=np.float64)
            if _cband.ndim == 2 and _cband.size > 0:
                cband_map = _cband

        sapm = None
        if "SAPM" in hdul:
            _sapm = np.asarray(hdul["SAPM"].data, dtype=np.float64)
            if _sapm.ndim == 2 and _sapm.shape == img.shape:
                sapm = _sapm

    wcs = WCS(img_hdr).celestial
    crpix1a = float(img_hdr.get("CRPIX1A", 1))
    crpix2a = float(img_hdr.get("CRPIX2A", 1))
    psf_oversamp = int(primary.get("OVERSAMP", 10))
    detector = int(primary.get("DETECTOR", img_hdr.get("DETECTOR", -1)))
    return Cutout(
        image=img, flags=flg, variance=var, zodi=zodi,
        psf_cube=psf_cube, psf_zones=psf_zones,
        wcs=wcs, image_header=img_hdr, primary_header=primary,
        crpix1a=crpix1a, crpix2a=crpix2a,
        psf_oversamp=psf_oversamp, detector=detector,
        cwave_center=cwave_center, cwave_map=cwave_map,
        cband_map=cband_map, sapm=sapm,
        psf_kind=psf_kind_from_header(primary),
    )


def psf_kind_from_header(primary) -> str:
    """``"effective"`` for a bundle whose PRIMARY says ``PSFKIND = 'EPSF'``, else
    ``"optical"`` (the QR2 cube; bundles written before the keyword existed)."""
    kind = str(primary.get("PSFKIND", "OPTICAL") or "OPTICAL").strip().upper()
    if kind in ("EPSF", "EFFECTIVE"):
        return "effective"
    if kind in ("OPTICAL", "PSF", ""):
        return "optical"
    raise ValueError(f"unknown PSFKIND {kind!r} in the cutout primary header")


def sample_map_bilinear(arr, x, y) -> float:
    """Bilinearly sample a 2-D map at 0-based pixel (x=col, y=row), clamped.

    Matches ``spherex_retrieval.wavelength._bilinear``. Returns NaN if ``arr``
    is None/empty.
    """
    if arr is None or getattr(arr, "ndim", 0) != 2 or arr.size == 0:
        return float("nan")
    ny, nx = arr.shape
    x = min(max(x, 0.0), nx - 1.0)
    y = min(max(y, 0.0), ny - 1.0)
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    return float(arr[y0, x0] * (1 - fx) * (1 - fy)
                 + arr[y0, x1] * fx * (1 - fy)
                 + arr[y1, x0] * (1 - fx) * fy
                 + arr[y1, x1] * fx * fy)


def sample_map_bilinear_vec(arr, x, y):
    """Vectorized :func:`sample_map_bilinear` (bit-identical per element)."""
    x = np.asarray(x, dtype=np.float64)
    if arr is None or getattr(arr, "ndim", 0) != 2 or arr.size == 0:
        return np.full(x.shape, np.nan)
    y = np.asarray(y, dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64)
    ny, nx = arr.shape
    x = np.clip(x, 0.0, nx - 1.0)
    y = np.clip(y, 0.0, ny - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    fx, fy = x - x0, y - y0
    return (arr[y0, x0] * (1 - fx) * (1 - fy)
            + arr[y0, x1] * fx * (1 - fy)
            + arr[y1, x0] * (1 - fx) * fy
            + arr[y1, x1] * fx * fy)


def cutout_pixel_area_sr(cutout: Cutout) -> np.ndarray:
    """Per-pixel solid angle (sr), preferring the SAPM HDU when present.

    SAPM is the standalone Solid Angle Pixel Map calibration product (arcsec^2),
    which already absorbs SIP distortion. Falls back to the WCS projected pixel
    area at the tangent point.
    """
    if cutout.get("sapm") is not None:
        return cutout["sapm"].astype(np.float64, copy=False) * ARCSEC2_TO_SR
    return np.full(cutout["image"].shape,
                   cutout["wcs"].proj_plane_pixel_area().to_value(u.sr),
                   dtype=np.float64)
