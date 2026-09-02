"""fitsio-based fast path for reading cutout MEFs.

``astropy.io.fits`` charges a substantial per-cutout cost that has nothing to
do with the pixels: header parsing and card verification of the ten HDUs,
``HDUList.index_of`` walks, the copies, and ``astropy.wcs`` parsing a ~200-card
header through wcslib.  ``cfitsio`` reads the same arrays in a few
milliseconds.  Measured on the reference field (24 real cutouts, warm page
cache): **20.5 -> 5.1 ms per cutout, 4.9 ms with the PSF-cube cache (4.2x)**.

.. warning::

   **That speedup is regime-dependent, and it inverts on cold reads of large
   files.**  ``fitsio.FITS()`` indexes every HDU when it opens the file, while
   ``astropy.io.fits.open(lazy_load_hdus=True)`` walks only as far as the HDU
   asked for.  When each header touch is a cold storage request, that costs one
   request per HDU before a single pixel is read.

   Measured on a GPFS archive of 71.6 MB full-frame L2 MEFs (7 HDUs), cold —
   each file touched once::

       reading the IMAGE plane   astropy 45.6 ms | fitsio 215.6 ms | raw pread 38.0 ms
       same files, warm          astropy  9.8 ms | fitsio   7.2 ms | raw pread  2.5 ms

   fitsio's header-only, plane and stamp timings were all ~200-215 ms there,
   i.e. entirely the open, which is the signature of eager HDU indexing.

   So: this module is the right default for the **cutout** MEFs it was written
   for — small files, usually warm, read many times per field.  A reader for
   large cold frames (a whole-archive scan) should not assume it wins; measure,
   and consider a positional read at a known data offset instead.

Nothing here changes a number.  Every function is a re-expression of what
:func:`spherex_photometry.io.cutouts.read_cutout` does with astropy, and
``tests/test_io_fast.py`` asserts array-by-array, keyword-by-keyword and
WCS-projection equality against the astropy reader on real fixtures.  The
fast path is used only when ``fitsio`` is importable; otherwise the astropy
reader runs unchanged.

Three sub-paths, each independently switchable for bisecting a suspected
difference:

``FAST_HEADERS``
    Hand back plain keyword -> value mappings (:class:`HeaderDict`) built from
    the fitsio records instead of :class:`astropy.io.fits.Header` objects.
    Callers only ever ``.get()`` / ``[]`` these (``CRPIX*A``, ``OVERSAMP``,
    ``DETECTOR``, ``OBSID``); building two Header objects from 200 card
    strings and paying their slow item access costs ~6 ms per cutout.
    :meth:`HeaderDict.to_astropy` recovers a real Header on demand.
``FAST_WCS``
    Build the celestial WCS from the header *values* (CTYPE/CRVAL/CRPIX/CUNIT,
    CD or PC+CDELT, LONPOLE/LATPOLE/RADESYS/EQUINOX and the SIP A/B/AP/BP
    polynomials) instead of letting wcslib parse the whole header: ~9 ms -> 1.4 ms.
``FAST_PSF_CUBE``
    The PSF cube is byte-identical for every cutout sharing a detector and a
    zone subset.  Re-read it only when the cheap identity below misses.

Backported from the production tiled driver
(``proj-spherex-gpupipe/analysis/fast_cutout_io.py``, 2026-08-22), where the
same code is validated against the astropy reader on the campaign fields.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

FAST_HEADERS = True
FAST_WCS = True
FAST_PSF_CUBE = True

#: Largest number of distinct PSF cubes to hold.  A field has a handful of
#: keys (one per detector x zone subset); the cap only guards a pathological
#: mix of products in one process.
PSF_CUBE_CACHE_MAX = 64

_PSF_CUBE_CACHE: dict[tuple, np.ndarray] = {}


def have_fitsio() -> bool:
    """True when the fitsio fast path can be used in this environment."""
    try:
        import fitsio  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken cfitsio build must fall back too
        return False
    return True


def clear_caches() -> None:
    """Drop the PSF-cube cache (tests, or a change of product mid-process)."""
    _PSF_CUBE_CACHE.clear()


# --------------------------------------------------------------------------- #
# Headers
# --------------------------------------------------------------------------- #

class HeaderDict(dict):
    """Keyword -> value mapping with the ``Header`` access callers actually use.

    A plain ``dict`` subclass, so ``hdr["OBSID"]`` / ``hdr.get("CRPIX1A", 1)``
    / ``"CWAVE" in hdr`` all behave as with an :class:`astropy.io.fits.Header`.
    ``COMMENT`` / ``HISTORY`` / ``CONTINUE`` cards are dropped.
    """

    __slots__ = ()

    def to_astropy(self) -> fits.Header:
        """A real :class:`astropy.io.fits.Header` with these keyword values."""
        return fits.Header([(k, v) for k, v in self.items()
                            if k not in ("COMMENT", "HISTORY", "")])


def _dict_from_fitsio(fh) -> HeaderDict:
    out = HeaderDict()
    for r in fh.records():
        name = r.get("name", "")
        if not name or name in ("COMMENT", "HISTORY", "CONTINUE"):
            continue
        out[name] = r.get("value")
    return out


def _astropy_header_from_fitsio(fh) -> fits.Header:
    """astropy Header from a fitsio FITSHDR (card strings, no file re-parse)."""
    cards = "".join(r["card_string"].ljust(80) for r in fh.records())
    return fits.Header.fromstring(cards)


# --------------------------------------------------------------------------- #
# WCS
# --------------------------------------------------------------------------- #

def wcs_from_header_values(h) -> WCS:
    """Celestial 2-axis WCS rebuilt from header values (see ``FAST_WCS``).

    ``h`` is any keyword -> value mapping (a :class:`HeaderDict` or an
    :class:`astropy.io.fits.Header`).  SIP is carried when ``A_ORDER`` is
    present, which it always is for SPHEREx L2 (``CTYPE = RA---TAN-SIP``).
    """
    from astropy.wcs import Sip

    w = WCS(naxis=2)
    w.wcs.ctype = [h["CTYPE1"], h["CTYPE2"]]
    w.wcs.crval = [h["CRVAL1"], h["CRVAL2"]]
    w.wcs.crpix = [h["CRPIX1"], h["CRPIX2"]]
    w.wcs.cunit = [h.get("CUNIT1", "deg"), h.get("CUNIT2", "deg")]
    if "CD1_1" in h:
        w.wcs.cd = [[h["CD1_1"], h.get("CD1_2", 0.0)],
                    [h.get("CD2_1", 0.0), h["CD2_2"]]]
    else:
        w.wcs.pc = [[h.get("PC1_1", 1.0), h.get("PC1_2", 0.0)],
                    [h.get("PC2_1", 0.0), h.get("PC2_2", 1.0)]]
        w.wcs.cdelt = [h.get("CDELT1", 1.0), h.get("CDELT2", 1.0)]
    if "LONPOLE" in h:
        w.wcs.lonpole = h["LONPOLE"]
    if "LATPOLE" in h:
        w.wcs.latpole = h["LATPOLE"]
    if "RADESYS" in h:
        w.wcs.radesys = h["RADESYS"]
    if "EQUINOX" in h:
        w.wcs.equinox = h["EQUINOX"]

    def coef(prefix, order):
        m = np.zeros((order + 1, order + 1))
        for i in range(order + 1):
            for j in range(order + 1 - i):
                m[i, j] = h.get(f"{prefix}_{i}_{j}", 0.0)
        return m

    if "A_ORDER" in h:
        a = coef("A", h["A_ORDER"])
        b = coef("B", h["B_ORDER"])
        ap = coef("AP", h["AP_ORDER"]) if "AP_ORDER" in h else None
        bp = coef("BP", h["BP_ORDER"]) if "BP_ORDER" in h else None
        w.sip = Sip(a, b, ap, bp, [h["CRPIX1"], h["CRPIX2"]])
    w.wcs.set()
    return w


# --------------------------------------------------------------------------- #
# PSF cube cache
# --------------------------------------------------------------------------- #

def _psf_cube_cached(hdu, primary, zones_rec) -> np.ndarray:
    """The cutout's PSF cube, re-read only when the cheap identity misses.

    The identity is (detector, cube dims, zone table) plus a fingerprint of the
    middle plane -- read anyway, ~40 KB -- so a changed product cannot be
    served from the cache.
    """
    if not FAST_PSF_CUBE:
        return np.asarray(hdu.read(), dtype=np.float64)
    dims = tuple(int(d) for d in hdu.get_dims())
    mid = dims[0] // 2 if len(dims) == 3 else 0
    plane = np.asarray(hdu[mid:mid + 1, :, :] if len(dims) == 3 else hdu.read(),
                       dtype=np.float64)
    key = (int(primary.get("DETECTOR", -1)), dims,
           tuple(int(z) for z in np.asarray(zones_rec["zone_id"])),
           tuple(int(p) for p in np.asarray(zones_rec["plane_idx"])),
           float(plane.sum()), float(plane.max()),
           float(plane.ravel()[plane.size // 3]),
           float(plane.ravel()[2 * plane.size // 3]))
    cube = _PSF_CUBE_CACHE.get(key)
    if cube is None:
        cube = np.asarray(hdu.read(), dtype=np.float64)
        if len(_PSF_CUBE_CACHE) >= PSF_CUBE_CACHE_MAX:
            _PSF_CUBE_CACHE.clear()
        _PSF_CUBE_CACHE[key] = cube
    return cube


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #

def read_cutout_fields(path) -> dict:
    """Read one cutout MEF with fitsio; return the :class:`Cutout` field dict.

    Keys, dtypes and values match
    :func:`spherex_photometry.io.cutouts.read_cutout` exactly, so the caller
    can hand the result straight to ``Cutout(**fields)``.
    """
    import fitsio

    with fitsio.FITS(str(path)) as f:
        names = [h.get_extname() for h in f]
        _mk = _dict_from_fitsio if FAST_HEADERS else _astropy_header_from_fitsio
        primary = _mk(f[0].read_header())
        img_hdr = _mk(f["IMAGE"].read_header())

        def _read(name, dtype=None):
            if name not in names:
                return None
            arr = f[name].read()
            if arr is None:
                return None
            return np.asarray(arr, dtype=dtype) if dtype is not None else np.asarray(arr)

        img = _read("IMAGE", np.float64)
        flg_native = _read("FLAGS")
        var = _read("VARIANCE", np.float64)
        zodi = _read("ZODI", np.float64)
        zones_rec = f["PSF_ZONES"].read()
        psf_cube = _psf_cube_cached(f["PSF"], primary, zones_rec)
        cwave = _read("CWAVE", np.float64)
        cband = _read("CBAND", np.float64)
        sapm = _read("SAPM", np.float64)

    # The astropy reader keeps FLAGS in the file's (big-endian) int32 dtype.
    # Only the values matter downstream ((flg & MASKBITS) != 0), but keeping the
    # dtype identical makes dtype-sensitive comparisons hold too.
    flg = np.asarray(flg_native).astype(">i4", copy=False)
    psf_zones = Table(zones_rec)

    cwave_map = None
    cwave_center = None
    if cwave is not None and cwave.ndim == 2 and cwave.size > 0:
        cwave_map = cwave
        cy, cx = cwave.shape[0] // 2, cwave.shape[1] // 2
        cwave_center = float(cwave[cy, cx])
    cband_map = None
    if cband is not None and cband.ndim == 2 and cband.size > 0:
        cband_map = cband
    if sapm is not None and not (sapm.ndim == 2 and sapm.shape == img.shape):
        sapm = None

    wcs = wcs_from_header_values(img_hdr) if FAST_WCS else WCS(
        img_hdr if isinstance(img_hdr, fits.Header) else img_hdr.to_astropy()
    ).celestial

    return {
        "image": img, "flags": flg, "variance": var, "zodi": zodi,
        "psf_cube": psf_cube, "psf_zones": psf_zones,
        "wcs": wcs, "image_header": img_hdr, "primary_header": primary,
        "crpix1a": float(img_hdr.get("CRPIX1A", 1)),
        "crpix2a": float(img_hdr.get("CRPIX2A", 1)),
        "psf_oversamp": int(primary.get("OVERSAMP", 10)),
        "detector": int(primary.get("DETECTOR", img_hdr.get("DETECTOR", -1))),
        "cwave_center": cwave_center,
        "cwave_map": cwave_map,
        "cband_map": cband_map,
        "sapm": sapm,
    }


def read_image_geometry(path) -> tuple[int, int, WCS]:
    """``(H, W, celestial WCS)`` from the IMAGE header alone.

    The occupancy pre-scan needs only this; going through fitsio avoids
    opening and verifying every HDU for a header-only read.
    """
    import fitsio

    with fitsio.FITS(str(path)) as f:
        hdr = _dict_from_fitsio(f["IMAGE"].read_header())
    return (int(hdr["NAXIS2"]), int(hdr["NAXIS1"]),
            wcs_from_header_values(hdr) if FAST_WCS
            else WCS(hdr.to_astropy()).celestial)
