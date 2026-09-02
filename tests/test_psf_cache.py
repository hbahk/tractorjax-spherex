"""Sharing PSF kernels across cutouts must not change a flux — and must be safe.

The engine keys its Fourier-transform cache on ``id(kernel)``, so this cache
trades a real correctness hazard for the speed: a kernel dropped while its
transform stays cached lets a later array reuse the address and hit a stale
entry.  These tests pin both halves — that the sharing actually happens (same
object, not just equal values), and that kernels and transforms are held and
released together.
"""

import numpy as np
import pytest

from spherex_photometry.backends import jax_backend as JB
from spherex_photometry.config import PhotometryConfig
from spherex_photometry.io.cutouts import read_cutout
from spherex_photometry.pipeline import run_photometry
from spherex_photometry.prepare import zone_psf_basis
from spherex_photometry.psf_cache import PSFCache, cube_signature


def _cutouts(synth_field):
    return [read_cutout(p) for p in
            sorted(synth_field["cutouts_dir"].glob("cutout_*.fits"))]


def test_same_cube_gives_the_same_list_object(synth_field):
    cache = PSFCache()
    cuts = _cutouts(synth_field)
    basis0, _ = zone_psf_basis(cuts[0], cache=cache)
    for c in cuts[1:]:
        basis, _ = zone_psf_basis(c, cache=cache)
        if cube_signature(c) == cube_signature(cuts[0]):
            assert basis is basis0, "same cube rebuilt instead of reused"
            for a, b in zip(basis, basis0):
                assert a is b
    assert cache.stats()["cubes"] >= 1


def test_no_cache_rebuilds(synth_field):
    cuts = _cutouts(synth_field)
    b1, _ = zone_psf_basis(cuts[0])
    b2, _ = zone_psf_basis(cuts[0])
    assert b1 is not b2
    for a, b in zip(b1, b2):
        assert np.array_equal(a, b)      # same values, different objects


def test_signature_separates_different_cubes(synth_field):
    """Detector alone is not a safe key; the fingerprint must notice a change."""
    c = _cutouts(synth_field)[0]
    sig = cube_signature(c)

    class _Bumped:
        def __init__(self, base):
            self._b = base
            self.psf_cube = np.array(base.psf_cube, copy=True)
            self.psf_cube[0] += 1.0
        def __getitem__(self, k):
            return self.psf_cube if k == "psf_cube" else self._b[k]

    assert cube_signature(_Bumped(c)) != sig


def test_cache_clear_drops_kernels_and_ffts_together(synth_field):
    """The id-aliasing hazard: an FFT must never outlive its kernel."""
    cache = PSFCache()
    zone_psf_basis(_cutouts(synth_field)[0], cache=cache)
    cache.fft[("fake", 1)] = "transform"
    assert cache.stats()["cubes"] == 1 and cache.stats()["ffts"] == 1
    cache.clear()
    s = cache.stats()
    assert s == {"cubes": 0, "stamps": 0, "shift_tables": 0, "ffts": 0}


def test_eviction_clears_everything(synth_field):
    cache = PSFCache(max_cubes=1)
    cache.fft[("fake", 1)] = "transform"
    c = _cutouts(synth_field)[0]
    zone_psf_basis(c, cache=cache)          # fills the single slot

    class _Other:
        def __init__(self, base):
            self._b = base
            self.psf_cube = np.array(base.psf_cube, copy=True) + 3.0
        def __getitem__(self, k):
            return self.psf_cube if k == "psf_cube" else self._b[k]

    zone_psf_basis(_Other(c), cache=cache)  # forces the evict-then-store path
    assert cache.stats()["ffts"] == 0, "FFTs survived an eviction of their kernels"


def _run(synth_field, monkeypatch, enabled):
    monkeypatch.setattr(JB, "PSF_CACHE_ACROSS_CUTOUTS", enabled)
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           solver="linear")
    out = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         cfg, progress=False)
    monkeypatch.undo()
    return out


@pytest.mark.parametrize("col", ["flux", "flux_err", "central_wavelength"])
def test_photometry_identical_with_and_without_cache(synth_field, monkeypatch,
                                                     col):
    on = _run(synth_field, monkeypatch, True)
    off = _run(synth_field, monkeypatch, False)
    assert len(on) == len(off) > 0
    assert np.array_equal(np.asarray(on[col]), np.asarray(off[col]),
                          equal_nan=True), f"{col} differs"


def test_backend_holds_a_cache_when_enabled(monkeypatch):
    cfg = PhotometryConfig(backend="jax", device="cpu")
    monkeypatch.setattr(JB, "PSF_CACHE_ACROSS_CUTOUTS", True)
    assert isinstance(JB.JaxBackend(cfg)._psf_cache, PSFCache)
    monkeypatch.setattr(JB, "PSF_CACHE_ACROSS_CUTOUTS", False)
    assert JB.JaxBackend(cfg)._psf_cache is None
