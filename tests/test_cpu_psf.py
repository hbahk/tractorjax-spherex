"""The critical PixelizedPSF flux-normalization fix (see docs/cpu_backend)."""

import numpy as np
import pytest

pytest.importorskip("tractor")

from fixtures.synth import gaussian_oversampled
from tractorjax_spherex.backends.cpu_psf import OversampledPixelizedPSF


def _psf5x():
    # 5x-oversampled 51x51 Gaussian normalized to unit flux.
    p = gaussian_oversampled(51, 5, fwhm_native=2.5)
    return (p / p.sum()).astype(np.float32)


def test_fixed_psf_conserves_flux():
    psf = OversampledPixelizedPSF(_psf5x(), sampling=0.2)
    patch = psf.getPointSourcePatch(20.0, 20.0)   # integer position
    # a unit-flux point source must produce a native model summing to ~1
    assert patch.patch.sum() == pytest.approx(1.0, abs=2e-2)


def test_unpatched_psf_is_wrong_by_oversample_squared():
    from tractor.psf import PixelizedPSF
    buggy = PixelizedPSF(_psf5x(), sampling=0.2)
    patch = buggy.getPointSourcePatch(20.0, 20.0)
    # upstream omits the 1/sampling**2 factor: sums to ~sampling**2 = 0.04
    assert patch.patch.sum() < 0.1
    fixed = OversampledPixelizedPSF(_psf5x(), sampling=0.2)
    ratio = fixed.getPointSourcePatch(20.0, 20.0).patch.sum() \
        / patch.patch.sum()
    assert ratio == pytest.approx(25.0, rel=0.05)   # 1/0.2**2


def test_subpixel_shift_conserves_flux():
    psf = OversampledPixelizedPSF(_psf5x(), sampling=0.2)
    patch = psf.getPointSourcePatch(20.3, 19.6)    # sub-pixel offset
    assert patch.patch.sum() == pytest.approx(1.0, abs=3e-2)
