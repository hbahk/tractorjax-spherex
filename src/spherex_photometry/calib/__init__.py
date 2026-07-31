"""Static SPHEREx calibration tables shipped with the package.

``psf_core_offsets`` — the measured L2 PSF core-registration offsets (726/726
detector-zone cells), vendored from the ``spherex_gpupipe`` project where they
were derived (research note 2026-07-30-psf-centring-audit). The loader module
carries the full sign-convention documentation; read it before using either
direction of the pair.
"""
from .psf_core_offsets import (  # noqa: F401
    DOWNSAMPLE_GRID_SHIFT_NATIVE,
    psf_core_offset,
    psf_core_shift,
    psf_core_shift_batch,
)

__all__ = [
    "DOWNSAMPLE_GRID_SHIFT_NATIVE",
    "psf_core_offset",
    "psf_core_shift",
    "psf_core_shift_batch",
]
