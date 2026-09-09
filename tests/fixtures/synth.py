"""Test-fixture shim: the synthetic-data makers are public package API now.

Kept so tests read ``from fixtures.synth import ...``; the implementation lives
in :mod:`tractorjax_spherex.simulate` (usable by end users for offline demos).
"""

from tractorjax_spherex.simulate import (  # noqa: F401
    IMG_SCALE,
    OMEGA_SR,
    PIXSCALE,
    gaussian_oversampled,
    make_synth_catalog,
    make_synth_cutout,
    make_synth_field,
)
