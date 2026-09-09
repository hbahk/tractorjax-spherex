"""Physical constants, unit factors, and FLAGS bit definitions for SPHEREx L2.

Lifted from the production driver (proj-spherex-gpupipe
analysis/test_optimizer_spherex_batch_tiled.py, L153-232). Tunable pipeline
defaults (tiling, background, solver) live in :mod:`tractorjax_spherex.config`;
this module holds only values that are fixed by the instrument / data format.
"""

from __future__ import annotations

import astropy.units as u

# SPHEREx native detector pixel scale.
SPHEREX_PIXSCALE = 6.15  # arcsec / native pixel
PIXAREA_CONST_SR = ((SPHEREX_PIXSCALE * u.arcsec) ** 2).to_value(u.sr)
ARCSEC2_TO_SR = (1.0 * u.arcsec ** 2).to_value(u.sr)

# Internal flux unit scaling: the L2 IMAGE is MJy/sr; multiplying by the
# per-pixel solid angle (sr) gives MJy/pixel, and IMG_SCALE converts that to
# mJy/pixel for numerical conditioning of the linear solve. Reported fluxes are
# in mJy.
IMG_SCALE = 1.0e9  # MJy -> mJy

# Bit definitions from the L2 FLAGS extension header (matches the production
# driver and the SPHEREx L2 documentation).
FLAG_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "PHANTOM": 4,
    "REFERENCE": 5, "NONFUNC": 6, "DICHROIC": 7, "MISSING_DATA": 9,
    "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "SOURCE": 21,
}

# Flags that mark a pixel unusable for photometry (zeroed inverse variance).
MASK_FLAGS = ["SUR_ERROR", "PHANMISS", "NONFUNC", "MISSING_DATA",
              "HOT", "COLD", "PERSIST", "OUTLIER"]
MASKBITS = 0
for _name in MASK_FLAGS:
    MASKBITS |= (1 << FLAG_BITS[_name])

# Pixels flagged as belonging to a detected source (used only to mask the
# background fit, never the photometry).
SOURCE_BIT = 1 << FLAG_BITS["SOURCE"]

# Fixed batch-shape caps (F2 policy) — the A2537 full-depth field maxima with
# headroom. Padding every tile's flux vector to a fixed width lets the jitted
# solver compile once per run instead of once per distinct shape. Applied
# automatically only when fitting the full catalog (see PhotometryConfig).
MAX_PS_CAP = 112
MAX_GAL_CAP = 352
MAX_MOG_K_CAP = 9
