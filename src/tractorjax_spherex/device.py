"""JAX device / precision setup.

JAX picks its backend and x64 mode at import time, so these must be configured
BEFORE the first `import jax`. Library callers should call :func:`setup_device`
before :func:`tractorjax_spherex.pipeline.run_photometry`; the pipeline also
calls it defensively. The CLI additionally peeks ``--device cpu`` out of argv
before importing anything JAX-touching (see :mod:`tractorjax_spherex.cli`).
"""

from __future__ import annotations

import logging
import os
import sys
import warnings

logger = logging.getLogger(__name__)

# Remembers the last successfully-applied (device, precision) so repeated calls
# in one process (e.g. photometering several fields in a notebook) don't emit a
# spurious "jax already imported" warning when nothing actually changes.
_APPLIED: tuple[str, str] | None = None


def setup_device(device: str = "auto", precision: str = "fp32",
                 mem_fraction: float | None = None,
                 preallocate: bool = False) -> None:
    """Configure the JAX backend and precision via environment variables.

    Parameters
    ----------
    device : {"auto", "gpu", "cpu"}
        ``"cpu"`` forces the JAX CPU backend (``JAX_PLATFORMS=cpu``); ``"auto"``
        and ``"gpu"`` leave backend selection to JAX (GPU if available).
    precision : {"fp32", "fp64"}
        ``"fp64"`` enables ``jax_enable_x64`` (needed for calibration-grade
        variances and for bit-level agreement with the CPU/x64 reference).
    mem_fraction : float, optional
        Sets ``XLA_PYTHON_CLIENT_MEM_FRACTION`` — the fraction of GPU memory JAX
        may use. Use a value like 0.45 when sharing a GPU with other jobs.
    preallocate : bool
        If False (default), sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so JAX
        grows its allocation on demand instead of grabbing ~75% of the card up
        front (which OOM-kills co-running jobs).

    Notes
    -----
    If ``jax`` is already imported, backend/precision env changes will not take
    effect and a warning is emitted; only call this once, early.
    """
    global _APPLIED
    if precision not in ("fp32", "fp64"):
        raise ValueError(f"precision must be 'fp32' or 'fp64', got {precision!r}")
    # No-op if we already applied the same settings this process.
    if _APPLIED == (device, precision):
        return

    jax_already = "jax" in sys.modules

    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if mem_fraction is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{float(mem_fraction):.3f}"

    if jax_already and (device == "cpu"):
        warnings.warn(
            "setup_device(device='cpu') was called after `jax` was already "
            "imported; the backend cannot be changed now. Call setup_device() "
            "(or pass --device on the CLI) before the first JAX import.",
            RuntimeWarning, stacklevel=2)
    if precision == "fp64":
        import jax
        jax.config.update("jax_enable_x64", True)
    _APPLIED = (device, precision)
