"""Backend registry: lazily import and construct the backend for a config."""

from __future__ import annotations

from .base import Backend, CutoutRecords, FieldContext

__all__ = ["Backend", "CutoutRecords", "FieldContext", "get_backend"]

_TRACTOR_JAX_HINT = (
    "The 'jax' backend needs the tractor-jax engine, which is not installed.\n"
    "Install it first (it is not on PyPI yet):\n"
    "    pip install git+https://github.com/hbahk/tractor-jax\n"
    "For GPU add the CUDA jax build:  pip install 'spherex-photometry[gpu]'")

_TRACTOR_HINT = (
    "The 'cpu-tractor' backend needs the upstream Tractor package, which is not "
    "on PyPI. Install it from source:\n"
    "    pip install git+https://github.com/dstndstn/tractor\n"
    "See the 'CPU backend' page in the docs for build notes.")


def get_backend(config):
    """Construct the backend named by ``config.backend`` (validated, lazy import)."""
    config.validate()
    if config.backend == "jax":
        try:
            from .jax_backend import JaxBackend
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_TRACTOR_JAX_HINT) from exc
        return JaxBackend(config)
    if config.backend == "cpu-tractor":
        try:
            from .cpu_backend import CpuTractorBackend
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(_TRACTOR_HINT) from exc
        return CpuTractorBackend(config)
    raise ValueError(f"unknown backend {config.backend!r}")
