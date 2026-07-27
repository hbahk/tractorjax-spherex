"""Backend protocol and the per-run field context shared across cutouts.

A backend implements three stages so the pipeline can prefetch the CPU ``build``
of cutout N+1 while the ``solve`` of cutout N runs:

    build(cutout, ctx) -> inputs        # pure CPU (safe in a worker thread)
    solve(inputs)      -> (fluxes, variances)
    extract(inputs, fluxes, variances) -> (records, cwave_center)

``records`` is a tuple of arrays ``(catalog_index, flux, flux_err, lambda,
band)`` — one entry per catalog source that lands inside the cutout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table


@dataclass
class FieldContext:
    """Per-run, per-field state built once by the pipeline and reused per cutout."""

    catalog: Table
    sco_all: SkyCoord
    main_idx: int
    protect_ci: set[int] | None = None
    prior_ctx: dict | None = None
    profile_lookup_fn: Any = None
    # Densest-tile occupancy measured over the field before the first solve;
    # set only when a cap is "auto" (see spherex_photometry.occupancy).
    occupancy: Any = None


# records = (ci, flux, flux_err, lambda, band); each a 1-D array of equal length.
CutoutRecords = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


class Backend(Protocol):
    """Structural type for a photometry backend."""

    name: str

    def build(self, cutout, ctx: FieldContext) -> Any: ...

    def solve(self, inputs: Any) -> tuple[np.ndarray, np.ndarray]: ...

    def extract(self, inputs: Any, fluxes: np.ndarray,
                variances: np.ndarray) -> tuple[CutoutRecords, float | None]: ...
