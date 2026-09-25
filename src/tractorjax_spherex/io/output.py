"""Photometry output table: schema, parquet writer/reader, resume-merge.

Each row is one spectrophotometric point: one catalog source measured on one
cutout (one SPHEREx visit / spectral channel). Fluxes are in mJy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

from ..version import __version__

SCHEMA_VERSION = 1

# (name, dtype) in output order.
COLUMNS = (
    ("cutout_index", "i8"),
    ("obs_id", "U32"),
    ("detector", "i4"),
    ("id", "i8"),
    ("ra", "f8"),
    ("dec", "f8"),
    ("central_wavelength", "f8"),   # micron
    ("bandwidth", "f8"),            # micron
    ("flux", "f8"),                 # mJy
    ("flux_err", "f8"),             # mJy
)
COLUMN_NAMES = tuple(name for name, _ in COLUMNS)

# Appended after COLUMNS when the per-visit diagnostics are on (the default on
# the jax backend; see tractorjax_spherex.quality).
QUALITY_COLUMNS = (
    ("fit_chi2", "f4"),       # template-weighted normalized squared residual
    ("mask_frac", "f4"),      # fraction of the source's template on masked pixels
    ("quality_flag", "i2"),   # bitmask, constants.QUALITY_BITS
)
QUALITY_COLUMN_NAMES = tuple(name for name, _ in QUALITY_COLUMNS)


def empty_table() -> Table:
    """An empty output Table with the correct columns and dtypes."""
    return Table(
        data=[np.zeros(0, dtype=dt) for _, dt in COLUMNS],
        names=COLUMN_NAMES,
    )


def make_table(columns: dict) -> Table:
    """Build an output Table from a dict of column-name -> array."""
    return Table(
        data=[np.asarray(columns[name]).astype(dt, copy=False)
              for name, dt in COLUMNS],
        names=COLUMN_NAMES,
    )


def write_photometry(table: Table, path, config=None, extra_meta=None) -> None:
    """Write the photometry Table to parquet with reproducibility metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    table.meta["tractorjax_spherex.schema_version"] = SCHEMA_VERSION
    table.meta["tractorjax_spherex.version"] = __version__
    if config is not None:
        table.meta["tractorjax_spherex.config"] = config.to_json()
        table.meta["tractorjax_spherex.solver_spec"] = str(config.solver_spec())
    if extra_meta:
        table.meta.update(extra_meta)
    table.write(str(path), overwrite=True)


def read_photometry(path):
    """Read a photometry parquet; returns ``(Table, meta_dict)``."""
    tab = Table.read(path)
    return tab, dict(tab.meta)


def existing_cutout_indices(path) -> set[int]:
    """Cutout indices already present in an output file (empty if absent)."""
    path = Path(path)
    if not path.exists():
        return set()
    tab = Table.read(path)
    if "cutout_index" not in tab.colnames or len(tab) == 0:
        return set()
    return {int(i) for i in tab["cutout_index"]}


def append_or_merge(path, table: Table, config=None) -> Table:
    """Append ``table`` to the existing parquet (for ``resume``), return merged.

    ``quality_flag`` is recomputed on the merged rows: its BAD_FIT bit compares
    each visit with its source's median over the whole product."""
    from ..quality import add_quality_flags

    path = Path(path)
    if path.exists():
        prev = Table.read(path)
        merged = vstack([prev, table]) if len(prev) else table
    else:
        merged = table
    rel_max = getattr(config, "visit_chi2_rel_max", 10.0) if config is not None else 10.0
    add_quality_flags(merged, rel_max)
    write_photometry(merged, path, config=config)
    return merged
