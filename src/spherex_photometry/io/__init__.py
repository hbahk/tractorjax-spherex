"""I/O subpackage: cutout MEFs, reference catalogs, photometry output."""

from .catalogs import (
    apply_depth_cut,
    find_nearest_source,
    load_catalog,
    normalize_catalog,
    protected_indices,
    zmag_from_flux_z,
)
from .cutouts import (
    FAST_IO,
    Cutout,
    cutout_pixel_area_sr,
    discover_cutouts,
    filter_ok,
    read_cutout,
    sample_map_bilinear,
    sample_map_bilinear_vec,
)
from .fast import HeaderDict, have_fitsio
from .output import (
    COLUMN_NAMES,
    SCHEMA_VERSION,
    empty_table,
    make_table,
    read_photometry,
    write_photometry,
)

__all__ = [
    "COLUMN_NAMES",
    "FAST_IO",
    "SCHEMA_VERSION",
    "Cutout",
    "HeaderDict",
    "apply_depth_cut",
    "cutout_pixel_area_sr",
    "discover_cutouts",
    "empty_table",
    "filter_ok",
    "find_nearest_source",
    "have_fitsio",
    "load_catalog",
    "make_table",
    "normalize_catalog",
    "protected_indices",
    "read_cutout",
    "read_photometry",
    "sample_map_bilinear",
    "sample_map_bilinear_vec",
    "write_photometry",
    "zmag_from_flux_z",
]
