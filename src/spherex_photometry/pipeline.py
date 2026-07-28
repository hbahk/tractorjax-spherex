"""High-level pipeline: cutouts + reference catalog + config -> spectrophotometry.

:func:`run_photometry` is the one call most users need. It discovers the cutouts,
prepares the catalog, runs the chosen backend over every cutout (prefetching the
CPU build of the next cutout while the current one solves, for the JAX backend),
and returns / writes a table with one spectrophotometric point per source per
visit.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table

from .backends import FieldContext, get_backend
from .config import PhotometryConfig
from .device import setup_device
from .io.catalogs import (
    apply_depth_cut,
    find_nearest_source,
    load_catalog,
    normalize_catalog,
    protected_indices,
)
from .io.cutouts import discover_cutouts, filter_ok, read_cutout
from .io.output import existing_cutout_indices, make_table, write_photometry
from .priors import make_prior_context

logger = logging.getLogger(__name__)


def _field_center(catalog: Table, target):
    """Return ``(ra, dec)`` for the always-kept/labelled main source."""
    if target is not None:
        return float(target[0]), float(target[1])
    return (float(np.median(np.asarray(catalog["ra"], dtype=float))),
            float(np.median(np.asarray(catalog["dec"], dtype=float))))


def run_photometry(cutouts_dir, catalog, config: PhotometryConfig | None = None,
                   *, target=None, output=None, resume=False,
                   max_cutouts=None, progress=True) -> Table:
    """Run forced photometry over a field of SPHEREx cutouts.

    Parameters
    ----------
    cutouts_dir : path
        Directory of ``cutout_*.fits`` (+ optional ``summary.ecsv``) written by
        ``spherex_retrieval.retrieve``.
    catalog : path or astropy Table
        Reference catalog (see :mod:`spherex_photometry.io.catalogs`).
    config : PhotometryConfig, optional
        All options; defaults to the blind-production profile.
    target : (ra, dec), optional
        The always-kept / labelled main source. Defaults to the catalog centroid.
    output : path, optional
        If given, write the result parquet there (with reproducibility metadata).
    resume : bool
        Skip cutouts already present in ``output`` (requires ``output``).
    max_cutouts : int, optional
        Process only the first N cutouts (debugging).
    progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    astropy.table.Table
        One row per (cutout, source): ``cutout_index, obs_id, detector, id, ra,
        dec, central_wavelength, bandwidth, flux, flux_err`` (flux in mJy).
    """
    config = config or PhotometryConfig()
    config.validate()
    setup_device(device=config.device, precision=config.precision,
                 mem_fraction=config.gpu_mem_fraction,
                 preallocate=config.gpu_preallocate)
    backend = get_backend(config)

    cutouts_dir = Path(cutouts_dir)
    pairs = filter_ok(discover_cutouts(cutouts_dir), cutouts_dir / "summary.ecsv")
    if not pairs:
        raise SystemExit(f"No cutouts found under {cutouts_dir}")
    if resume and output is not None:
        done = existing_cutout_indices(output)
        pairs = [(i, p) for i, p in pairs if i not in done]
    if max_cutouts:
        pairs = pairs[:max_cutouts]
    logger.info("Processing %d cutouts (backend=%s, solver=%s)",
                len(pairs), backend.name, config.solver)

    tab = normalize_catalog(load_catalog(catalog))
    ra0, dec0 = _field_center(tab, target)
    main_idx, _ = find_nearest_source(tab, ra0, dec0)

    tab, _kept = apply_depth_cut(tab, config.fit_zmag_max, keep_indices=(main_idx,))
    main_idx, sco_all = find_nearest_source(tab, ra0, dec0)

    protect_ci = None
    if config.solver in ("lasso", "eigfloor_prior"):
        protect_ci = protected_indices(tab, config.protect_zmag_max,
                                       always=(main_idx,))
    prior_ctx = make_prior_context(tab, config)

    profile_lookup_fn = None
    if backend.name == "jax":
        from .models import get_profile_cached
        profile_lookup_fn = get_profile_cached

    # "auto" caps: measure the field's real densest-tile occupancy before the
    # first solve. Tile assignment is pure geometry (positions + shape_r
    # through each WCS), so this costs one header parse per cutout and removes
    # the whole class of "cap too small -> cutout silently dropped" failures.
    occupancy = None
    if config.wants_auto_caps():
        from .occupancy import measure_occupancy
        occupancy = measure_occupancy(pairs, tab, tile_size=config.tile_size,
                                      halo=config.tile_halo, progress=progress)
        logger.info("Occupancy scan over %d cutouts: densest tile holds "
                    "%d point sources / %d galaxies",
                    occupancy.n_cutouts, occupancy.max_ps, occupancy.max_gal)

    ctx = FieldContext(catalog=tab, sco_all=sco_all, main_idx=main_idx,
                       protect_ci=protect_ci, prior_ctx=prior_ctx,
                       profile_lookup_fn=profile_lookup_fn,
                       occupancy=occupancy)

    # Warn up front about cutouts the configured caps would drop, so the loss
    # is visible before a long run rather than in a line at the end of it.
    ps_cap, gal_cap, _ = config.resolved_caps(occupancy)
    if occupancy is not None:
        doomed = occupancy.overflowing(ps_cap, gal_cap)
        if doomed:
            logger.warning(
                "%d of %d cutouts exceed the configured caps (max_ps_cap=%s, "
                "max_gal_cap=%s) and WILL BE SKIPPED: %s. Field needs %d/%d; "
                "use max_ps_cap='auto' or pad_bucket=32.",
                len(doomed), occupancy.n_cutouts, ps_cap, gal_cap, doomed,
                occupancy.max_ps, occupancy.max_gal)

    cat_id = np.asarray(tab["id"], dtype=np.int64)
    cat_ra = np.asarray(tab["ra"], dtype=np.float64)
    cat_dec = np.asarray(tab["dec"], dtype=np.float64)

    cols: dict[str, list] = {k: [] for k in
                             ("cutout_index", "obs_id", "detector", "id",
                              "ra", "dec", "central_wavelength", "bandwidth",
                              "flux", "flux_err")}
    failed, nan_wave = [], []

    def build_fn(item):
        cutout_index, path = item
        try:
            cutout = read_cutout(path)
            return cutout_index, backend.build(cutout, ctx)
        except Exception:  # per-cutout skip semantics
            if config.strict:
                raise
            logger.exception("Cutout %d build failed", cutout_index)
            return cutout_index, None

    for cutout_index, inputs in _iterate(pairs, build_fn, backend, config, progress):
        if inputs is None:
            failed.append(cutout_index)
            continue
        try:
            fluxes_np, var_np = backend.solve(inputs)
            (ci, flux, ferr, lam, band), cwave = backend.extract(
                inputs, fluxes_np, var_np)
        except Exception:
            if config.strict:
                raise
            logger.exception("Cutout %d solve failed", cutout_index)
            failed.append(cutout_index)
            continue
        cutout = inputs["cutout"]
        obs_id = str(cutout["primary_header"].get("OBSID", ""))
        det = int(cutout["detector"])
        if cwave is None:
            nan_wave.append(cutout_index)
        n = len(ci)
        cols["cutout_index"].append(np.full(n, cutout_index, dtype="i8"))
        cols["obs_id"].append(np.full(n, obs_id, dtype="U32"))
        cols["detector"].append(np.full(n, det, dtype="i4"))
        cols["id"].append(cat_id[ci])
        cols["ra"].append(cat_ra[ci])
        cols["dec"].append(cat_dec[ci])
        cols["central_wavelength"].append(lam)
        cols["bandwidth"].append(band)
        cols["flux"].append(flux)
        cols["flux_err"].append(ferr)

    if failed:
        logger.warning(
            "INCOMPLETE PRODUCT: %d of %d cutouts failed and were skipped: %s. "
            "The output parquet carries complete=False; re-run with "
            "strict=True to make this an error.",
            len(failed), len(pairs), failed)
    if nan_wave:
        logger.warning("%d cutouts had no CWAVE (wavelength=NaN, still "
                       "photometered): %s", len(nan_wave), nan_wave)

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    results = make_table({k: _cat(v) for k, v in cols.items()})
    # Completeness travels WITH the product: a reader must be able to tell a
    # partial run from a full one without access to the log that produced it.
    completeness = {
        "spherex_photometry.complete": not failed,
        "spherex_photometry.n_cutouts_attempted": len(pairs),
        "spherex_photometry.n_cutouts_failed": len(failed),
        "spherex_photometry.failed_cutouts": list(failed),
    }
    results.meta.update(completeness)

    if resume and output is not None and Path(output).exists():
        from .io.output import append_or_merge
        results = append_or_merge(output, results, config=config)
    elif output is not None:
        write_photometry(results, output, config=config,
                         extra_meta=completeness)
    logger.info("Photometered %d rows across %d cutouts",
                len(results), len(pairs) - len(failed))
    return results


def _iterate(pairs, build_fn, backend, config, progress):
    """Yield ``(cutout_index, inputs)``, prefetching builds for the JAX backend."""
    if backend.name == "jax" and config.prefetch == "thread":
        from tractor_jax.jax.pipeline import prefetch_pipeline
        it = prefetch_pipeline(pairs, build_fn, depth=2, executor="thread")
    else:
        it = (build_fn(item) for item in pairs)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=len(pairs), desc="Cutouts")
        except ImportError:
            pass
    return it
