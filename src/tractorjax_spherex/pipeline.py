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
    nearest_sources,
    normalize_catalog,
    protected_indices,
)
from .io.cutouts import Cutout, discover_cutouts, filter_ok, read_cutout
from .io.output import existing_cutout_indices, make_table, write_photometry
from .priors import make_prior_context
from .quality import add_quality_flags

logger = logging.getLogger(__name__)


def _field_center(catalog: Table, target):
    """Return ``(ra, dec)`` for the always-kept/labelled main source."""
    if target is not None:
        return float(target[0]), float(target[1])
    return (float(np.median(np.asarray(catalog["ra"], dtype=float))),
            float(np.median(np.asarray(catalog["dec"], dtype=float))))


def run_photometry(cutouts, catalog, config: PhotometryConfig | None = None,
                   *, target=None, targets=None, output=None, resume=False,
                   max_cutouts=None, progress=True, backend=None) -> Table:
    """Run forced photometry over a field of SPHEREx cutouts.

    Parameters
    ----------
    cutouts : path or iterable
        A directory of ``cutout_*.fits`` (+ optional ``summary.ecsv``) written by
        ``spherex_retrieval.retrieve``; or an iterable, consumed lazily, of
        ``(cutout_index, source)`` or ``(cutout_index, source, extra)``, where
        ``source`` is a bundle path, the bundle's bytes
        (``spherex_retrieval.bundle.bundle_bytes``), a binary file object or a
        :class:`~tractorjax_spherex.io.cutouts.Cutout`, and ``extra`` a dict of
        per-cutout values copied into every row of that cutout (e.g.
        ``{"target": 12}``; the same keys for every cutout).
    catalog : path or astropy Table
        Reference catalog (see :mod:`tractorjax_spherex.io.catalogs`).
    config : PhotometryConfig, optional
        All options; defaults to the configuration of record (``eigfloor`` on
        the catalog truncated at z-band AB 21).
    target : (ra, dec), optional
        The always-kept / labelled main source. Defaults to the catalog centroid.
    targets : sequence of (ra, dec), optional
        Several always-kept sources (a field holding more than one target); the
        first labels the log. Mutually exclusive with ``target``.
    output : path, optional
        If given, write the result parquet there (with reproducibility metadata).
    resume : bool
        Skip cutouts already present in ``output`` (requires ``output``).
    max_cutouts : int, optional
        Process only the first N cutouts (debugging).
    progress : bool
        Show a tqdm progress bar.
    backend : optional
        A backend made for the same config (``get_backend(config)``), shared
        by several calls so they reuse its compiled solvers — many small
        fields then compile once (see :func:`run_photometry_catalog`).

    Returns
    -------
    astropy.table.Table
        One row per (cutout, source): ``cutout_index, obs_id, detector, id, ra,
        dec, central_wavelength, bandwidth, flux, flux_err`` (flux in mJy); with
        the per-visit diagnostics on (the default on the jax backend) also
        ``fit_chi2, mask_frac, quality_flag`` (see :mod:`tractorjax_spherex.quality`);
        plus the ``extra`` columns.
    """
    import itertools
    import os

    config = config or PhotometryConfig()
    config.validate()
    setup_device(device=config.device, precision=config.precision,
                 mem_fraction=config.gpu_mem_fraction,
                 preallocate=config.gpu_preallocate)
    backend = backend or get_backend(config)
    if target is not None and targets is not None:
        raise ValueError("pass target or targets, not both")

    from_dir = isinstance(cutouts, (str, os.PathLike))
    if from_dir:
        cutouts_dir = Path(cutouts)
        pairs = filter_ok(discover_cutouts(cutouts_dir), cutouts_dir / "summary.ecsv")
        if not pairs:
            raise SystemExit(f"No cutouts found under {cutouts_dir}")
        if resume and output is not None:
            done = existing_cutout_indices(output)
            pairs = [(i, p) for i, p in pairs if i not in done]
        if max_cutouts:
            pairs = pairs[:max_cutouts]
        items = pairs
        logger.info("Processing %d cutouts (backend=%s, solver=%s)",
                    len(pairs), backend.name, config.solver)
    else:
        items = iter(cutouts)
        if resume and output is not None:
            done = existing_cutout_indices(output)
            items = (it for it in items if it[0] not in done)
        if max_cutouts:
            items = itertools.islice(items, max_cutouts)
        logger.info("Processing a cutout stream (backend=%s, solver=%s)",
                    backend.name, config.solver)

    tab = normalize_catalog(load_catalog(catalog))
    if targets is None:
        ra0, dec0 = _field_center(tab, target)
        main_idx, _ = find_nearest_source(tab, ra0, dec0)
        keep = (main_idx,)
    else:
        centers = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        ra0, dec0 = float(centers[0, 0]), float(centers[0, 1])
        keep = tuple(nearest_sources(tab, centers[:, 0], centers[:, 1]))

    n_catalog = len(tab)
    tab, _kept = apply_depth_cut(tab, config.fit_zmag_max, keep_indices=keep)
    main_idx, sco_all = find_nearest_source(tab, ra0, dec0)
    kept_after = (main_idx,) if targets is None else tuple(
        nearest_sources(tab, centers[:, 0], centers[:, 1]))
    # The depth cut is the setting a user most needs to see applied: say what
    # it did, in the same breath as the main source it kept.
    logger.info("Catalog: %d sources, %d fitted (fit_zmag_max=%s); main source "
                "id=%d at ra=%.5f dec=%.5f", n_catalog, len(tab),
                config.fit_zmag_max, int(tab["id"][main_idx]),
                float(tab["ra"][main_idx]), float(tab["dec"][main_idx]))

    protect_ci = None
    if config.solver in ("lasso", "eigfloor_prior"):
        protect_ci = protected_indices(tab, config.protect_zmag_max,
                                       always=kept_after)
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
        if not from_dir:
            from .config import ConfigError
            raise ConfigError("max_*_cap='auto' measures the field from a cutout "
                              "directory; give explicit caps or pad_bucket for a stream")
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
    extra_cols: dict[str, list] | None = None
    diag_cols: dict[str, list] = {"fit_chi2": [], "mask_frac": []}
    failed, nan_wave = [], []
    n_attempted = 0

    def build_fn(item):
        cutout_index, source = item[0], item[1]
        try:
            cutout = source if isinstance(source, Cutout) else read_cutout(source)
            return item, backend.build(cutout, ctx)
        except Exception:  # per-cutout skip semantics
            if config.strict:
                raise
            logger.exception("Cutout %d build failed", cutout_index)
            return item, None

    total = len(pairs) if from_dir else None
    for item, inputs in _iterate(items, build_fn, backend, config, progress, total=total):
        cutout_index = item[0]
        n_attempted += 1
        if inputs is None:
            failed.append(cutout_index)
            continue
        try:
            fluxes_np, var_np = backend.solve(inputs)
            (ci, flux, ferr, lam, band), cwave = backend.extract(
                inputs, fluxes_np, var_np)
            diag = backend.extract_diagnostics(inputs) if config.diagnostics_on() else None
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
        if diag is not None:
            for k, chunks in diag_cols.items():
                chunks.append(diag[k])
        extra = item[2] if len(item) > 2 else None
        if extra_cols is None:
            extra_cols = {k: [] for k in (extra or {})}
        if set(extra or {}) != set(extra_cols):
            raise ValueError(f"cutout {cutout_index}: extra keys {sorted(extra or {})} "
                             f"differ from {sorted(extra_cols)}")
        for k, v in (extra or {}).items():
            extra_cols[k].append(np.full(n, v))

    if failed:
        logger.warning(
            "INCOMPLETE PRODUCT: %d of %d cutouts failed and were skipped: %s. "
            "The output parquet carries complete=False; re-run with "
            "strict=True to make this an error.",
            len(failed), n_attempted, failed)
    if nan_wave:
        logger.warning("%d cutouts had no CWAVE (wavelength=NaN, still "
                       "photometered): %s", len(nan_wave), nan_wave)

    def _cat(chunks):
        return np.concatenate(chunks) if chunks else np.zeros(0)

    results = make_table({k: _cat(v) for k, v in cols.items()})
    if config.diagnostics_on():
        for k, chunks in diag_cols.items():
            results[k] = _cat(chunks).astype(np.float32)
        add_quality_flags(results, config.visit_chi2_rel_max)
    for k, chunks in (extra_cols or {}).items():
        results[k] = _cat(chunks)
    # Completeness travels WITH the product: a reader must be able to tell a
    # partial run from a full one without access to the log that produced it.
    completeness = {
        "tractorjax_spherex.complete": not failed,
        "tractorjax_spherex.n_cutouts_attempted": n_attempted,
        "tractorjax_spherex.n_cutouts_failed": len(failed),
        "tractorjax_spherex.failed_cutouts": list(failed),
    }
    results.meta.update(completeness)

    if resume and output is not None and Path(output).exists():
        from .io.output import append_or_merge
        results = append_or_merge(output, results, config=config)
    elif output is not None:
        write_photometry(results, output, config=config,
                         extra_meta=completeness)
    logger.info("Photometered %d rows across %d cutouts",
                len(results), n_attempted - len(failed))
    return results


def run_photometry_catalog(targets, neighbors, bundles, config: PhotometryConfig | None = None,
                           *, radius_arcsec: float = 300.0, output=None, progress=False,
                           backend=None) -> Table:
    """Photometer many targets, each as its own field, with one set of compiled solvers.

    Parameters
    ----------
    targets : astropy Table or (N, 2) array
        Target positions (``ra``, ``dec`` columns, deg).
    neighbors : path or astropy Table
        A reference catalog covering every target's field, e.g. one Legacy
        Survey query per sky region. Each target is fitted with the rows within
        ``radius_arcsec`` of it (the cone the single-target path queries), with
        the target the always-kept source, so a target's rows are those
        :func:`run_photometry` gives for the same cone.
    bundles : iterable of ``(target_index, sources)``
        Per target, its cutouts as :func:`run_photometry` takes them
        (``(cutout_index, source)`` items); the cutout indices must be unique
        over the whole run.
    config : PhotometryConfig, optional
    output : path, optional
        One parquet for all targets (a ``target`` column says whose field).

    Returns
    -------
    astropy.table.Table
        The per-target tables stacked, with a ``target`` column.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import vstack

    config = config or PhotometryConfig()
    config.validate()
    if backend is None:
        # one backend for every target: its compiled solvers are reused
        setup_device(device=config.device, precision=config.precision,
                     mem_fraction=config.gpu_mem_fraction,
                     preallocate=config.gpu_preallocate)
        backend = get_backend(config)
    if isinstance(targets, Table):
        t_ra = np.asarray(targets["ra"], dtype=np.float64)
        t_dec = np.asarray(targets["dec"], dtype=np.float64)
    else:
        arr = np.asarray(targets, dtype=np.float64).reshape(-1, 2)
        t_ra, t_dec = arr[:, 0], arr[:, 1]
    nb = normalize_catalog(load_catalog(neighbors))
    nb_sc = SkyCoord(nb["ra"], nb["dec"], unit="deg")
    parts = []
    n_targets = 0
    for t, sources in bundles:
        t = int(t)
        center = SkyCoord(t_ra[t], t_dec[t], unit="deg")
        sub = nb[nb_sc.separation(center) < radius_arcsec * u.arcsec]
        items = ((item[0], item[1], {"target": t}) for item in sources)
        parts.append(run_photometry(items, sub, config, target=(t_ra[t], t_dec[t]),
                                    progress=progress, backend=backend))
        n_targets += 1
    results = vstack(parts, metadata_conflicts="silent") if parts else make_table(
        {k: np.zeros(0) for k in ("cutout_index", "obs_id", "detector", "id", "ra", "dec",
                                  "central_wavelength", "bandwidth", "flux", "flux_err")})
    failed = [i for p in parts for i in p.meta.get("tractorjax_spherex.failed_cutouts", [])]
    completeness = {
        "tractorjax_spherex.complete": not failed,
        "tractorjax_spherex.n_targets": n_targets,
        "tractorjax_spherex.n_cutouts_attempted": int(sum(
            p.meta.get("tractorjax_spherex.n_cutouts_attempted", 0) for p in parts)),
        "tractorjax_spherex.n_cutouts_failed": len(failed),
        "tractorjax_spherex.failed_cutouts": failed,
    }
    results.meta.update(completeness)
    if output is not None:
        write_photometry(results, output, config=config, extra_meta=completeness)
    return results


def _iterate(items, build_fn, backend, config, progress, total=None):
    """Yield ``(item, inputs)``, prefetching builds for the JAX backend."""
    if backend.name == "jax" and config.prefetch == "thread":
        from tractor_jax.jax.pipeline import prefetch_pipeline
        it = prefetch_pipeline(items, build_fn, depth=2, executor="thread")
    else:
        it = (build_fn(item) for item in items)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, total=total, desc="Cutouts")
        except ImportError:
            pass
    return it
