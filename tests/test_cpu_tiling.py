"""The tiled cpu-tractor solve: geometry, bookkeeping, and parity with whole-cutout.

Tiling is ORCHESTRATION around the unmodified upstream Tractor — many small
``optimize_forced_photometry`` solves instead of one big one — so what has to be
tested is the orchestration: that the tile cores partition the cutout, that every
in-cutout source is reported exactly once (halo overlaps neither duplicate nor
drop it), and that the fluxes match the whole-cutout solve where the two
geometries are physically equivalent (isolated sources).
"""
import numpy as np
import pytest

from tractorjax_spherex.tiling import iter_tiles, tile_core_index

tractor = pytest.importorskip("tractor")

from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.io.cutouts import read_cutout

# 40x40 synth cutouts at tile_size=15 -> 3x3 tiles, cores [0,15) [15,30) [30,40).
# Positions chosen so that: 5 sits in the CORE of tile (1,1) but in the HALO box
# of the three tiles left/below it (their halo boxes reach x,y < 18); 6 sits in
# the core of tile (0,0) and in the halo of the three tiles right/above it
# (whose boxes start at 12). Both are the case the core/halo rule exists for.
TILE_SOURCES = [
    {"x": 5.0, "y": 5.0, "flux_mjy": 4.0},
    {"x": 35.0, "y": 6.0, "flux_mjy": 3.0},
    {"x": 6.0, "y": 35.0, "flux_mjy": 2.5},
    {"x": 35.0, "y": 35.0, "flux_mjy": 2.0},
    {"x": 22.0, "y": 22.0, "flux_mjy": 1.5},
    {"x": 15.2, "y": 22.0, "flux_mjy": 1.2},   # just inside a core, in 3 halos
    {"x": 14.6, "y": 14.6, "flux_mjy": 1.0},   # just outside it, in 3 halos
]


@pytest.fixture
def tile_field(tmp_path):
    """A 2-cutout synthetic field whose sources probe the tile core/halo rule."""
    from fixtures.synth import make_synth_catalog, make_synth_field

    cut = tmp_path / "cut"
    make_synth_field(cut, n_cutouts=2, seed=5, sources=TILE_SOURCES)
    c0 = read_cutout(min(cut.glob("cutout_*.fits")))
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, TILE_SOURCES, c0.wcs)
    return {"cutouts_dir": cut, "catalog": cat, "sources": TILE_SOURCES,
            "cutout": c0}


def _run(field, *, cpu_tiling, **kw):
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           cpu_tiling=cpu_tiling, **kw)
    return run_photometry(field["cutouts_dir"], field["catalog"], cfg,
                          progress=False)


def _by_id(res):
    """``{id: mean flux}`` across the field's visits."""
    return {int(i): float(np.mean(res["flux"][res["id"] == i]))
            for i in np.unique(res["id"])}


# --------------------------------------------------------------------------- #
# Tile geometry (backend-neutral, no tractor needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("H,W", [(40, 40), (45, 45), (30, 47), (10, 10)])
def test_cores_partition_the_cutout(H, W):
    """Every in-cutout pixel centre belongs to exactly one tile CORE.

    This is the invariant the whole read-back rule rests on: no double counts,
    no drops, whatever the cutout size does modulo the tile size.
    """
    metas = list(iter_tiles(H, W, 15, 3))
    covered = np.zeros((H, W), dtype=np.int32)
    for m in metas:
        covered[m["core_y0"]:m["core_y1"], m["core_x0"]:m["core_x1"]] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_halo_boxes_extend_past_the_edge_and_are_not_clipped():
    metas = list(iter_tiles(40, 40, 15, 3))
    first, last = metas[0], metas[-1]
    assert (first["x_start"], first["y_start"]) == (-3, -3)
    assert (last["x_end"], last["y_end"]) == (48, 48)
    # ...while the cores stop at the cutout edge
    assert (last["core_x1"], last["core_y1"]) == (40, 40)


def test_tile_core_index_rejects_out_of_cutout_positions():
    metas = list(iter_tiles(40, 40, 15, 3))
    sx = np.array([0.0, 14.9, 15.0, 39.9, 40.0, -0.5, 20.0, np.nan])
    sy = np.array([0.0, 14.9, 15.0, 39.9, 20.0, 20.0, 40.5, 5.0])
    ti = tile_core_index(metas, sx, sy)
    assert ti[0] == 0                       # (0,0) core
    assert ti[1] == 0                       # still core (0,0): upper edge open
    assert ti[2] == metas.index(            # first pixel of core (1,1)
        next(m for m in metas if m["ix"] == 1 and m["iy"] == 1))
    assert ti[3] >= 0                       # last in-cutout pixel
    assert (ti[4:] == -1).all()             # x>=W, x<0, y>=H, NaN


# --------------------------------------------------------------------------- #
# Tiling actually engages, and the bookkeeping holds
# --------------------------------------------------------------------------- #
def test_tiled_build_makes_one_tractor_per_occupied_tile(tile_field):
    """Guard against a silent fall-back to one whole-cutout solve."""
    from astropy.coordinates import SkyCoord

    from tractorjax_spherex.backends.base import FieldContext
    from tractorjax_spherex.backends.cpu_backend import CpuTractorBackend
    from tractorjax_spherex.io.catalogs import load_catalog, normalize_catalog

    tab = normalize_catalog(load_catalog(tile_field["catalog"]))
    ctx = FieldContext(catalog=tab, sco_all=SkyCoord(
        ra=tab["ra"], dec=tab["dec"], unit="deg"), main_idx=0)
    cutout = tile_field["cutout"]

    tiled = CpuTractorBackend(PhotometryConfig(
        backend="cpu-tractor", solver="linear", cpu_tiling=True))
    whole = CpuTractorBackend(PhotometryConfig(
        backend="cpu-tractor", solver="linear", cpu_tiling=False))
    ti, wi = tiled.build(cutout, ctx), whole.build(cutout, ctx)

    assert len(wi["tiles"]) == 1
    assert 2 <= len(ti["tiles"]) <= 9        # occupied tiles of the 3x3 grid
    # the halo duplicates sources into several tiles, so the tiled model holds
    # MORE slots than the whole-cutout one while reporting the same set
    assert ti["n_model"] > wi["n_model"]
    assert np.array_equal(ti["report_ci"], wi["report_ci"])
    assert len(np.unique(ti["report_pos"])) == len(ti["report_pos"])


def test_every_in_cutout_source_reported_exactly_once(tile_field):
    """Structural bookkeeping: the halo can neither duplicate nor drop a source."""
    res = _run(tile_field, cpu_tiling=True)
    expected = {i + 1 for i in range(len(TILE_SOURCES))}   # all land in-cutout
    for cidx in np.unique(res["cutout_index"]):
        ids = np.asarray(res["id"][res["cutout_index"] == cidx])
        assert len(ids) == len(set(ids.tolist())), "a source was double-counted"
        assert set(ids.tolist()) == expected, "a source was dropped"


def test_tiled_and_whole_report_the_same_sources(tile_field):
    tiled = _run(tile_field, cpu_tiling=True)
    whole = _run(tile_field, cpu_tiling=False)
    tiled.sort(["cutout_index", "id"])
    whole.sort(["cutout_index", "id"])
    assert np.array_equal(np.asarray(tiled["id"]), np.asarray(whole["id"]))
    assert np.array_equal(np.asarray(tiled["cutout_index"]),
                          np.asarray(whole["cutout_index"]))


# --------------------------------------------------------------------------- #
# Flux parity where the two geometries are physically equivalent
# --------------------------------------------------------------------------- #
def test_tiled_recovers_injected_flux(tile_field):
    res = _run(tile_field, cpu_tiling=True)
    got = _by_id(res)
    for i, s in enumerate(TILE_SOURCES, start=1):
        assert got[i] == pytest.approx(s["flux_mjy"], rel=0.05)


def test_tiled_matches_whole_cutout_on_isolated_sources(synth_field):
    """Two well-separated point sources: tiling changes nothing measurable."""
    tiled = _by_id(_run(synth_field, cpu_tiling=True))
    whole = _by_id(_run(synth_field, cpu_tiling=False))
    assert set(tiled) == set(whole)
    for sid in tiled:
        assert tiled[sid] == pytest.approx(whole[sid], rel=0.01)


def test_tile_boundary_sources_match_whole_cutout(tile_field):
    """Sources straddling a tile edge — the case the halo exists for.

    ids 6 and 7 sit within a pixel of the x=15 / y=15 tile boundary, so each is
    in the core of one tile and the halo of three others. Their fluxes must not
    depend on which tile claims them.
    """
    tiled = _by_id(_run(tile_field, cpu_tiling=True))
    whole = _by_id(_run(tile_field, cpu_tiling=False))
    for sid in (6, 7):
        assert tiled[sid] == pytest.approx(whole[sid], rel=0.03)


def test_tiled_matches_whole_cutout_with_the_psf_fixes_on(tile_field):
    """The per-tile constant kernel is the same PSF field the whole-cutout
    ZoneBlendedPSF quantizes to (grid == tile_size), so turning the PSF fixes on
    must not split the two paths apart."""
    kw = dict(psf_zone_interp=True, psf_core_shift=True)
    tiled = _by_id(_run(tile_field, cpu_tiling=True, **kw))
    whole = _by_id(_run(tile_field, cpu_tiling=False, **kw))
    for sid in tiled:
        assert tiled[sid] == pytest.approx(whole[sid], rel=0.03)


def test_tiled_agrees_with_the_jax_backend(tile_field):
    """Same tile geometry on both engines — the point of sharing tiling.py."""
    pytest.importorskip("tractor_jax")
    cpu = _by_id(_run(tile_field, cpu_tiling=True))
    jax = _by_id(run_photometry(
        tile_field["cutouts_dir"], tile_field["catalog"],
        PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                         prefetch="sync", solver="linear", pad_bucket=0),
        progress=False))
    assert set(cpu) == set(jax)
    for sid in cpu:
        assert cpu[sid] == pytest.approx(jax[sid], rel=0.02)


# --------------------------------------------------------------------------- #
# Per-tile PSF: one constant kernel per tile, resolved at the CORE centre
# --------------------------------------------------------------------------- #
def _multizone_cutout(tmp_path, *, interp_stamps=True):
    """A synth cutout re-labelled with a 2x2 PSF-zone lattice across its span.

    The delivered synth cutout is single-zone (one plane, one row), where every
    zone question is a no-op. Give it four zones at a lattice pitch small enough
    that the 40 px cutout spans them, and four visibly different kernels, so the
    per-tile kernel choice is observable.
    """
    import dataclasses

    from astropy.table import Table

    from fixtures.synth import make_synth_field
    from tractorjax_spherex.io.cutouts import read_cutout as _read
    from tractorjax_spherex.simulate import gaussian_oversampled

    d = tmp_path / "mz"
    make_synth_field(d, n_cutouts=1, seed=1, sources=TILE_SOURCES)
    c = _read(min(d.glob("cutout_*.fits")))

    # four 101x101 planes of increasing width -> the kernel choice is visible
    cube = np.stack([gaussian_oversampled(101, 10, f)
                     for f in (2.0, 2.4, 2.8, 3.2)]).astype(np.float32)
    # lattice pitch 60 px, so the whole 40 px cutout is strictly INSIDE it and
    # every position gets its own bilinear weights (outside, the simulator
    # convention clamps and neighbouring positions collapse onto one kernel)
    zones = Table({"zone_id": [1, 2, 3, 4],
                   "x": [0.0, 60.0, 0.0, 60.0],
                   "y": [0.0, 0.0, 60.0, 60.0],
                   "plane_idx": [0, 1, 2, 3]})
    return dataclasses.replace(c, psf_cube=cube, psf_zones=zones)


def test_per_tile_psf_tracks_the_tile_core_centre(tmp_path):
    """Tiles in different zones must get different kernels, blended AT the
    clipped core centre — the JAX backend's convention, not a quantized cell."""
    from tractorjax_spherex import prepare as _prepare
    from tractorjax_spherex.backends.zone_psf import build_cpu_psf_selector

    cutout = _multizone_cutout(tmp_path)
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           psf_zone_interp=True)
    select = build_cpu_psf_selector(cutout, cfg, prepare=_prepare)

    a = select(7.5, 7.5)        # core centre of tile (0,0)
    b = select(35.0, 35.0)      # core centre of the CLIPPED tile (2,2): [30,40)
    assert not np.allclose(a.img, b.img), "every tile got the same kernel"

    # the clipped tile's centre is 35.0, not the nominal cell centre 37.5
    assert np.allclose(select(35.0, 35.0).img, b.img)
    assert not np.allclose(select(37.5, 37.5).img, b.img)


def test_per_tile_blend_is_bilinear_and_matches_the_jax_weights(tmp_path):
    """The zone blend must be (a) genuinely bilinear and (b) the SAME blend the
    JAX backend applies.

    The JAX engine blends in Fourier space as ``sum_k w_k FFT(b_k)``, which by
    linearity equals ``FFT(sum_k w_k b_k)``, so comparing the real-space blended
    kernels is exact rather than an approximation. Verified on real 12-zone
    a2537 data at machine precision; this pins it in CI on a synthetic lattice.
    """
    from tractorjax_spherex import prepare as _prepare
    from tractorjax_spherex.backends.zone_psf import (
        build_cpu_psf_selector,
        zone_stamp_provider,
    )

    cutout = _multizone_cutout(tmp_path)
    zones = cutout.psf_zones
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           psf_zone_interp=True)
    select = build_cpu_psf_selector(cutout, cfg, prepare=_prepare)
    get = zone_stamp_provider(cutout, cfg, prepare=_prepare)
    basis = np.stack([get(r) for r in range(len(zones))])
    _jax_basis, jax_weights = _prepare.zone_psf_basis(cutout)

    H, W = cutout.image.shape
    n_blended = 0
    for m in iter_tiles(H, W, cfg.tile_size, cfg.tile_halo):
        cx = 0.5 * (m["core_x0"] + m["core_x1"])
        cy = 0.5 * (m["core_y0"] + m["core_y1"])
        w = np.asarray(jax_weights(cx, cy), dtype=np.float64)

        # (a) a partition of unity over the lattice, and genuinely interpolating
        assert w.min() >= 0.0
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        if np.count_nonzero(w > 1e-12) > 1:
            n_blended += 1

        # (b) the CPU kernel IS sum_k w_k b_k with those same weights
        ref = np.tensordot(w, basis, axes=(0, 0))
        got = np.asarray(select(cx, cy).img, dtype=np.float64)
        assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-6

    assert n_blended > 0, "no tile blended >1 zone — the test lattice is a no-op"


def test_nearest_zone_selection_is_per_tile_when_interp_is_off(tmp_path):
    """With psf_zone_interp=False the JAX backend still picks the nearest zone
    PER TILE. Picking the cutout-centre zone for every tile is the bug."""
    from tractorjax_spherex import prepare as _prepare
    from tractorjax_spherex.backends.zone_psf import build_cpu_psf_selector

    cutout = _multizone_cutout(tmp_path)
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           psf_zone_interp=False)
    select = build_cpu_psf_selector(cutout, cfg, prepare=_prepare)
    corners = [select(7.5, 7.5), select(35.0, 7.5),
               select(7.5, 35.0), select(35.0, 35.0)]
    # four tiles nearest four different zones -> four different kernels
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.allclose(corners[i].img, corners[j].img)
    # ...and the same tile centre returns the SAME cached object
    assert select(7.5, 7.5) is corners[0]


# --------------------------------------------------------------------------- #
# Per-tile background column
# --------------------------------------------------------------------------- #
def test_tile_background_column_absorbs_a_pedestal():
    """The mechanism: a DC residual the per-cutout prefit missed biases every
    flux in the tile; the free constant takes it instead and the fluxes come
    back exact."""
    from tractor import Flux, PixPos, PointSource, Tractor

    from tractorjax_spherex.backends.cpu_backend import _forced_solve, _make_image
    from tractorjax_spherex.backends.cpu_psf import OversampledPixelizedPSF
    from tractorjax_spherex.simulate import gaussian_oversampled

    psf = OversampledPixelizedPSF(
        gaussian_oversampled(51, 5, 2.5).astype(np.float32), sampling=0.2)
    truth = [(8.0, 8.0, 3.0), (14.0, 13.0, 1.5)]
    pedestal = 0.05

    def solve(level, fit_sky):
        data = np.full((21, 21), level, dtype=np.float32)
        for x, y, f in truth:
            (psf.getPointSourcePatch(x, y) * f).addTo(data)
        tim = _make_image(data, np.full((21, 21), 1e4), psf, fit_sky=fit_sky)
        srcs = []
        for x, y, _f in truth:
            s = PointSource(PixPos(x, y), Flux(0.1))
            s.freezeAllRecursive()
            s.thawParam("brightness")
            srcs.append(s)
        return _forced_solve(Tractor([tim], srcs), fit_sky=fit_sky)[0], tim

    exact, _ = solve(0.0, False)
    biased, _ = solve(pedestal, False)
    fixed, tim = solve(pedestal, True)

    want = np.array([f for _, _, f in truth])
    assert exact == pytest.approx(want, rel=1e-3)
    assert np.max(np.abs(biased / want - 1)) > 0.2, "pedestal did not bias"
    assert fixed == pytest.approx(want, rel=1e-3)
    assert tim.getSky().getValue() == pytest.approx(pedestal, rel=1e-3)


def test_tile_background_flag_reaches_every_tile(tile_field):
    """Guard the silent-no-op failure mode: the flag must thaw the sky on the
    tile images the solve actually runs on."""
    from astropy.coordinates import SkyCoord

    from tractorjax_spherex.backends.base import FieldContext
    from tractorjax_spherex.backends.cpu_backend import CpuTractorBackend
    from tractorjax_spherex.io.catalogs import load_catalog, normalize_catalog

    tab = normalize_catalog(load_catalog(tile_field["catalog"]))
    ctx = FieldContext(catalog=tab, sco_all=SkyCoord(
        ra=tab["ra"], dec=tab["dec"], unit="deg"), main_idx=0)

    for flag, want in ((False, 0), (True, 1)):
        be = CpuTractorBackend(PhotometryConfig(
            backend="cpu-tractor", solver="linear", cpu_tiling=True,
            cpu_tile_background=flag))
        inputs = be.build(tile_field["cutout"], ctx)
        assert inputs["fit_sky"] is flag
        for trac in inputs["tiles"]:
            assert trac.images[0].numberOfParams() == want


def test_tile_background_is_a_small_correction_after_a_real_prefit(tile_field):
    """With the normal background model the prefit already removed the DC, so
    the column is a refinement, not a different measurement."""
    off = _by_id(_run(tile_field, cpu_tiling=True, cpu_tile_background=False))
    on = _by_id(_run(tile_field, cpu_tiling=True, cpu_tile_background=True))
    for sid in off:
        assert on[sid] == pytest.approx(off[sid], rel=0.05)


# --------------------------------------------------------------------------- #
# A source no pixel constrains must not report the Flux(0.1) seed
# --------------------------------------------------------------------------- #
def test_fully_masked_tile_reports_no_flux_not_the_seed(tmp_path):
    """Upstream's forced photometry is an UPDATE from the current parameters: a
    source with an all-zero column is never stepped, so it still carries the
    Flux(0.1) seed. Reading that back would publish a fabricated ~0.1 mJy
    detection. Tiling makes this reachable — one masked bright-star footprint can
    cover a whole 21x21 tile, where the whole-cutout solve needed the entire
    cutout to be unusable.
    """
    from astropy.io import fits

    from fixtures.synth import make_synth_catalog, make_synth_field
    from tractorjax_spherex.constants import MASKBITS

    srcs = [{"x": 7.0, "y": 7.0, "flux_mjy": 4.0},      # tile (0,0) core
            {"x": 30.0, "y": 8.0, "flux_mjy": 3.0}]     # untouched
    d = tmp_path / "cut"
    make_synth_field(d, n_cutouts=1, seed=2, sources=srcs)
    path = min(d.glob("cutout_*.fits"))

    # mask tile (0,0)'s ENTIRE in-cutout halo box, [0,18) x [0,18)
    bit = int(MASKBITS) & -int(MASKBITS)      # lowest bit that MASKBITS selects
    with fits.open(path, mode="update") as hdul:
        hdul["FLAGS"].data[0:18, 0:18] |= bit

    c0 = read_cutout(path)
    cat = tmp_path / "cat.parquet"
    make_synth_catalog(cat, srcs, c0.wcs)
    field = {"cutouts_dir": d, "catalog": cat}

    res = _run(field, cpu_tiling=True)
    res.sort("id")
    flux = np.asarray(res["flux"])
    ferr = np.asarray(res["flux_err"])

    # source 1: no live pixel constrains it -> no flux, infinite error.
    assert flux[0] == 0.0, f"reported {flux[0]} — the Flux(0.1) seed leaked out"
    assert not np.isfinite(ferr[0])
    # source 2 is unaffected and still measured
    assert flux[1] == pytest.approx(srcs[1]["flux_mjy"], rel=0.05)
    assert np.isfinite(ferr[1]) and ferr[1] > 0


# --------------------------------------------------------------------------- #
# Config surface
# --------------------------------------------------------------------------- #
def test_tiling_and_tile_background_default_on_and_round_trip(tmp_path):
    """Both default on: the cpu-tractor backend out of the box runs the SAME
    solve as the JAX backend — same tiles, same per-tile background column."""
    cfg = PhotometryConfig()
    assert cfg.cpu_tiling is True
    assert cfg.cpu_tile_background is True
    p = tmp_path / "cfg.yaml"
    PhotometryConfig(backend="cpu-tractor", solver="linear", cpu_tiling=False,
                     cpu_tile_background=False).to_yaml(p)
    loaded = PhotometryConfig.from_file(p)
    assert loaded.cpu_tiling is False and loaded.cpu_tile_background is False


def test_defaults_do_not_reject_the_other_backends_or_the_untiled_path():
    """The default cpu_tile_background=True must not make an ordinary config
    unconstructible. It is inert without tiles and true by construction on the
    JAX backend, so neither combination is an error."""
    assert PhotometryConfig(backend="jax").cpu_tile_background is True
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           cpu_tiling=False)
    assert cfg.cpu_tile_background is True      # accepted, and inert


def test_untiled_path_ignores_the_background_flag(tile_field, caplog):
    """Inert, and SAID to be inert — not silently dropped."""
    import logging

    from astropy.coordinates import SkyCoord

    from tractorjax_spherex.backends.base import FieldContext
    from tractorjax_spherex.backends.cpu_backend import CpuTractorBackend
    from tractorjax_spherex.io.catalogs import load_catalog, normalize_catalog

    tab = normalize_catalog(load_catalog(tile_field["catalog"]))
    ctx = FieldContext(catalog=tab, sco_all=SkyCoord(
        ra=tab["ra"], dec=tab["dec"], unit="deg"), main_idx=0)
    cfg = PhotometryConfig(backend="cpu-tractor", solver="linear",
                           cpu_tiling=False, cpu_tile_background=True)
    with caplog.at_level(logging.INFO, logger="tractorjax_spherex"):
        be = CpuTractorBackend(cfg)
    assert "inert" in caplog.text
    inputs = be.build(tile_field["cutout"], ctx)
    assert inputs["fit_sky"] is False
    assert inputs["tiles"][0].images[0].numberOfParams() == 0
