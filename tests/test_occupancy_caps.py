"""Padding caps: measured auto-sizing, typed overflow, completeness reporting.

A tile that needs more flux slots than a fixed cap takes its whole cutout out
of the product, so these tests pin the three guards against that: the caps can
be sized from a measured occupancy, an overflow raises something actionable
rather than a bare ValueError, and a partial product is labelled as one.
"""

import numpy as np
import pytest
from astropy.table import Table

from tractorjax_spherex.config import CapExceededError, ConfigError, PhotometryConfig
from tractorjax_spherex.io.cutouts import discover_cutouts, read_cutout
from tractorjax_spherex.occupancy import Occupancy, measure_occupancy


def test_auto_cap_requires_a_measurement():
    cfg = PhotometryConfig(pad_bucket=0, max_ps_cap="auto")
    assert cfg.wants_auto_caps()
    with pytest.raises(ConfigError, match="needs a measured occupancy"):
        cfg.resolved_caps()


def test_auto_cap_sizes_from_occupancy_with_margin():
    occ = Occupancy(max_ps=100, max_gal=200, n_cutouts=1,
                    per_cutout={0: (100, 200)})
    cfg = PhotometryConfig(pad_bucket=0, max_ps_cap="auto",
                           max_gal_cap="auto", cap_auto_margin=1.1)
    ps, gal, _ = cfg.resolved_caps(occ)
    assert ps == 110 and gal == 220
    # A margin of exactly 1.0 must still admit the densest tile.
    cfg.cap_auto_margin = 1.0
    ps, gal, _ = cfg.resolved_caps(occ)
    assert ps == 100 and gal == 200


def test_auto_cap_is_inert_when_caps_are_off():
    # pad_bucket on => bucketed padding replaces the caps entirely.
    cfg = PhotometryConfig(pad_bucket=32, max_ps_cap="auto")
    assert not cfg.wants_auto_caps()
    assert cfg.resolved_caps() == (None, None, None)
    # z-cut => caps off as well, and "auto" must not demand a measurement.
    cfg = PhotometryConfig(pad_bucket=0, fit_zmag_max=21.0, max_ps_cap="auto")
    assert cfg.resolved_caps() == (None, None, None)


def test_bad_cap_string_rejected():
    with pytest.raises(ConfigError):
        PhotometryConfig(max_ps_cap="automatic")
    with pytest.raises(ConfigError):
        PhotometryConfig(cap_auto_margin=0.9)


def test_occupancy_scan_counts_point_sources_and_galaxies(synth_field):
    pairs = discover_cutouts(synth_field["cutouts_dir"])
    cat = Table.read(synth_field["catalog"])
    occ = measure_occupancy(pairs, cat, tile_size=15, halo=3)

    assert occ.n_cutouts == len(pairs)
    # The synthetic field is two well-separated point sources, so no tile can
    # hold more than the whole catalog and none of them are galaxies.
    assert 0 < occ.max_ps <= len(cat)
    assert occ.max_gal == 0
    # An "auto" cap built from this must admit every cutout.
    cfg = PhotometryConfig(pad_bucket=0, max_ps_cap="auto", max_gal_cap="auto")
    ps, gal, _ = cfg.resolved_caps(occ)
    assert occ.overflowing(ps, gal) == []


def test_occupancy_flags_the_cutouts_a_small_cap_would_drop(synth_field):
    pairs = discover_cutouts(synth_field["cutouts_dir"])
    cat = Table.read(synth_field["catalog"])
    occ = measure_occupancy(pairs, cat, tile_size=15, halo=3)
    # A cap of zero point sources cannot hold the densest tile of any cutout
    # that contains a source.
    doomed = occ.overflowing(0, None)
    assert doomed == sorted(i for i, (ps, _) in occ.per_cutout.items() if ps > 0)
    assert doomed  # the fixture does put sources in the cutouts


def test_cap_overflow_raises_actionable_error(synth_field):
    """An over-tight cap must fail with the width it actually needed."""
    from tractorjax_spherex.backends.jax_backend import _check_caps

    cat = Table.read(synth_field["catalog"])
    n = len(cat)
    tile_records = [{"src_indices": list(range(n))}]

    with pytest.raises(CapExceededError) as ei:
        _check_caps(tile_records, cat, 0, None, cutout_index=7)
    err = ei.value
    assert err.kind == "ps" and err.needed == n and err.cap == 0
    assert err.cutout_index == 7
    # The message has to carry the remedy, not just the number.
    assert "auto" in str(err) and "SKIPPED" in str(err)

    # Roomy caps pass silently, and None means "no cap".
    _check_caps(tile_records, cat, n, n, cutout_index=7)
    _check_caps(tile_records, cat, None, None)


def test_galaxy_overflow_reported_as_gal(synth_field):
    from tractorjax_spherex.backends.jax_backend import _check_caps

    cat = Table.read(synth_field["catalog"])
    cat["shape_r"] = np.full(len(cat), 1.0)   # every source is now extended
    tile_records = [{"src_indices": list(range(len(cat)))}]
    with pytest.raises(CapExceededError) as ei:
        _check_caps(tile_records, cat, None, 0)
    assert ei.value.kind == "gal"


def test_complete_flag_written_for_a_clean_run(synth_field, tmp_path):
    from tractorjax_spherex.io.output import read_photometry
    from tractorjax_spherex.pipeline import run_photometry

    out = tmp_path / "phot.parquet"
    cfg = PhotometryConfig(device="cpu", prefetch="sync", solver="linear")
    res = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         cfg, output=out, progress=False)

    assert res.meta["tractorjax_spherex.complete"] is True
    assert res.meta["tractorjax_spherex.n_cutouts_failed"] == 0
    _tab, meta = read_photometry(out)
    assert meta["tractorjax_spherex.complete"] is True


def test_strict_turns_a_skipped_cutout_into_an_error(synth_field, tmp_path,
                                                     monkeypatch):
    """Default: skip and label incomplete. strict=True: raise."""
    from tractorjax_spherex import pipeline as pl
    from tractorjax_spherex.backends import jax_backend

    def boom(*a, **k):
        raise RuntimeError("synthetic build failure")

    monkeypatch.setattr(jax_backend.JaxBackend, "build", boom)

    lenient = PhotometryConfig(device="cpu", prefetch="sync", solver="linear")
    res = pl.run_photometry(synth_field["cutouts_dir"],
                            synth_field["catalog"], lenient,
                            output=tmp_path / "partial.parquet", progress=False)
    assert res.meta["tractorjax_spherex.complete"] is False
    assert res.meta["tractorjax_spherex.n_cutouts_failed"] > 0
    assert len(res) == 0

    strict = PhotometryConfig(device="cpu", prefetch="sync", solver="linear",
                              strict=True)
    with pytest.raises(RuntimeError, match="synthetic build failure"):
        pl.run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                          strict, progress=False)


def test_tiles_take_their_own_zone_psf(one_cutout):
    """Each tile must resolve the PSF at its own centre, not the cutout's."""
    from tractorjax_spherex.prepare import zone_psf_selector

    cutout = read_cutout(one_cutout["path"])
    select = zone_psf_selector(cutout)
    H, W = cutout.image.shape

    a = select(0.0, 0.0)
    b = select(W - 1.0, H - 1.0)
    assert a.ndim == 2 and a.shape == b.shape
    n_planes = cutout.psf_cube.shape[0]
    if n_planes == 1:
        # Single-zone cutout: every tile shares the one plane (and the cache
        # must hand back the identical object rather than re-downsampling).
        assert a is b
    # Repeat lookups are served from the cache.
    assert select(0.0, 0.0) is a
