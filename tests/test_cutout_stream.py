"""run_photometry over a cutout stream, and run_photometry_catalog over many targets.

A stream of cutouts (paths, bundle bytes, Cutout objects) must photometer
exactly like the directory the cutouts came from; per-cutout extras become
columns; several targets are all kept through the depth cut; and the catalogue
driver gives each target the rows its own single-field run gives.
"""
import numpy as np
import pytest
from astropy.table import Table

from tractorjax_spherex.backends import get_backend
from tractorjax_spherex.config import ConfigError, PhotometryConfig
from tractorjax_spherex.io.catalogs import load_catalog, nearest_sources
from tractorjax_spherex.io.cutouts import discover_cutouts, read_cutout
from tractorjax_spherex.pipeline import run_photometry, run_photometry_catalog

COLS = ("cutout_index", "obs_id", "detector", "id", "flux", "flux_err", "central_wavelength")


def _cfg(**kw):
    return PhotometryConfig(backend="jax", device="cpu", precision="fp64", solver="linear", **kw)


def _same(a, b):
    assert len(a) == len(b) > 0
    for c in COLS:
        x, y = np.asarray(a[c]), np.asarray(b[c])
        assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), c


@pytest.mark.parametrize("kind", ["path", "bytes", "cutout"])
def test_a_stream_photometers_like_its_directory(synth_field, kind):
    ref = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"], _cfg(), progress=False)
    pairs = discover_cutouts(synth_field["cutouts_dir"])
    if kind == "path":
        items = pairs
    elif kind == "bytes":
        items = [(i, p.read_bytes()) for i, p in pairs]
    else:
        items = [(i, read_cutout(p)) for i, p in pairs]
    got = run_photometry(iter(items), synth_field["catalog"], _cfg(), progress=False)
    _same(ref, got)
    assert got.meta["tractorjax_spherex.n_cutouts_attempted"] == len(pairs)
    assert got.meta["tractorjax_spherex.complete"] is True


def test_extras_become_columns(synth_field):
    pairs = discover_cutouts(synth_field["cutouts_dir"])
    items = [(i, p, {"target": 7, "field": "synth"}) for i, p in pairs]
    got = run_photometry(items, synth_field["catalog"], _cfg(), progress=False)
    assert set(np.asarray(got["target"])) == {7} and set(np.asarray(got["field"])) == {"synth"}
    bad = [(i, p, {"target": 1}) if k == 0 else (i, p, {"other": 1}) for k, (i, p) in enumerate(pairs)]
    with pytest.raises(ValueError):
        run_photometry(bad, synth_field["catalog"], _cfg(), progress=False)


def test_several_targets_are_all_kept_through_the_depth_cut(synth_field):
    cat = load_catalog(synth_field["catalog"])
    cat["flux_z"] = np.where(np.arange(len(cat)) < 2, 1e-3, 1e6)   # two faint sources
    faint = np.flatnonzero(np.asarray(cat["flux_z"]) < 1)
    pos = [(float(cat["ra"][i]), float(cat["dec"][i])) for i in faint]
    one = run_photometry(synth_field["cutouts_dir"], cat, _cfg(fit_zmag_max=21), target=pos[0],
                         progress=False)
    both = run_photometry(synth_field["cutouts_dir"], cat, _cfg(fit_zmag_max=21), targets=pos,
                          progress=False)
    ids = np.asarray(cat["id"])[faint]
    assert ids[0] in set(np.asarray(one["id"])) and ids[1] not in set(np.asarray(one["id"]))
    assert set(ids) <= set(np.asarray(both["id"]))
    assert list(nearest_sources(cat, [p[0] for p in pos], [p[1] for p in pos])) == list(faint)
    with pytest.raises(ValueError):
        run_photometry(synth_field["cutouts_dir"], cat, _cfg(), target=pos[0], targets=pos,
                       progress=False)


def test_auto_caps_need_a_directory(synth_field):
    cfg = PhotometryConfig(backend="jax", device="cpu", fit_zmag_max=None, pad_bucket=0,
                           max_ps_cap="auto")
    with pytest.raises(ConfigError):
        run_photometry(iter(discover_cutouts(synth_field["cutouts_dir"])),
                       synth_field["catalog"], cfg, progress=False)


def test_catalog_driver_gives_each_target_its_own_field(synth_field):
    cat = load_catalog(synth_field["catalog"])
    pairs = discover_cutouts(synth_field["cutouts_dir"])
    targets = Table({"ra": [float(cat["ra"][0]), float(cat["ra"][1])],
                     "dec": [float(cat["dec"][0]), float(cat["dec"][1])]})
    # the same two cutouts serve both targets, under distinct cutout indices
    bundles = [(0, [(100 + i, p) for i, p in pairs]), (1, [(200 + i, p) for i, p in pairs])]
    backend = get_backend(_cfg())
    got = run_photometry_catalog(targets, cat, bundles, _cfg(), radius_arcsec=3600.0,
                                 backend=backend)
    assert got.meta["tractorjax_spherex.n_targets"] == 2
    for t in (0, 1):
        mine = got[np.asarray(got["target"]) == t]
        ref = run_photometry(synth_field["cutouts_dir"], cat, _cfg(),
                             target=(targets["ra"][t], targets["dec"][t]), progress=False)
        assert np.array_equal(np.asarray(mine["flux"]), np.asarray(ref["flux"]))
        assert set(np.asarray(mine["cutout_index"])) == {100 * (t + 1) + i for i, _ in pairs}
