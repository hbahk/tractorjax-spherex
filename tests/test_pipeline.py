import pytest
from astropy.table import Table

pytest.importorskip("tractor_jax")

from tractorjax_spherex import PhotometryConfig, run_photometry
from tractorjax_spherex.io.output import COLUMN_NAMES, QUALITY_COLUMN_NAMES, read_photometry


def _cfg(**kw):
    return PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                            prefetch="sync", solver="linear", pad_bucket=0, **kw)


def test_output_schema_and_metadata(synth_field, tmp_path):
    out = tmp_path / "phot.parquet"
    res = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         _cfg(), output=out, progress=False)
    assert tuple(res.colnames) == COLUMN_NAMES + QUALITY_COLUMN_NAMES
    tab, meta = read_photometry(out)
    assert meta["tractorjax_spherex.schema_version"] == 1
    assert "tractorjax_spherex.config" in meta
    assert len(tab) == len(res)


def test_two_cutouts_two_sources(synth_field):
    res = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         _cfg(), progress=False)
    # 2 cutouts x 2 in-frame sources
    assert len(res) == 4
    assert set(res["id"]) == {1, 2}


def test_resume_skips_done(synth_field, tmp_path):
    out = tmp_path / "phot.parquet"
    run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                   _cfg(), output=out, max_cutouts=1, progress=False)
    n1 = len(read_photometry(out)[0])
    run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                   _cfg(), output=out, resume=True, progress=False)
    n2 = len(read_photometry(out)[0])
    assert n2 > n1   # remaining cutout(s) appended, nothing duplicated
    tab = read_photometry(out)[0]
    assert set(tab["cutout_index"]) == {0, 1}


def test_bring_your_own_table(synth_field):
    cat = Table.read(synth_field["catalog"])
    res = run_photometry(synth_field["cutouts_dir"], cat, _cfg(), progress=False)
    assert len(res) == 4


def test_target_forces_main_source(synth_field):
    # explicit target near source 1; run still succeeds
    s = synth_field["wcs"].pixel_to_world(10.0, 10.0)
    res = run_photometry(synth_field["cutouts_dir"], synth_field["catalog"],
                         _cfg(), target=(s.ra.deg, s.dec.deg), progress=False)
    assert len(res) == 4
