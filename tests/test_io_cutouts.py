import numpy as np

from fixtures.synth import make_synth_cutout
from tractorjax_spherex.io.cutouts import (
    Cutout,
    cutout_pixel_area_sr,
    discover_cutouts,
    filter_ok,
    read_cutout,
    sample_map_bilinear,
    sample_map_bilinear_vec,
)


def test_read_cutout_basic(one_cutout):
    c = read_cutout(one_cutout["path"])
    assert isinstance(c, Cutout)
    assert c.image.shape == (40, 40)
    assert c.psf_oversamp == 10
    assert c.cwave_map is not None and c.sapm is not None
    # dict-like access matches attribute access
    assert c["image"] is c.image and c.get("cwave_map") is c.cwave_map
    assert c.get("missing", "d") == "d"


def test_empty_side_hdus_guarded(tmp_path):
    p = tmp_path / "cutout_0000_X_D1.fits"
    make_synth_cutout(p, sources=[{"x": 20, "y": 20, "flux_mjy": 1.0}],
                      empty_side_hdus=True)
    c = read_cutout(p)
    assert c.cwave_map is None and c.cwave_center is None and c.sapm is None
    # falls back to WCS pixel area, finite and positive
    area = cutout_pixel_area_sr(c)
    assert np.all(np.isfinite(area)) and np.all(area > 0)


def test_sapm_gives_pixel_area(one_cutout):
    c = read_cutout(one_cutout["path"])
    area = cutout_pixel_area_sr(c)
    # 6.15 arcsec pixel -> ~8.9e-10 sr
    assert np.allclose(area, area.flat[0])
    assert 8e-10 < area.flat[0] < 1e-9


def test_discover_and_filter(tmp_path):
    from fixtures.synth import make_synth_field
    d = tmp_path / "cut"
    make_synth_field(d, n_cutouts=3)
    pairs = discover_cutouts(d)
    assert [i for i, _ in pairs] == [0, 1, 2]
    ok = filter_ok(pairs, d / "summary.ecsv")
    assert len(ok) == 3


def test_bilinear_scalar_matches_vector():
    arr = np.arange(20.0).reshape(4, 5)
    xs = np.array([0.3, 2.7, 4.9, -1.0])
    ys = np.array([0.1, 1.5, 3.2, 10.0])
    vec = sample_map_bilinear_vec(arr, xs, ys)
    for k in range(len(xs)):
        assert sample_map_bilinear(arr, xs[k], ys[k]) == vec[k]


def test_bilinear_none_is_nan():
    assert np.isnan(sample_map_bilinear(None, 1, 1))
    assert np.all(np.isnan(sample_map_bilinear_vec(None, [1, 2], [1, 2])))
