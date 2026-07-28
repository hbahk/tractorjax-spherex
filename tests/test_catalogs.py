import numpy as np
import pytest
from astropy.table import Table

from spherex_photometry.io.catalogs import (
    apply_depth_cut,
    find_nearest_source,
    normalize_catalog,
    protected_indices,
    zmag_from_flux_z,
)


def _cat():
    return Table({
        "ls_id": [10, 20, 30],
        "ra": [150.0, 150.01, 149.99],
        "dec": [2.0, 2.01, 1.99],
        "flux_z": [1000.0, 10.0, -1.0],   # bright, faint, undefined
        "shape_r": [0.0, 1.5, 0.0],
        "sersic": [1.0, 2.0, 1.0],
        "shape_e1": [0.0, 0.2, 0.0],
        "shape_e2": [0.0, 0.1, 0.0],
    })


def test_normalize_fills_and_derives():
    t = normalize_catalog(_cat())
    assert "id" in t.colnames and list(t["id"]) == [10, 20, 30]
    assert "shape_ab" in t.colnames and "shape_phi" in t.colnames
    # round source -> ab == 1, phi == 0
    assert t["shape_ab"][0] == pytest.approx(1.0)
    assert 0.0 <= t["shape_phi"][1] < 180.0


def test_normalize_requires_id_and_coords():
    with pytest.raises(ValueError):
        normalize_catalog(Table({"ra": [1.0], "dec": [1.0]}))
    with pytest.raises(ValueError):
        normalize_catalog(Table({"id": [1], "dec": [1.0]}))


def test_id_alias_and_missing_shapes():
    t = normalize_catalog(Table({"id": [5], "ra": [1.0], "dec": [1.0]}))
    assert t["shape_r"][0] == 0.0 and t["flux_z"][0] != t["flux_z"][0]  # NaN


def test_zmag_and_depth_cut():
    t = normalize_catalog(_cat())
    zmag = zmag_from_flux_z(t["flux_z"])
    assert np.isnan(zmag[2])  # flux_z < 0
    _cut, kept = apply_depth_cut(t, fit_zmag_max=18.0, keep_indices=(1,))
    # source 0 (z~15) passes; source 1 kept by keep_indices; source 2 dropped
    assert set(kept) == {0, 1}
    # no cut
    cut2, _kept2 = apply_depth_cut(t, fit_zmag_max=None)
    assert len(cut2) == 3


def test_protected_indices():
    t = normalize_catalog(_cat())
    prot = protected_indices(t, protect_zmag_max=18.0, always=(2,))
    assert 0 in prot and 2 in prot and 1 not in prot


def test_find_nearest():
    t = normalize_catalog(_cat())
    idx, sco = find_nearest_source(t, 150.01, 2.01)
    assert idx == 1 and len(sco) == 3
