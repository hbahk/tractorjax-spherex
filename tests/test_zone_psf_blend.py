"""Bilinear zone-PSF blending: weights, and the single-zone no-op guarantee."""
import numpy as np
import pytest
from astropy.table import Table

from tractorjax_spherex.prepare import zone_bilinear_weights, zone_psf_basis

PITCH = 185.4


def _zones(nx, ny, x0=1000.0, y0=1000.0):
    xs, ys = np.meshgrid(x0 + PITCH * np.arange(nx), y0 + PITCH * np.arange(ny))
    return Table({"x": xs.ravel(), "y": ys.ravel(),
                  "plane_idx": np.arange(xs.size)})


def test_single_zone_is_one_hot():
    """The no-op guarantee: one zone -> weight 1, so blending == nearest."""
    z = _zones(1, 1)
    w = zone_bilinear_weights(z, 1234.0, 4321.0)
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0)


def test_on_a_zone_centre_is_one_hot():
    z = _zones(3, 3)
    w = zone_bilinear_weights(z, z["x"][4], z["y"][4])
    assert w[4] == pytest.approx(1.0)
    assert np.count_nonzero(w) == 1


def test_midpoint_splits_evenly_and_sums_to_one():
    z = _zones(3, 3)
    w = zone_bilinear_weights(z, 0.5 * (z["x"][0] + z["x"][1]),
                              0.5 * (z["y"][0] + z["y"][3]))
    assert w.sum() == pytest.approx(1.0)
    assert np.sort(w)[-4:] == pytest.approx([0.25] * 4)


def test_outside_the_lattice_clamps_not_extrapolates():
    """Simulator convention: pin to the boundary rather than extrapolate."""
    z = _zones(3, 3)
    edge = zone_bilinear_weights(z, z["x"][0] - 10 * PITCH, z["y"][0])
    assert edge.min() >= 0.0
    assert edge.sum() == pytest.approx(1.0)
    assert edge[0] == pytest.approx(1.0)


def test_zone_psf_basis_shares_one_object_and_normalizes():
    """The engine keys its transform cache on identity, so the basis must be
    one object; weights must sum to 1 at every tile centre."""
    n = 3
    zones = _zones(n, n, x0=0.0, y0=0.0)
    rng = np.random.default_rng(0)
    cube = rng.random((n * n, 21, 21)).astype(np.float32)
    cube /= cube.sum(axis=(1, 2), keepdims=True)
    cutout = {"psf_zones": zones, "psf_cube": cube,
              "crpix1a": 1.0, "crpix2a": 1.0}
    basis, weights = zone_psf_basis(cutout)
    assert len(basis) == n * n
    assert all(isinstance(k, np.ndarray) for k in basis)
    for xy in ((0.0, 0.0), (50.0, 120.0), (300.0, 300.0)):
        w = weights(*xy)
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()
