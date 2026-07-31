"""The two PSF fixes (zone-interp, core-shift) on the CPU backend, and the
basis pass-through on the JAX backend.

The CPU implementation must satisfy three contracts:

1. one-hot blending == the plain single-stamp path (so the fix is a provable
   no-op wherever it should be one);
2. the core-shift moves the stamp centroid by exactly the applied amount, in
   the same direction as the engine's Fourier phase ramp;
3. the calib table is shipped and loads from the installed package.
"""

import numpy as np
import pytest

pytest.importorskip("tractor")

from spherex_photometry.backends.cpu_psf import OversampledPixelizedPSF
from spherex_photometry.backends.zone_psf import (
    ZoneBlendedPSF,
    shift_stamp_native,
)


def _gauss(sigma, n=51):
    y, x = np.mgrid[:n, :n] - (n - 1) / 2.0
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return g / g.sum()


def _centroid(img):
    y, x = np.mgrid[: img.shape[0], : img.shape[1]]
    s = img.sum()
    return float((y * img).sum() / s), float((x * img).sum() / s)


class _Zones:
    """Duck-typed stand-in for the psf_zones Table used by the weight fn."""

    def __init__(self, xs, ys):
        self._d = {"x": np.asarray(xs, float), "y": np.asarray(ys, float),
                   "plane_idx": np.arange(len(xs))}

    def __getitem__(self, k):
        return self._d[k]

    def __len__(self):
        return len(self._d["x"])


def test_one_hot_blend_matches_plain_path():
    """A blend that lands exactly on a zone centre must reproduce the plain
    OversampledPixelizedPSF for that zone's stamp, patch for patch."""
    from spherex_photometry.prepare import zone_bilinear_weights

    stamps = [_gauss(3.5), _gauss(4.5)]
    zones = _Zones([100.0, 300.0], [0.0, 0.0])
    blended = ZoneBlendedPSF(stamps, zones, pix_to_det=lambda x, y: (x, y),
                             sampling=0.2,
                             weights_fn=zone_bilinear_weights, grid=15)
    plain = OversampledPixelizedPSF(np.asarray(stamps[0], np.float32),
                                    sampling=0.2)
    # cell (0,0) centre (7.5, 7.5) lies OUTSIDE the lattice (leftmost zone at
    # x=100), where the weight convention CLAMPS -> exact one-hot on zone 0.
    # (Between zone centres it interpolates; one-hot holds only off-lattice.)
    pb = blended.getPointSourcePatch(5.0, 5.0)
    pp = plain.getPointSourcePatch(5.0, 5.0)
    assert pb.x0 == pp.x0 and pb.y0 == pp.y0
    np.testing.assert_allclose(pb.patch, pp.patch, rtol=0, atol=1e-6)


def test_midpoint_blend_is_the_average_kernel():
    """Halfway between two zones the rendered patch is the 50/50 kernel."""
    from spherex_photometry.prepare import zone_bilinear_weights

    stamps = [_gauss(3.5), _gauss(4.5)]
    zones = _Zones([0.0, 15.0], [0.0, 0.0])   # pitch = one grid cell
    blended = ZoneBlendedPSF(stamps, zones, pix_to_det=lambda x, y: (x, y),
                             sampling=0.2,
                             weights_fn=zone_bilinear_weights, grid=15)
    mean = OversampledPixelizedPSF(
        (0.5 * stamps[0] + 0.5 * stamps[1]).astype(np.float32), sampling=0.2)
    # cell centre 7.5 = the midpoint of the two zone centres
    pb = blended.getPointSourcePatch(7.0, 7.0)
    pm = mean.getPointSourcePatch(7.0, 7.0)
    np.testing.assert_allclose(pb.patch, pm.patch, rtol=0, atol=1e-6)


def test_delegates_are_cached_per_cell():
    from spherex_photometry.prepare import zone_bilinear_weights

    stamps = [_gauss(3.5), _gauss(4.5)]
    zones = _Zones([0.0, 40.0], [0.0, 0.0])
    b = ZoneBlendedPSF(stamps, zones, pix_to_det=lambda x, y: (x, y),
                       sampling=0.2, weights_fn=zone_bilinear_weights, grid=15)
    for x in (1.0, 5.0, 14.0):        # same cell
        b.getPointSourcePatch(x, 3.0)
    n_same = len(b._delegates)
    b.getPointSourcePatch(20.0, 3.0)  # next cell
    assert len(b._delegates) == n_same + 1


def test_shift_stamp_moves_centroid_by_the_applied_amount():
    """+dy native must move the stamp centroid by +dy/sampling stamp px, the
    engine phase-ramp direction (test_spherex_core_offset_correction_sign)."""
    stamp = _gauss(4.0)
    dy, dx = 0.05, 0.03               # native px, the realistic magnitude
    out = shift_stamp_native(stamp, dy, dx, sampling=0.2)
    cy0, cx0 = _centroid(stamp)
    cy1, cx1 = _centroid(out)
    assert cy1 - cy0 == pytest.approx(dy / 0.2, abs=0.02)
    assert cx1 - cx0 == pytest.approx(dx / 0.2, abs=0.02)
    # flux-preserving, like the engine's DC-untouched phase ramp
    assert out.sum() == pytest.approx(stamp.sum(), rel=1e-12)


def test_calib_table_ships_and_loads():
    from spherex_photometry.calib import (
        DOWNSAMPLE_GRID_SHIFT_NATIVE,
        psf_core_shift,
    )
    s = psf_core_shift(1, 60)
    assert s.source == "zone"
    # measured offsets are ~ +0.05 native px applied, both axes
    assert 0.0 < s.dy_apply < 0.2 and 0.0 < s.dx_apply < 0.2
    assert DOWNSAMPLE_GRID_SHIFT_NATIVE == pytest.approx(0.05)


def test_cross_backend_agreement_with_fixes(synth_field):
    """Both backends, both fixes requested: fluxes still agree.

    The synth fixture is single-zone, so zone-interp is a no-op on both sides
    by construction (the CPU one-hot test above and the JAX engine's own
    one-hot test each prove their half) and core-shift on the JAX backend is
    rejected without a basis — which the config contract documents. What THIS
    test pins is that the CPU backend's standalone core-shift path runs
    end-to-end and stays within tolerance of the unshifted JAX solve: a 0.05
    native px kernel shift moves synthetic point-source fluxes well under 1%.
    """
    from spherex_photometry import PhotometryConfig, run_photometry

    pytest.importorskip("tractor_jax")
    jax_res = run_photometry(
        synth_field["cutouts_dir"], synth_field["catalog"],
        PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                         prefetch="sync", solver="linear", pad_bucket=0),
        progress=False)
    cpu_res = run_photometry(
        synth_field["cutouts_dir"], synth_field["catalog"],
        PhotometryConfig(backend="cpu-tractor", solver="linear",
                         psf_zone_interp=True, psf_core_shift=True),
        progress=False)

    def flux(res, sid):
        return float(np.mean(res["flux"][res["id"] == sid]))

    for sid in (1, 2):
        assert flux(cpu_res, sid) == pytest.approx(flux(jax_res, sid), rel=0.01)


def test_oversampled_radius_is_native_units():
    """getRadius() must be NATIVE px. The parent sets stamp-px hypot (36 for
    51x51@5x); consumers (galaxy patch halfsize) treat it as native, which
    inflated every galaxy patch by ~+29 px/side and made the full-depth
    forced solve ~90 s instead of ~5 s."""
    img = np.zeros((51, 51), np.float32)
    img[25, 25] = 1.0
    p = OversampledPixelizedPSF(img, sampling=0.2)
    assert p.getRadius() == pytest.approx(np.hypot(25.5, 25.5) * 0.2, rel=1e-6)
