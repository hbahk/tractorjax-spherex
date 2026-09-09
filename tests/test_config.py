import pytest

from tractorjax_spherex.config import ConfigError, PhotometryConfig


def test_defaults_are_blind_production():
    c = PhotometryConfig()
    assert c.solver == "eigfloor"
    assert c.tile_size == 15 and c.tile_halo == 3 and c.pad_bucket == 32
    assert c.precision == "fp32" and c.prefetch == "thread"
    assert c.bkg_model == "cwave+photutils"
    assert c.psf_zone_interp is True and c.psf_core_shift is True
    assert c.solver_spec() == {"kind": "eigfloor", "floor": 1e-2}


def test_jax_core_shift_requires_zone_interp():
    """The shifts ride on the zone basis on the JAX backend, so this pair is
    rejected at config time rather than one cutout into the run. Reachable by
    accident since psf_core_shift became a default: a user who only turns zone
    interpolation off would otherwise hit it deep in the backend."""
    with pytest.raises(ConfigError):
        PhotometryConfig(backend="jax", psf_zone_interp=False)
    # opting out of both is fine, and cpu-tractor shifts the stamp standalone
    PhotometryConfig(backend="jax", psf_zone_interp=False, psf_core_shift=False)
    PhotometryConfig(backend="cpu-tractor", solver="linear",
                     psf_zone_interp=False, psf_core_shift=True)


def test_cpu_tractor_rejects_nonlinear_solver():
    for solver in ("eigfloor", "lasso", "eigfloor_prior"):
        with pytest.raises(ConfigError):
            PhotometryConfig(backend="cpu-tractor", solver=solver)
    PhotometryConfig(backend="cpu-tractor", solver="linear")  # ok


def test_bad_enums_raise():
    for kw in (dict(solver="nope"), dict(backend="nope"), dict(bkg_model="nope"),
               dict(device="nope"), dict(precision="nope"), dict(prefetch="nope")):
        with pytest.raises(ConfigError):
            PhotometryConfig(**kw)


def test_resolved_caps_policy():
    # pad_bucket set (F3 default) -> caps off
    assert PhotometryConfig(pad_bucket=32).resolved_caps() == (None, None, None)
    # full depth, no pad_bucket -> field caps on
    assert PhotometryConfig(pad_bucket=0, fit_zmag_max=None).resolved_caps() \
        == (112, 352, 9)
    # z-cut -> caps off
    assert PhotometryConfig(pad_bucket=0, fit_zmag_max=21.0).resolved_caps() \
        == (None, None, None)
    # explicit override wins; 0 disables
    assert PhotometryConfig(pad_bucket=0, max_ps_cap=50).resolved_caps()[0] == 50
    assert PhotometryConfig(pad_bucket=0, max_gal_cap=0).resolved_caps()[1] is None


def test_lasso_solver_spec():
    c = PhotometryConfig(solver="lasso", lasso_alpha="auto")
    spec = c.solver_spec()
    assert spec["kind"] == "lasso" and spec["alpha"] == "auto"
    c2 = PhotometryConfig(solver="lasso", lasso_alpha="3.0")
    assert c2.solver_spec()["alpha"] == 3.0


def test_yaml_and_toml_round_trip(tmp_path):
    c = PhotometryConfig(solver="lasso", tile_halo=2, gpu_mem_fraction=0.45)
    p = tmp_path / "cfg.yaml"
    c.to_yaml(p)
    c2 = PhotometryConfig.from_file(p)
    assert c2.solver == "lasso" and c2.tile_halo == 2 and c2.gpu_mem_fraction == 0.45

    ptoml = tmp_path / "cfg.toml"
    ptoml.write_text('solver = "eigfloor"\ntile_size = 21\n')
    c3 = PhotometryConfig.from_file(ptoml)
    assert c3.solver == "eigfloor" and c3.tile_size == 21


def test_unknown_key_rejected():
    with pytest.raises(ConfigError):
        PhotometryConfig.from_dict({"solver": "linear", "bogus": 1})
