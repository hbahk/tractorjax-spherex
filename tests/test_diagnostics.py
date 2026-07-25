import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

pytest.importorskip("tractor_jax")

from spherex_photometry import PhotometryConfig, run_photometry
from spherex_photometry.diagnostics import (fluxes_for_cutout, plot_fit,
                                            render_model_image)
from spherex_photometry.io.catalogs import load_catalog
from spherex_photometry.io.cutouts import read_cutout


def _run(field):
    cfg = PhotometryConfig(backend="jax", device="cpu", precision="fp64",
                           prefetch="sync", solver="linear", pad_bucket=0)
    return run_photometry(field["cutouts_dir"], field["catalog"], cfg,
                          progress=False), cfg


def test_fluxes_for_cutout_selects_one_cutout(synth_field):
    phot, _ = _run(synth_field)
    fluxes = fluxes_for_cutout(phot, 0)
    assert set(fluxes) == {1, 2}
    assert all(np.isfinite(v) for v in fluxes.values())


def test_render_model_image_reproduces_the_sources(synth_field):
    phot, cfg = _run(synth_field)
    cutout = read_cutout(sorted(synth_field["cutouts_dir"].glob("*.fits"))[0])
    catalog = load_catalog(synth_field["catalog"])
    model, prepared = render_model_image(cutout, catalog,
                                         fluxes_for_cutout(phot, 0), cfg)

    assert model.shape == prepared.data.shape
    assert np.all(np.isfinite(model))
    # the model must carry the fitted flux and sit where the sources are
    assert model.sum() == pytest.approx(sum(fluxes_for_cutout(phot, 0).values()),
                                        rel=0.05)
    for src in synth_field["sources"]:
        y, x = int(round(src["y"])), int(round(src["x"]))
        assert model[y, x] > 0.1 * model.max()
    # residuals should be small where the model is good
    chi = (prepared.data - model) * np.sqrt(np.maximum(prepared.invvar, 0.0))
    assert np.abs(np.median(chi)) < 1.0


def test_plot_fit_returns_triptych(synth_field):
    phot, cfg = _run(synth_field)
    cutout = read_cutout(sorted(synth_field["cutouts_dir"].glob("*.fits"))[0])
    fig = plot_fit(cutout, load_catalog(synth_field["catalog"]), phot,
                   cutout_index=0, config=cfg)
    assert len(fig.axes) >= 3          # 3 panels (+ colorbars)
    matplotlib.pyplot.close(fig)
