# Worked example (offline)

A complete, **self-contained** run you can execute right after installing — no
network, no GPU, no IRSA account. The field is simulated with the package's toy
simulator ({mod}`spherex_photometry.simulate`), so you see the whole pipeline
(catalog → forced photometry → fit inspection → spectrum) and learn the API
before touching real data. The full script is `examples/03_offline_demo.py`;
swap step 1 for {func}`~spherex_photometry.retrieve` + a real catalog to do the
same thing on real SPHEREx L2 cutouts.

## 1. Define sample sources and simulate a field

```python
from spherex_photometry.simulate import make_synth_field, make_synth_catalog
from spherex_photometry.io.cutouts import read_cutout

truth = [
    {"x": 12.0, "y": 14.0, "flux_mjy": 5.0},                              # point
    {"x": 27.0, "y": 24.0, "flux_mjy": 2.0},                              # point
    {"x": 18.0, "y": 30.0, "flux_mjy": 1.2, "shape_r": 1.5, "sersic": 1.0},  # galaxy
]
make_synth_field("cutouts", n_cutouts=6, sources=truth, seed=42,
                 noise_mjy_sr=0.02)

cutout0 = read_cutout(sorted(Path("cutouts").glob("cutout_*.fits"))[0])
make_synth_catalog("catalog.parquet", truth, cutout0.wcs)
```

Six cutouts play the role of six SPHEREx visits, each sampling a different
wavelength band — exactly the shape of real retrieved data (same MEF layout,
`summary.ecsv` included).

## 2. Run forced photometry

```python
from spherex_photometry import PhotometryConfig, run_photometry

cfg = PhotometryConfig(solver="eigfloor", device="cpu", prefetch="sync")
phot = run_photometry("cutouts", "catalog.parquet", cfg, output="phot.parquet")
```

`phot` has one row per (source, visit) — 18 rows here.

## 3. Inspect the fit: data / model / chi

```python
from spherex_photometry.diagnostics import plot_fit
from spherex_photometry.io.catalogs import load_catalog

fig = plot_fit(cutout0, load_catalog("catalog.parquet"), phot,
               cutout_index=0, config=cfg)
fig.savefig("fit_comparison.png", dpi=150)
```

```{image} _static/fit_comparison.png
:alt: data / fitted model / chi triptych for one synthetic cutout
:width: 100%
```

Read it left to right: the background-subtracted **data**, the **fitted model**
(every catalog source rendered at its measured flux through the same
5×-oversampled PSF the solver used), and **chi** = (data − model)/σ. A good fit
leaves chi as structureless noise, as here. A red/blue blob centred on a source
means its flux, shape, or position is off; coherent large-scale chi structure
suggests trying a different `bkg_model` ({doc}`backgrounds_systematics`).

## 4. Spectra vs the injected truth

```python
from spherex_photometry import build_spectra
from spherex_photometry.spectra import bin_spectrum

spectra = build_spectra(phot)      # {id: table sorted by wavelength}
```

```{image} _static/spectra_vs_truth.png
:alt: measured spectrophotometry vs injected truth for the three sources
:width: 100%
```

Each point is one visit's forced flux at that visit's wavelength; dashed lines
are the injected truths. The per-point error bars come from the solver's Fisher
variances and match the visit-to-visit scatter.

Recovered fluxes land within ~1–2σ of truth with a few-percent offset — expected
here because the *toy* simulator injects sources as pixel-integrated Gaussians
(an approximation of the render-then-bin model, and the "galaxy" is just a
broadened Gaussian rather than a true Sérsic). On real data the package's
accuracy is established differently: its output is bit-identical to the
validated production driver (see the project notes), and the estimator
calibrations quoted in {doc}`solvers` apply.

## Where to go next

- Run the same four steps on real data: {doc}`quickstart`.
- Choose the right estimator for your science: {doc}`solvers`.
- Understand each processing stage: {doc}`how_it_works`.
