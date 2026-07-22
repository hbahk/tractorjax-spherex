# spherex-photometry

Forced photometry and spectrophotometry from **SPHEREx L2 spectral images**, on
your own hardware (GPU or CPU).

`spherex-photometry` takes you from a sky position to a per-source SPHEREx
spectrum: download the L2 cutouts, supply a reference catalog, run
reference-catalog forced photometry with a joint deblending solve, and assemble
the per-visit measurements into spectra. It wraps the
[`tractor-jax`](https://github.com/hbahk/tractor-jax) GPU/JAX engine and the
[`spherex-retrieval`](https://github.com/hbahk/spherex-retrieval) cutout
downloader.

## The product

SPHEREx observes each patch of sky many times, each visit through a different
part of the linear variable filter — a different wavelength. Forced photometry on
one cutout yields **one flux at one wavelength** for every catalog source in it.
Collect those across all the visits of a field and you get, per source, a set of
`(wavelength, flux, flux_err)` points — a low-resolution **spectrum**. That
assembly is what {func}`spherex_photometry.spectra.build_spectra` does.

Fluxes are in **mJy**; magnitudes are AB.

## Start here

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
```

```{toctree}
:maxdepth: 1
:caption: Guides

solvers
hardware
data_model
catalogs
backgrounds_systematics
cpu_backend
configuration
roadmap
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```

## At a glance

```python
from spherex_photometry import PhotometryConfig, run_photometry, build_spectra

cfg = PhotometryConfig(solver="eigfloor")     # blind-production default
phot = run_photometry("cutouts", "catalog.parquet", cfg, output="phot.parquet")
spectra = build_spectra(phot)                 # {id: per-source spectrum}
```

See {doc}`solvers` to pick the estimator that matches your science, and
{doc}`hardware` for GPU-memory and CPU-only guidance.
