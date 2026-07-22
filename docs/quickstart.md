# Quickstart

From a sky position to a SPHEREx spectrum in four steps. The full script is
`examples/01_quickstart_end_to_end.py`.

## 1. Download the L2 cutouts

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from spherex_photometry import retrieve

retrieve(SkyCoord(150.0*u.deg, 2.0*u.deg), 100, output_dir="cutouts",
         include_wavelength=True, include_sapm=True)
```

This writes one multi-extension FITS per overlapping SPHEREx pointing (plus a
`summary.ecsv`) into `cutouts/`. `include_wavelength` and `include_sapm` add the
per-pixel wavelength maps and the solid-angle map the pipeline needs — keep them
on. From the shell:

```bash
spherex-phot retrieve --ra 150.0 --dec 2.0 --out cutouts
```

## 2. Get a reference catalog

Positions and shapes are fixed from a reference catalog. Fetch Legacy Survey
DR10 (needs the `[catalog]` extra):

```python
from spherex_photometry import fetch_ls_dr10
fetch_ls_dr10(150.0, 2.0, out="catalog.parquet")
```

Or bring your own — see {doc}`catalogs` for the required columns.

## 3. Run forced photometry

```python
from spherex_photometry import PhotometryConfig, run_photometry

cfg = PhotometryConfig(solver="eigfloor")          # blind-production default
phot = run_photometry("cutouts", "catalog.parquet", cfg,
                      target=(150.0, 2.0), output="phot.parquet")
```

`phot` has one row per (source, visit): `cutout_index, obs_id, detector, id, ra,
dec, central_wavelength, bandwidth, flux, flux_err` (flux in mJy). On a shared
GPU add `gpu_preallocate=False, gpu_mem_fraction=0.45`; on a machine with no GPU
add `device="cpu"`.

From the shell:

```bash
spherex-phot run --cutouts-dir cutouts --catalog catalog.parquet \
    --solver eigfloor --ra 150.0 --dec 2.0 --output phot.parquet
```

## 4. Assemble spectra

```python
from spherex_photometry import build_spectra, bin_spectrum
from spherex_photometry.spectra import plot_spectrum

spectra = build_spectra(phot)          # {id: table sorted by wavelength}
spec = spectra[next(iter(spectra))]
ax = plot_spectrum(spec, binned=bin_spectrum(spec, dlam=0.05))
ax.figure.savefig("spectrum.png", dpi=150)
```

`build_spectra` groups the per-visit points by source and sorts them by
wavelength (which transparently handles the within-detector wavelength reversal);
`bin_spectrum` combines repeat visits with inverse-variance weighting.

## Next steps

- {doc}`solvers` — pick the estimator for your science.
- {doc}`hardware` — GPU-memory sizing and CPU-only guidance.
- {doc}`configuration` — the full option reference.
