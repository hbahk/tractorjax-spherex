# Quickstart

From a sky position to a SPHEREx spectrum in four steps. The full script is
`examples/01_quickstart_end_to_end.py`.

## 1. Download the L2 cutouts

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from tractorjax_spherex import retrieve

retrieve(SkyCoord(150.0*u.deg, 2.0*u.deg), 100, output_dir="cutouts",
         include_wavelength=True, include_sapm=True)
```

This writes one multi-extension FITS per overlapping SPHEREx pointing (plus a
`summary.ecsv`) into `cutouts/`. `include_wavelength` and `include_sapm` add the
per-pixel wavelength maps and the solid-angle map the pipeline needs — keep them
on. From the shell:

```bash
tractorjax-spherex retrieve --ra 150.0 --dec 2.0 --out cutouts
```

## 2. Get a reference catalog

Positions and shapes are fixed from a reference catalog. Fetch Legacy Survey
DR10 (needs the `[catalog]` extra):

```python
from tractorjax_spherex import fetch_ls_dr10
fetch_ls_dr10(150.0, 2.0, out="catalog.parquet")
```

Or bring your own — see {doc}`catalogs` for the required columns.

## 3. Run forced photometry

```python
from tractorjax_spherex import PhotometryConfig, run_photometry

cfg = PhotometryConfig()      # configuration of record: eigfloor, m_z < 21
phot = run_photometry("cutouts", "catalog.parquet", cfg,
                      target=(150.0, 2.0), output="phot.parquet")
```

The defaults are the tuned setup (estimator, catalog depth, background, PSF
handling); a first run should not change them, see {doc}`configuration`.
`phot` has one row per (source, visit): `cutout_index, obs_id, detector, id, ra,
dec, central_wavelength, bandwidth, flux, flux_err` (flux in mJy). On a shared
GPU add `gpu_preallocate=False, gpu_mem_fraction=0.45`; on a machine with no GPU
add `device="cpu"`.

From the shell:

```bash
tractorjax-spherex run --cutouts-dir cutouts --catalog catalog.parquet \
    --ra 150.0 --dec 2.0 --output phot.parquet
```

## 4. Assemble spectra

```python
from tractorjax_spherex import build_spectra, bin_to_channels
from tractorjax_spherex.spectra import plot_spectrum

spectra = build_spectra(phot)          # {id: table sorted by wavelength}
spec = spectra[next(iter(spectra))]
ax = plot_spectrum(spec, binned=bin_to_channels(spec))
ax.figure.savefig("spectrum.png", dpi=150)
```

`build_spectra` groups the per-visit points by source and sorts them by
wavelength (which transparently handles the within-detector wavelength reversal);
`bin_to_channels` combines repeat visits with inverse-variance weighting on the
102 SPHEREx spectral channels (17 per detector band, constant resolving power;
`spherex_channels()` lists them). `bin_spectrum(spec, dlam=...)` does the same
on a uniform wavelength grid. On the CLI: `tractorjax-spherex spectra ...
--channels`.

## Next steps

- {doc}`cluster_example` — the same four steps on a real, crowded cluster
  field (Abell 2537), with the box sizing, the fit inspection and the spectra
  of blended sources worked through.
- {doc}`solvers` — pick the estimator for your science.
- {doc}`hardware` — GPU-memory sizing and CPU-only guidance.
- {doc}`configuration` — the full option reference.
