# FAQ & troubleshooting

Common errors, and how to read what comes out of the pipeline.

## Interpreting results

### Some of my fluxes are negative — is that a bug?

No, and you should keep them. The default `eigfloor` estimator is **sign-free**:
a source fainter than the noise scatters symmetrically around its true flux, so
roughly half of the undetected sources come out negative. Clipping or discarding
them biases the faint end high, which is exactly what breaks stacking, mean
fluxes, and photo-z at low S/N.

Average the negatives in like any other measurement (`bin_spectrum` does this
with inverse-variance weights). If you *want* non-negative fluxes for a handful
of bright targets, that is what `lasso` is for — read the coverage caveat in
{doc}`solvers` first.

### What exactly is `flux_err`?

The 1σ from the solver's Fisher information for that source in that visit,
propagated through the same unit scaling as the flux (mJy). For `eigfloor` at
`fp32` these are calibrated — the pull distribution `(f − f_true)/σ` has NMAD
≈ 0.95–0.99 on real data. `lasso` posteriors under-cover by construction; do not
use its errors for population statistics.

`flux_err` covers photon/read noise and the deblending covariance with
neighbours. It does *not* include absolute calibration, PSF-model error, or
background systematics ({doc}`backgrounds_systematics`).

### My source scatters more between visits than its error bars suggest

Check in this order:

1. **Real variability or a moving object** — plot flux against `obs_id`/time.
2. **Wavelength, not noise.** Different visits sample different wavelengths, so a
   source with a strong SED feature *should* move. Plot against
   `central_wavelength` before concluding anything.
3. **A neighbour is missing from your catalog.** Unmodelled flux lands on
   whichever fitted source is closest, and how much it lands depends on the
   dither. Run {func}`~spherex_photometry.diagnostics.plot_fit` on a couple of
   cutouts — an unmodelled neighbour shows up as a bright blob in `chi`.
4. **Background model.** Try `bkg_model="cwave+photutils"` if the field is near
   an airglow line.

### One of my galaxies came out near zero (or far too bright)

Check whether the catalog **shredded** it: optical catalogs sometimes list a
large galaxy's substructure (star-forming knots, the bulge, disc pieces) as
independent sources. Those entries are artifacts — their positions and shapes
describe fragments of one object, not real sources — so the fit divides the
galaxy's light among spurious components. The main entry can scatter around
zero while the fragments absorb its flux, and even a *real* source sitting
among the fragments can be biased. Note this is not about blending: deblending
real sources that share a pixel is exactly what the joint fit is designed to
do; the problem is spurious entries in the source list.

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

sc = SkyCoord(cat["ra"], cat["dec"], unit="deg")
_, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)
print(cat[sep < 6.15 * u.arcsec])      # entries sharing a pixel with another
```

An optical thumbnail separates real blends (fine) from shredding (several
entries inside one extended galaxy). For shredded groups: drop the fragment
entries, keep one entry with the galaxy's overall shape, and refit; summing the
group's fitted fluxes only recovers the total. {ref}`The gallery
<fragmentation>` shows a real A2537 example. Every estimator behaves this way —
it is the catalog, not the solver.

### Why is `central_wavelength` NaN for some rows?

That cutout shipped an empty `CWAVE` extension. The source is still photometered
(the flux is fine), but it cannot be placed in a spectrum, so
{func}`~spherex_photometry.spectra.build_spectra` drops those points. The
pipeline logs a warning listing the affected cutouts. Re-retrieving with
`include_wavelength=True` usually fixes it.

### How do I get magnitudes?

```python
from spherex_photometry import to_ab_mag
mag, mag_err = to_ab_mag(phot["flux"], phot["flux_err"])   # flux in mJy
```

Negative fluxes give NaN magnitudes — another reason to do science in flux space
and convert only for display.

## Errors and warnings

### `RuntimeWarning: setup_device(device='cpu') was called after jax was already imported`

JAX picks its backend at import time, so `device="cpu"` has to be set before
anything imports `jax`. Either use the CLI (`spherex-phot run --device cpu`,
which handles this before importing the backend) or call
{func}`~spherex_photometry.device.setup_device` at the very top of your script,
before importing anything that pulls in `tractor_jax`. Setting
`JAX_PLATFORMS=cpu` in the environment always works.

### GPU out of memory / my job killed someone else's

JAX preallocates ~75% of the card by default. Use:

```python
PhotometryConfig(gpu_preallocate=False, gpu_mem_fraction=0.45, tile_chunk=8)
```

`gpu_mem_fraction` caps the pool, and `tile_chunk` bounds the transient of a
single cutout's solve without changing the output. See {doc}`hardware`.

### `ImportError: The 'jax' backend needs the tractor-jax engine`

`tractor-jax` is not on PyPI yet:

```bash
pip install git+https://github.com/hbahk/tractor-jax
```

### `ImportError: The 'cpu-tractor' backend needs the upstream Tractor package`

Install it from source (`pip install git+https://github.com/dstndstn/tractor`),
or — usually easier — just use the JAX engine on CPU with `device="cpu"`, which
supports every solver. See {doc}`cpu_backend`.

### `ConfigError: backend='cpu-tractor' supports only solver='linear'`

The upstream Tractor does a single weighted-least-squares solve; it has no
eigenvalue-floor, LASSO, or SED-prior estimator. Use `backend="jax",
device="cpu"` if you want those without a GPU.

### `SystemExit: No cutouts found under ...`

Point `--cutouts-dir` at the directory containing `cutout_*.fits` (the one
`retrieve` wrote, which also has `summary.ecsv`). Files whose `summary.ecsv`
status is not `"ok"` are skipped by design.

### A few cutouts failed and were skipped

The pipeline logs `N cutouts failed and were skipped: [...]` and keeps going —
one bad file does not lose the field. The traceback for each is in the log. If
you fix the cause, re-run with `resume=True` and the same `output` to fill in
only the missing cutouts.

## Practical questions

### Can I run without a GPU?

Yes, and `PhotometryConfig(backend="cpu-tractor")` is the one to reach for: the
classic Tractor, tiled on the same grid as the GPU path, `linear` only.
`PhotometryConfig(device="cpu")` runs the same JAX engine and all four solvers on
CPU — numerically the same code path as the GPU, but markedly slower — so use it
for cross-checks and for the solvers `cpu-tractor` lacks ({doc}`cpu_backend`).

### Can I use my own catalog instead of Legacy Survey?

Yes — you need `id`, `ra`, `dec`, and optionally shape columns; see
{doc}`catalogs`. Anything not in your catalog is not modelled, so include the
neighbours, not just your targets.

### How do I resume an interrupted run?

Pass the same `output` path with `resume=True` (CLI: `--resume`). Cutouts already
present in the file are skipped and new rows are appended.

### How do I reproduce a run later?

The output parquet carries the full config as JSON in its metadata:

```python
from spherex_photometry.io.output import read_photometry
tab, meta = read_photometry("phot.parquet")
print(meta["spherex_photometry.config"])
```

You can also keep the config as a file (`cfg.to_yaml("run.yaml")`,
`spherex-phot run --config run.yaml`).

### Can I try the package before I have any data?

Yes — {doc}`worked_example` simulates a field, photometers it, and plots the
spectra with no network access at all (`examples/03_offline_demo.py`).
