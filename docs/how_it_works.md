# How it works

What actually happens between "a directory of cutouts" and "a spectrum", which
module owns each step, and why the pipeline is shaped this way. Useful when you
need to debug a result, tune an option, or extend the package — see
{doc}`data_model` for the units and file formats and {doc}`solvers` for the
estimator itself.

## The pipeline at a glance

```text
spherex_retrieval.retrieve            L2 cutout MEFs + summary.ecsv
        │
        ▼
  discover_cutouts / filter_ok        keep status=="ok"      io/cutouts.py
        │
        ├─ catalog once per run:      normalize → depth cut → protect set
        │                                                    io/catalogs.py
        ▼   ┌──────────────── per cutout ────────────────┐
  read_cutout                         IMAGE/FLAGS/VAR/ZODI/PSF/CWAVE/SAPM
        │                                                    io/cutouts.py
        ▼
  prepare_pixels                      background fit (ZODI + model)  background.py
        │                             MJy/sr → mJy/pixel  (× Ω_SAPM × 1e9)
        │                             invvar = 1/var, zeroed on MASKBITS
        │                                                    prepare.py
        ▼
  project_sources / select_psf_native RA,Dec → pixels; PSF zone → 5× stamp
        │                                                    prepare.py
        ▼
  build_cutout_tiles                  15-px cores + 3-px halo
        │
        ▼
  extract_tiled_batches               padded, vmap-ready batches
        │                                          backends/jax_backend.py
        ▼
  solve                               one vmapped linear solve over all tiles
        │                                     (tractor_jax.jax.batching)
        ▼
  extract                             read each source from its CORE tile;
        │                             sample CWAVE/CBAND at its own pixel
        └──────────────────────────────────────────┘
        ▼
  make_table / write_photometry       one row per (source, visit)  io/output.py
        │
        ▼
  build_spectra                       group by id, sort by wavelength  spectra.py
```

## Stage by stage

### 1. Cutouts in

`spherex_retrieval.retrieve` writes one multi-extension FITS per overlapping
SPHEREx pointing plus a `summary.ecsv`.
{func}`~spherex_photometry.io.cutouts.discover_cutouts` finds them and
{func}`~spherex_photometry.io.cutouts.filter_ok` keeps only the ones the
retrieval marked `status == "ok"`.
{func}`~spherex_photometry.io.cutouts.read_cutout` parses one MEF into a
{class}`~spherex_photometry.io.cutouts.Cutout`, guarding the
present-but-empty CWAVE/CBAND/SAPM extensions some cutouts ship.

### 2. Catalog, once per run

The fit is *reference-catalog forced photometry*: positions and shapes are read
from your catalog and **never fitted** — only fluxes are solved.
{func}`~spherex_photometry.io.catalogs.normalize_catalog` fills the canonical
columns (including `shape_ab` / `shape_phi` from the ellipticities),
{func}`~spherex_photometry.io.catalogs.apply_depth_cut` optionally prunes to
sources SPHEREx can constrain, and — for `lasso` / `eigfloor_prior` —
{func}`~spherex_photometry.io.catalogs.protected_indices` marks the bright
sources that must stay unpenalized. See {doc}`catalogs`.

### 3. Pixels: background, units, masking

{func}`~spherex_photometry.prepare.prepare_pixels` turns the raw extensions into
what the solver actually sees:

1. **Background** — the ZODI extension is the base; `bkg_model` refines it
   ({doc}`backgrounds_systematics`).
2. **Units** — `data = (IMAGE − background) × Ω × 1e9`, where Ω is the per-pixel
   solid angle from SAPM. That converts MJy/sr to **mJy per pixel**, so fitted
   fluxes come out in mJy directly.
3. **Weights** — `invvar = 1/variance` (scaled the same way), zeroed wherever the
   FLAGS extension hits `MASKBITS` or the variance is invalid. Masked pixels
   simply carry no weight in the solve.

This stage is backend-neutral: the JAX and CPU backends consume identical
pixels, which is what makes their agreement a meaningful cross-check.

### 4. Sources and PSF

{func}`~spherex_photometry.prepare.project_sources` maps catalog RA/Dec to cutout
pixels through the cutout WCS.
{func}`~spherex_photometry.prepare.select_psf_native` picks the PSF-cube plane
for the zone the source sits in (the SPHEREx PSF varies across the detector) and
downsamples the delivered 10×-oversampled plane to a **5× stamp**.

**Why oversampled?** SPHEREx pixels are 6.15″ — the PSF is undersampled. If you
convolved the PSF with a source at native resolution you would bias the flux. So
every source is rendered on the 5× grid and *then* summed into native pixels
(`psf_sampling=0.2`, `fixed_max_factor=5`). Both backends do this; the CPU one
needs {class}`~spherex_photometry.backends.cpu_psf.OversampledPixelizedPSF`
because the stock Tractor class mis-normalizes oversampled stamps
({doc}`cpu_backend`).

### 5. Tiling and the batched solve

A 100×100 cutout at full catalog depth has far too many sources to solve as one
dense system, and most source pairs do not overlap. The cutout is therefore split
into `tile_size` (15 px) **cores**, each padded by a `tile_halo` (3 px) so a
source just outside a core still contributes its PSF wings. Every tile is an
independent linear solve; all tiles of a cutout are padded to a common shape and
run in **one `vmap`** on the GPU.

`pad_bucket` rounds those shapes to a multiple (fewer XLA compiles, near-natural
matrix sizes) and `tile_chunk` splits the tile axis to bound peak GPU memory —
both are output-preserving ({doc}`hardware`).

The grid itself lives in {mod}`spherex_photometry.tiling`, not in either backend:
the `cpu-tractor` backend walks the same tiles and runs one upstream
`optimize_forced_photometry` per tile (`cpu_tiling=True`, the default), so the
two engines are compared on one geometry rather than two ({doc}`cpu_backend`).

### 6. Extraction

Because tiles overlap in their halos, a source could be measured twice. It is
read from **the tile whose core box contains it**, so every source is counted
exactly once. Its wavelength and bandwidth are sampled from the CWAVE/CBAND maps
**at its own pixel** — the wavelength drifts across a cutout, so the cutout-center
value would mislabel off-axis sources.

### 7. Output and spectra

{func}`~spherex_photometry.io.output.write_photometry` writes one row per
(source, visit) with the config and schema version in the parquet metadata.
{func}`~spherex_photometry.spectra.build_spectra` groups those rows by source and
sorts by wavelength — which is also what makes the within-detector wavelength
reversal a non-issue — and
{func}`~spherex_photometry.spectra.bin_spectrum` combines repeat visits with
inverse-variance weights.

## Why the backend has three stages

A backend implements `build` → `solve` → `extract` rather than one function
(`backends/base.py`). `build` is pure CPU work (FITS I/O, background fitting,
tiling, batch assembly) and `solve` is the GPU part. Splitting them lets
{func}`~spherex_photometry.pipeline.run_photometry` run `build` for cutout *N+1*
in a worker thread while the GPU solves cutout *N* (`prefetch="thread"`), so the
CPU stage is hidden behind the GPU stage. Set `prefetch="sync"` to disable that
when debugging.

The same three-stage protocol is what lets the `cpu-tractor` backend slot in
behind the identical interface and produce the identical output schema.

## Where to look when something is off

| symptom | look at |
|---|---|
| structured residuals across the whole cutout | `bkg_model` ({doc}`backgrounds_systematics`) |
| one source's residual is a bright/dark blob | its catalog shape/position; {doc}`worked_example` |
| fluxes shift when tiling changes | degenerate `linear` solver at depth ({doc}`solvers`) |
| GPU out of memory | `tile_chunk`, `gpu_mem_fraction` ({doc}`hardware`) |
| wavelength is NaN | the cutout shipped an empty CWAVE extension ({doc}`faq`) |

{func}`~spherex_photometry.diagnostics.plot_fit` renders the data / model / chi
triptych for any cutout so you can see which of these you have.
