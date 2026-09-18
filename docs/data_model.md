# Data model

This page describes what a SPHEREx cutout carries, how its pixels become fluxes
in **mJy**, and the schema of the photometry table the pipeline writes. It is the
reference for anyone reading the inputs or the output parquet directly.

## The cutout MEF

Each input is one L2 cutout written by
[`spherex-retrieval`](https://github.com/hbahk/spherex-retrieval) as a
multi-extension FITS file. {func}`tractorjax_spherex.io.cutouts.read_cutout`
parses one into a {class}`~tractorjax_spherex.io.cutouts.Cutout` dataclass (which
also supports `cutout["key"]` / `cutout.get("key")` access). The HDU layout is:

```
PRIMARY  IMAGE  FLAGS  VARIANCE  ZODI  PSF  PSF_ZONES  [CWAVE] [CBAND] [SAPM]
```

| HDU | `Cutout` field | contents |
|---|---|---|
| `PRIMARY` | `primary_header` | `OBSID`, `DETECTOR`, `OVERSAMP`, `PSFSRC`, `PSFKIND`, … |
| `IMAGE` | `image` | L2 surface brightness, **MJy/sr** |
| `FLAGS` | `flags` | per-pixel L2 bitmask (see [FLAGS bits](#flags-bits)) |
| `VARIANCE` | `variance` | per-pixel variance of `IMAGE`, (MJy/sr)² |
| `ZODI` | `zodi` | zodiacal-light model (seeds the background fit) |
| `PSF` | `psf_cube` | oversampled PSF planes, one per detector zone: the QR2 optical cube (10×, 101×101) or the R7 effective PSF (5×, 33×33) |
| `PSF_ZONES` | `psf_zones` | table mapping detector `(x, y)` → PSF-cube `plane_idx` (R7 adds `xwidth`, `ywidth`, `nstar`, `neff`) |
| `CWAVE` | `cwave_map` | per-pixel central wavelength (µm) — *optional* |
| `CBAND` | `cband_map` | per-pixel bandwidth (µm) — *optional* |
| `SAPM` | `sapm` | Solid Angle Pixel Map, arcsec² — *optional* |

The last three extensions are optional and are also guarded against being
present-but-empty (shape `(0,)`): a missing/empty `CWAVE` yields
`cwave_center=None` and `cwave_map=None` (the source is still photometered, just
labelled NaN wavelength), a missing `SAPM` falls back to the WCS pixel area, and
a missing `CBAND` leaves the bandwidth NaN.

`PSFSRC` records where the PSF cube came from. The 121-plane cube inside an L2
file is a per-detector calibration constant, byte-identical to IRSA's
`average_psf` calibration product, so `spherex-retrieval` fetches it once per
detector (`PSFSRC = 'cal:average_psf_D4_…fits'`) instead of re-downloading 5 MB
of it with every cutout; `PSFSRC = 'l2'` marks a cube taken from the L2 file
itself (every retrieval with `psf_source="l2"`, and the sampled cutouts the
default mode downloads in full to check the two against each other). The pixel
values are the same either way and the zone table always comes from the L2 file.
Bundles written before this keyword existed are read unchanged.

`PSFKIND` says what kind of PSF the planes are, and the pipeline renders each
kind the way it must be rendered. `'OPTICAL'` (QR2, pipeline 6.x; also the
value assumed for bundles written before the keyword): the 10× plane with the
detector pixel response deconvolved, which the engine 2×-downsamples to 5×,
re-registers (`psf_core_shift`) and integrates over each native pixel.
`'EPSF'` (QR3 and DR1, pipeline R7): the effective PSF of Anderson & King
(2000), 5× and with the pixel response *included*, which the engine uses as
delivered and samples at the native pixel centres (`pixel_integration="point"`
in `tractor_jax`); integrating it again would apply the pixel window twice
(+1/12 px² of variance, ~30 % in N_eff, a +4–13 % central residual on SPHEREx
stars), and the QR2 core-shift table does not apply to it. `cutout.psf_kind`
is `"optical"` or `"effective"`; `OVERSAMP` is 10 or 5; `EPSFCAL` names the
ePSF calibration source file and `DETCOORD = 'sky'` records that the R7 arrays
and zone centres are in the L2 image orientation for every detector. Both
kinds are normalised to unit sum on their own oversampled grid (`PSFNORM`).

The `IMAGE` header carries the celestial WCS plus `CRPIX1A`/`CRPIX2A` — the
1-based detector positions of the cutout's `(0, 0)` pixel — used to map cutout
pixels back to the native detector for PSF-zone selection.

### Discovering cutouts

{func}`~tractorjax_spherex.io.cutouts.discover_cutouts` scans a directory for
files matching `cutout_<index>_<obs>_D<detector>.fits` and returns
`(cutout_index, path)` pairs sorted by index.
{func}`~tractorjax_spherex.io.cutouts.filter_ok` optionally keeps only the
indices marked `status == "ok"` in a sibling `summary.ecsv` (if that file is
absent, all pairs pass through).

## Flux unit flow

The L2 `IMAGE` is a **surface brightness in MJy/sr**; forced photometry needs a
**flux per pixel**. {func}`tractorjax_spherex.prepare.prepare_pixels` does the
conversion, producing a {class}`~tractorjax_spherex.prepare.PreparedPixels` that
both backends consume identically.

**1. Per-pixel solid angle (`omega_sr`, sr).**
{func}`~tractorjax_spherex.io.cutouts.cutout_pixel_area_sr` prefers the `SAPM`
HDU — the standalone calibration product in arcsec² that already absorbs SIP
distortion — scaled to steradians by `ARCSEC2_TO_SR`. When `SAPM` is absent it
falls back to the single WCS projected pixel area
(`wcs.proj_plane_pixel_area()`) broadcast across the cutout. So `omega_sr` is a
*per-pixel map* under SAPM and a constant otherwise.

**2. Scale to mJy/pixel.** Multiplying MJy/sr by the pixel solid angle gives
MJy/pixel; `IMG_SCALE = 1e9` (MJy → mJy) converts that to **mJy/pixel**, which
also conditions the linear solve. With the ZODI-seeded background `bkg` (refined
per `config.bkg_model`; see {doc}`backgrounds_systematics`):

```python
data      = (img - bkg) * omega_sr * IMG_SCALE          # mJy/pixel
var_scaled = var * omega_sr**2 * IMG_SCALE**2           # (mJy/pixel)²
```

**3. Inverse variance and masking.** `invvar = 1 / var_scaled`, then zeroed
wherever it is non-finite (bad/zero variance) **and** wherever the `FLAGS`
bitmask hits `MASKBITS`. Non-finite `data` pixels are set to `0`. A masked pixel
therefore has zero weight but is not removed, keeping array shapes fixed for the
jitted solver.

Every reported **flux and `flux_err` is in mJy** — the solver works entirely in
the mJy/pixel space above, so no further unit conversion happens on output.

(flags-bits)=
### FLAGS bits and masking

`FLAG_BITS` in {mod}`tractorjax_spherex.constants` names the L2 bit positions.
Two derived masks matter:

- **`MASKBITS`** — the OR of `MASK_FLAGS`
  (`SUR_ERROR`, `PHANMISS`, `NONFUNC`, `MISSING_DATA`, `HOT`, `COLD`,
  `PERSIST`, `OUTLIER`). Pixels hitting any of these get zeroed inverse variance
  and are excluded from photometry.
- **`SOURCE_BIT`** (`SOURCE`, bit 21) — marks detected-source pixels. Used only
  to mask the **background fit**, never the photometry itself.

## Per-source wavelength labelling

SPHEREx's linear variable filter makes the central wavelength a function of
**detector position**, so `CWAVE` is a full per-pixel map, not a scalar. Across a
cutout it varies by roughly 0.2 nm/pixel — about 22 nm over a 100-pixel cutout —
which is far larger than the photometric precision. Labelling every source with a
single cutout wavelength would therefore be wrong.

Instead, each source is labelled at **its own pixel**: the backend bilinearly
samples `cwave_map` and `cband_map` at the source's projected position with
{func}`~tractorjax_spherex.io.cutouts.sample_map_bilinear_vec`, giving that
source's `central_wavelength` and `bandwidth` (both µm). `cwave_center` (the
value at the cutout's center pixel) is retained only as a coarse per-cutout
label; when `CWAVE` is missing the sampled wavelength is `NaN` and the source is
still photometered.

## Output schema

{func}`tractorjax_spherex.io.output.write_photometry` writes a parquet whose
schema is fixed by `COLUMNS` (`SCHEMA_VERSION = 1`). **One row is one
spectrophotometric point: one catalog source measured on one cutout (one
SPHEREx visit / spectral channel)** — i.e. one row per `(source, visit)`.

| column | dtype | unit | source |
|---|---|---|---|
| `cutout_index` | `i8` | — | cutout's index in the field |
| `obs_id` | `U32` | — | `OBSID` from the primary header |
| `detector` | `i4` | — | `DETECTOR` (1–6) |
| `id` | `i8` | — | reference-catalog source id |
| `ra` | `f8` | deg | catalog RA |
| `dec` | `f8` | deg | catalog Dec |
| `central_wavelength` | `f8` | µm | `CWAVE` sampled at the source pixel |
| `bandwidth` | `f8` | µm | `CBAND` sampled at the source pixel |
| `flux` | `f8` | mJy | fitted forced-photometry flux |
| `flux_err` | `f8` | mJy | forward-model 1σ from the solver Fisher information |

Collecting the rows for one `id` across all its visits, ordered by
`central_wavelength`, gives that source's spectrum — what
{func}`tractorjax_spherex.spectra.build_spectra` assembles.

### Reproducibility metadata

`write_photometry` stamps the table `meta` so a run is self-describing:

| key | value |
|---|---|
| `tractorjax_spherex.schema_version` | `SCHEMA_VERSION` (`1`) |
| `tractorjax_spherex.version` | package version |
| `tractorjax_spherex.config` | full {class}`~tractorjax_spherex.config.PhotometryConfig` as JSON (see {doc}`configuration`) |
| `tractorjax_spherex.solver_spec` | resolved solver spec string (see {doc}`solvers`) |

{func}`~tractorjax_spherex.io.output.read_photometry` returns
`(Table, meta_dict)`. For `--resume`,
{func}`~tractorjax_spherex.io.output.existing_cutout_indices` reports which
cutout indices are already written, and
{func}`~tractorjax_spherex.io.output.append_or_merge` vstacks new rows onto the
existing parquet and rewrites it.
