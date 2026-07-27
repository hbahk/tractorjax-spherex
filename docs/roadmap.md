# Roadmap

`spherex-photometry` v0.1 ships one thing well: reference-catalog forced
photometry and spectrum assembly from SPHEREx L2 cutouts. Source positions and
shapes come from the input catalog and are held fixed; only fluxes are solved
(see {doc}`solvers`). A few capabilities that live in the research code, or that
depend on data SPHEREx has not yet released publicly, are intentionally left out
of this release. This page records what and why.

## Deconfusion image models

The `eigfloor_prior` solver already carries the *predictive* half of
deconfusion: {func}`spherex_photometry.priors.predict_flux_ujy` builds a per-source
Legacy-Survey SED and evaluates it at each cutout's wavelength, and
{func}`spherex_photometry.priors.make_prior_context` turns that prediction into
the Gaussian flux prior that constrains the faint, penalized nuisance sources
(see {doc}`solvers`). **What is deferred is the image-model rendering** — using
those SED predictions to render the sub-catalog of faint confusing sources into
a model image and subtract it as a confusion pedestal before the solve. v0.1
constrains faint-source confusion in flux space (via the prior) rather than in
image space (via a rendered, subtracted pedestal); the rendered-pedestal path is
a later release.

## Free-parameter (LM) fitting

Every v0.1 estimator solves the *linear* forced-photometry system with positions
and shapes frozen. The underlying `tractor_jax` engine also has a
Levenberg–Marquardt optimizer (`JaxOptimizer`) that can thaw positions and shapes
and fit them jointly with flux. Exposing that as an **optional, non-forced mode**
— thaw a chosen subset of parameters and refine the catalog geometry against the
SPHEREx data — is deferred. It is an add-on to the forced core, not a change to
it: the blind-production default stays fully forced and linear, and forced
photometry remains the right tool at SPHEREx's undersampled resolution for most
fields.

## Secondary (channel / collapsed) catalog conversion

v0.1 produces the *per-visit* product: one `(wavelength, flux, flux_err)` point
per source per cutout, assembled into spectra by
{func}`spherex_photometry.spectra.build_spectra` (see {doc}`data_model`).
Converting those points into a SPHEREx-style **secondary catalog** — fluxes
collapsed onto the fixed spectral channels / points — needs the per-point (per
spectral-channel) **filter response curves** to integrate each spectrum against.
Those are not public yet, so this conversion is deferred until they are. In the
meantime, {func}`spherex_photometry.spectra.bin_spectrum` gives an
inverse-variance-weighted rebinning onto user-chosen wavelength bins as a
stand-in.

## SPHEREx sky-simulator paths

The photometry engine renders against the **retrieved L2 PSF cube**, not a
forward instrument model (see {doc}`cpu_backend`). Paths that would drive
rendering or validation from the SPHEREx sky / instrument simulator are deferred
until that simulator is publicly available; v0.1 depends only on released L2
data products.

## Automatic batch-width sizing (planned)

The fixed batch-width caps (`max_ps_cap` / `max_gal_cap`, defaults
`MAX_PS_CAP = 112` / `MAX_GAL_CAP = 352`) exist so the jitted solver compiles
once per run instead of once per distinct tile shape. They are applied
automatically to full-depth fits when `pad_bucket` is off — and their defaults
were sized on a single sparse field (A2537), so denser fields overflow them.
A tile that overflows takes its **entire cutout** out of the product: the
pipeline logs the traceback, skips the cutout, and reports the count in one
warning at the end, while still exiting 0 and writing a parquet. An incomplete
product is therefore indistinguishable from a complete one without reading the
log. Measured occupancies for eight real fields are tabulated in
{doc}`configuration` — half of them exceed a default, COSMOS by 2.4×.

`pad_bucket=32` (the v0.1 default) already sidesteps this by turning the caps
off and sizing each batch near its natural width, so the shipped default is
safe. What is deferred is making the *capped* path safe as well:

1. **Pre-scan and auto-size.** Tile source assignment is pure geometry — it
   needs only catalog positions, `shape_r`, and the cutout WCS, not the PSF or
   any solve. A cheap pass over the cutouts can measure the true occupancy and
   set the caps from it, so `max_ps_cap="auto"` becomes the default and no
   field-specific tuning is needed.
2. **Warn loudly, or fail closed.** Until then the cap overflow should raise a
   dedicated exception carrying the required width, and the run summary should
   make partial products obvious — a nonzero exit code, or a `complete=False`
   flag written into the output metadata, rather than a log line a caller can
   miss.

Both are small, self-contained changes to `PhotometryConfig.resolved_caps` and
`pipeline`; they are queued rather than shipped because the default path does
not hit the failure.

---

None of these blocks the v0.1 product. Each is an additive layer on the same
forced-photometry core, and the deferred data-dependent items (secondary-catalog
conversion, simulator paths) unblock the moment the corresponding SPHEREx
products become public.
