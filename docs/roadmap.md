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

---

None of these blocks the v0.1 product. Each is an additive layer on the same
forced-photometry core, and the deferred data-dependent items (secondary-catalog
conversion, simulator paths) unblock the moment the corresponding SPHEREx
products become public.
