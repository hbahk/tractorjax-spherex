# Real-data gallery

Six bright galaxies in the A2537 field, measured on real SPHEREx L2 cutouts with
the three full-catalog estimators. Left: the Legacy Survey DR10 colour thumbnail
with the catalog entries drawn on it (cyan = the target, white = other fitted
sources, grey dotted = catalog-only) and a 6.15″ SPHEREx-pixel scale bar. Right:
the resulting spectrophotometry, binned to the 102 fiducial channels, with the
LS/WISE broadband fluxes as an independent anchor.

```{image} _static/gallery_a2537.png
:alt: six A2537 galaxies — LS thumbnails and their SPHEREx spectra from three estimators
:width: 100%
```

## What to notice

**The estimators mostly agree — until the fit is hard.** In panels (c), (d), and
(f) all three land on the same spectrum: an isolated, well-resolved galaxy is an
easy fit and the choice of regularizer barely matters. Panel (b) is the
instructive one: `linear` (orange) swings wildly — ±10 mJy excursions on a 4 mJy
source — while `lasso` and `eigfloor_prior` stay smooth. That is the degeneracy
of an unregularized fit at full catalog depth, exactly what {doc}`solvers`
describes. If your field is crowded, this is what you are choosing between.

**The broadband anchors are a sanity check, not truth.** The LS/WISE diamonds are
measured through different apertures at different epochs; they should bracket the
SPHEREx spectrum, not match it point for point.

(fragmentation)=
## Caveat: catalog shredding — substructure fitted as extra sources

First, what is *not* the problem: blending. SPHEREx pixels are 6.15″, so real
sources routinely share a pixel, and dividing their blended light using the
catalog's positions and shapes is exactly what this pipeline is built to do.
That works — **when every catalog entry is a real source**.

The failure mode to watch for is a wrong source list. Surveys built at
sub-arcsecond resolution sometimes **shred** a large, well-resolved galaxy,
cataloguing its substructure — star-forming knots, the bulge, pieces of the
disc — as independent sources. Those entries are artifacts: their positions and
shapes describe fragments of one object, not real sources, so the fit gains
spurious model components stacked on top of the real galaxy.

Panels **(a)** and **(e)** show what that does. In (a)'s thumbnail the big cyan
ellipse is the galaxy's main catalog entry, and the smaller markers inside it
are its shredded substructure, each fitted as if it were a separate source:

- **(a)** — the main entry's spectrum scatters around zero, far below its own
  LS/WISE anchors (~2.2–2.8 mJy). The galaxy's light has been distributed over
  the artifact components, and because their positions and shapes are not
  descriptions of real sources, that division carries no physical meaning.
- **(e)** — this target is a *real* compact source near the galaxy's centre:
  the entry itself is legitimate and correctly placed. But it is fitted jointly
  with the surrounding artifact components, so its flux is not safe either — it
  can absorb host-galaxy light the fragments fail to model, or lose flux to
  them.

All three estimators behave identically here. The source list is wrong, and no
flux regularizer can turn a wrong model into a right one.

### How to protect yourself

1. **Find candidates.** Entries with another entry inside one SPHEREx pixel:

   ```python
   from astropy.coordinates import SkyCoord
   import astropy.units as u

   sc = SkyCoord(cat["ra"], cat["dec"], unit="deg")
   idx, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)   # nearest other entry
   candidates = sep < 6.15 * u.arcsec
   print(f"{candidates.sum()} of {len(cat)} entries share a pixel with another")
   ```

   A flagged group can be two real sources — that is ordinary blending, and it
   is fine — or a shredded galaxy, which is not. A glance at an optical
   thumbnail tells them apart: shredding looks like several entries sitting
   *inside* one extended galaxy.

2. **Clean the catalog before fitting.** For a shredded group, drop the
   substructure entries and keep a single entry carrying the galaxy's overall
   position and shape, then refit. This is the only fix that also protects a
   real source embedded in the group, like (e).

3. **Salvaging after the fact.** Summing the fitted fluxes over a shredded
   group approximately recovers the galaxy's total light — useful if the
   galaxy's spectrum is all you want. But the sum cannot separate out a real
   source like (e), and the individual fragment spectra remain meaningless.

4. **Look at the fit.** {func}`~tractorjax_spherex.diagnostics.plot_fit` — a
   shredded model often reproduces the *image* tolerably while dividing the
   flux arbitrarily; structured residuals centred on the galaxy mean even the
   image is not reproduced.

Large nearby galaxies are the common case; distant compact sources are rarely
shredded. See {doc}`catalogs` for catalog-side guidance.

## Reproducing this figure

The figure is generated from the private analysis repo (it needs the A2537
cutouts, the Legacy Survey thumbnails, and the channel definitions):
`proj-spherex-gpupipe/figures/docs_gallery_a2537.py`. The estimator products
themselves come from ordinary
{func}`~tractorjax_spherex.pipeline.run_photometry` runs with
`solver="linear"`, `"lasso"`, and `"eigfloor_prior"` on the full catalog.
