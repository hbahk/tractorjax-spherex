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
## Caveat: a fragmented galaxy gives unreliable per-source spectra

Compare panels **(a)** and **(e)**. They are the *same physical galaxy*: their
catalog positions differ by only 4.1″, well inside one 6.15″ SPHEREx pixel. The
Legacy Survey, working at 0.26″ resolution, resolved that galaxy into more than
one entry — a large elliptical component (the big cyan ellipse in (a)) plus a
compact component (the cyan `+` in (e)) — and the pipeline dutifully fits both.

SPHEREx cannot separate them. One PSF's worth of light has to be divided between
two models that are nearly identical from SPHEREx's point of view, so the split
is almost unconstrained by the data:

- **(a)** ends up with essentially nothing — the spectrum scatters around zero,
  far below the LS/WISE anchor of ~2.2–2.8 mJy that the same catalog entry
  predicts.
- **(e)** absorbs the light and produces a clean, well-behaved spectrum.

Neither number is the galaxy's flux. Their **sum** is roughly right; the
individual entries are not. Note that every estimator does this — it is not a
solver artefact but a consequence of the input catalog describing the sky at a
resolution SPHEREx does not have.

### How to protect yourself

1. **Check your target's neighbourhood before trusting its spectrum.** Any
   catalog entry with another entry within ~6″ is suspect. A quick check on your
   own catalog:

   ```python
   from astropy.coordinates import SkyCoord
   import astropy.units as u

   sc = SkyCoord(cat["ra"], cat["dec"], unit="deg")
   idx, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)   # nearest other entry
   fragmented = sep < 6.15 * u.arcsec
   print(f"{fragmented.sum()} of {len(cat)} entries have a neighbour inside one pixel")
   ```

2. **Merge fragments before fitting, or sum after.** Either replace the group
   with a single entry (one position, one shape) so the fit solves for one flux,
   or add the fitted fluxes of the group afterwards and treat the sum as the
   galaxy's spectrum. Summing is easier and keeps the errors meaningful if you
   propagate them together; merging gives a cleaner model but needs you to choose
   a representative shape.

3. **Look at the fit.** {func}`~spherex_photometry.diagnostics.plot_fit` shows
   whether the model reproduces the galaxy's light even when the individual
   fluxes are split oddly — a fragmented galaxy usually fits the *image* fine
   while distributing the flux badly between entries.

4. **Prefer a catalog matched to the resolution of your science.** Nearby, well-
   resolved galaxies are the common failure case; distant compact sources are
   rarely affected. See {doc}`catalogs`.

## Reproducing this figure

The figure is generated from the private analysis repo (it needs the A2537
cutouts, the Legacy Survey thumbnails, and the channel definitions):
`proj-spherex-gpupipe/figures/docs_gallery_a2537.py`. The estimator products
themselves come from ordinary
{func}`~spherex_photometry.pipeline.run_photometry` runs with
`solver="linear"`, `"lasso"`, and `"eigfloor_prior"` on the full catalog.
