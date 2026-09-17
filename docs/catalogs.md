# Reference catalogs

`tractorjax-spherex` does **reference-catalog forced photometry**: source
positions and shapes come from an input catalog and are held **fixed** — only
per-source fluxes are solved. The catalog therefore defines *what* is fit and
*where*; the SPHEREx cutouts only ever contribute fluxes. Pick the estimator that
regularizes that solve in {doc}`solvers`.

You either bring your own catalog (any table with the columns below) or fetch one
from Legacy Survey DR10 with {func}`~tractorjax_spherex.fetch_ls_dr10`.

## The canonical schema

{func}`~tractorjax_spherex.io.catalogs.load_catalog` reads the input and
{func}`~tractorjax_spherex.io.catalogs.normalize_catalog` fills in the canonical
columns — adds `id`, defaults missing shape columns, and derives the engine
shape columns `shape_ab` / `shape_phi` from the ellipticity. Original columns
(including SED bands) are preserved.

| column | required | units | meaning |
|---|---|---|---|
| `id` (int64) | **yes** | — | unique source id; `ls_id` accepted as an alias |
| `ra`, `dec` | **yes** | deg (ICRS) | source position |
| `flux_z` | depth / protect | nanomaggies | z-band flux → AB `zmag = 22.5 − 2.5·log₁₀(flux_z)` |
| `shape_r` | no | arcsec | effective radius; `0` or missing ⇒ **point source** |
| `sersic` | no | — | Sérsic index (galaxies only) |
| `shape_e1`, `shape_e2` | no | — | ellipticity components; `0` ⇒ round |
| `dered_flux_*` / `flux_*` | prior only | nanomaggies | SED bands g/r/i/z/w1/w2, used **only** by `eigfloor_prior` |

Details on the optional columns:

- **`id`** — required and must be unique. If absent but `ls_id` is present, `id`
  is derived from it (and vice versa). With neither,
  {func}`~tractorjax_spherex.io.catalogs.normalize_catalog` raises `ValueError`.
- **`flux_z`** — drives the depth cut (on by default, `fit_zmag_max=21`) and
  LASSO / prior *protection* (see below). If the column is absent, or has no
  positive value anywhere, the pipeline logs a warning and fits the catalog at
  its own depth rather than reducing it to the kept target. If it is present,
  sources whose own `flux_z` is `NaN` or `<= 0` have an undefined z-mag and
  are dropped by the cut. Full-catalog runs (`fit_zmag_max=None`) never read
  it.
- **`shape_r`** — missing ⇒ `0` ⇒ the source is rendered as a point source.
  A positive radius makes it a galaxy.
- **`sersic`** — missing ⇒ `1.0` (exponential); only meaningful for galaxies.
- **`shape_e1` / `shape_e2`** — missing ⇒ `0` (round). Normalization turns these
  into `shape_ab` / `shape_phi` via
  {func}`~tractorjax_spherex.models.ls_shapes_to_ab_phi`, using the Legacy Survey
  sky-frame position-angle convention.
- **SED bands** — `dered_flux_{g,r,i,z,w1,w2}` (preferred) or `flux_{...}`
  (fallback), in nanomaggies. These feed the per-source SED flux predictor for
  the `eigfloor_prior` solver only; sources without at least two positive bands
  stay free (no prior). No other solver reads them. See {doc}`solvers`.

## Bring your own catalog

Any table Astropy can read works — pass a **path** (`.parquet`, `.fits`,
`.ecsv`, anything `astropy.table.Table.read` understands) or an in-memory
`Table`. Just provide the required columns; the optional ones are filled with the
defaults above when missing.

```python
from astropy.table import Table
from tractorjax_spherex import run_photometry, PhotometryConfig

cat = Table()
cat["id"]  = [1, 2, 3]
cat["ra"]  = [150.001, 150.004, 149.998]   # deg, ICRS
cat["dec"] = [2.000, 2.002, 1.997]
cat["shape_r"] = [0.0, 0.7, 0.0]           # 0 ⇒ point source, else galaxy
cat.write("catalog.parquet", overwrite=True)

phot = run_photometry("cutouts", "catalog.parquet",
                      PhotometryConfig(solver="eigfloor"),
                      target=(150.0, 2.0), output="phot.parquet")
```

`run_photometry` accepts the `Table` directly too — `run_photometry("cutouts",
cat, cfg)`. To make a bright source a **protected** reported target for the
`lasso` / `eigfloor_prior` solvers, or to survive a depth cut, give it a
`flux_z`; for a galaxy add `sersic`, `shape_e1`, `shape_e2`.

## Watch out for catalog shredding

The catalog is a *prior* on which sources exist and what they look like — so a
wrong source list produces a wrong fit, no matter the estimator. Blended real
sources are not the concern: deblending sources that share a pixel is exactly
what the joint fit is for. The concern is **shredding** — optical catalogs built
at sub-arcsecond resolution sometimes list a large galaxy's substructure (knots,
the bulge, pieces of the disc) as independent sources. Those entries are
artifacts with non-physical positions and shapes; the fit dutifully includes
them, the galaxy's light gets divided among spurious components, and even a
*real* source sitting among them can have its flux biased. See the worked case
in {ref}`the gallery <fragmentation>`.

To find candidates, look for entries with another entry inside one SPHEREx
pixel:

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

sc = SkyCoord(cat["ra"], cat["dec"], unit="deg")
_, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)   # nearest OTHER entry
candidates = sep < 6.15 * u.arcsec
```

A flagged group can be two real sources (fine) or a shredded galaxy (not fine);
an optical thumbnail tells them apart — shredding looks like several entries
sitting *inside* one extended galaxy. Clean such groups before fitting: drop the
substructure entries and keep a single entry carrying the galaxy's overall
position and shape. Summing the group's fitted fluxes afterwards approximately
recovers the galaxy's total light, but it cannot rescue a real source embedded
in the group. Large nearby galaxies are the common case; distant compact
sources are rarely shredded.

## Fetching Legacy Survey DR10

{func}`~tractorjax_spherex.fetch_ls_dr10` queries `ls_dr10.tractor` around a sky
position and returns (and optionally writes) an Astropy Table already carrying
the required columns plus the SED bands. It is an **optional** feature:

```bash
pip install 'tractorjax-spherex[catalog]'      # adds the astro-datalab client
```

It needs **NOIRLab Data Lab credentials**, passed as `user` / `password`
arguments or via the `DATALAB_USER` / `DATALAB_PASSWORD` environment variables;
missing credentials raise `ValueError`.

```python
from tractorjax_spherex import fetch_ls_dr10

cat = fetch_ls_dr10(150.0, 2.0, out="catalog.parquet")   # creds from env
```

```bash
tractorjax-spherex fetch-catalog --ra 150.0 --dec 2.0 --out catalog.parquet
```

The cone `radius_deg` defaults to the half-diagonal of a `cutout_pixels`
(default 100) SPHEREx cutout, so every source that could fall in any cutout of
the field is captured; pass `radius_deg` to override. `type == "DUP"` rows
(Legacy Survey duplicates) are dropped by default (`drop_dup`), float64 columns
are downcast to float32, and `out` writes a parquet. The default column set
(`DEFAULT_COLUMNS`) includes `ls_id`, `ra`/`dec`, `flux_z`, `sersic`,
`shape_r`/`shape_e1`/`shape_e2`, and the g/r/i/z/w1/w2 `dered_flux_*` / `flux_*`
bands — everything the solvers can use.

## Depth cuts

The catalog is truncated at z-band AB 21 **by default** (`fit_zmag_max=21` on
{class}`~tractorjax_spherex.config.PhotometryConfig`). This is the configuration
of record of the SPHEREx deblending campaign, and the single setting a first
run most often gets wrong by turning it off:

- At full Legacy Survey DR10 depth a SPHEREx cutout carries ~22 catalog sources
  per PSF core. The flux solve is then under-determined, and even the
  regularized `eigfloor` estimator pays for it: on the same sources, the paired
  scatter of the full-catalog fit is ~1.3× that of the m_z<21 fit (~1.9× for
  `linear`), and a much larger share of the reported fluxes has S/N below 5.
- Sources fainter than z ≈ 21 are below the single-visit SPHEREx noise, so
  dropping them loses almost no measurable flux, while keeping them adds a
  degenerate nuisance per source. The cut removes about 85 % of a DR10 catalog
  and the photometry of what remains gets *better*.

```python
PhotometryConfig()                        # eigfloor on the m_z < 21 catalog
PhotometryConfig(fit_zmag_max=None)       # the full catalog (eigfloor_prior arm)
PhotometryConfig(fit_zmag_max=20.0)       # a shallower cut
```

- A finite value keeps only sources **brighter** than that AB z-mag
  (`apply_depth_cut`). Sources with `flux_z ≤ 0` (undefined z-mag) are dropped;
  the always-kept `target` survives the cut unconditionally. A catalog with no
  usable `flux_z` at all is fitted unchanged, with a warning.
- `fit_zmag_max=None` fits the **full catalog**. Use it with `eigfloor_prior`
  (which ridges the faint nuisances toward their SED-predicted fluxes,
  {doc}`solvers`), not with a blind estimator.
- A *much* more aggressive cut removes real flux from blends and biases the fit:
  cut only sources genuinely too faint to matter at SPHEREx S/N. The depth cut
  is distinct from `protect_zmag_max`, which does not remove sources but marks
  bright ones as unpenalized reported targets for `lasso` / `eigfloor_prior`
  (see {doc}`solvers` and {doc}`configuration`).

Because positions and shapes are fixed, catalog quality is the dominant input to
the photometry: a missing or mislocated source is never recovered, and a wrong
shape biases the flux. Start from {doc}`quickstart` for the end-to-end flow.
