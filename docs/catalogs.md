# Reference catalogs

`spherex-photometry` does **reference-catalog forced photometry**: source
positions and shapes come from an input catalog and are held **fixed** — only
per-source fluxes are solved. The catalog therefore defines *what* is fit and
*where*; the SPHEREx cutouts only ever contribute fluxes. Pick the estimator that
regularizes that solve in {doc}`solvers`.

You either bring your own catalog (any table with the columns below) or fetch one
from Legacy Survey DR10 with {func}`~spherex_photometry.fetch_ls_dr10`.

## The canonical schema

{func}`~spherex_photometry.io.catalogs.load_catalog` reads the input and
{func}`~spherex_photometry.io.catalogs.normalize_catalog` fills in the canonical
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
  {func}`~spherex_photometry.io.catalogs.normalize_catalog` raises `ValueError`.
- **`flux_z`** — only needed for the depth cut and for LASSO / prior *protection*
  (see below). If absent it is filled with `NaN`; those sources then have an
  undefined z-mag and are dropped by any active depth cut. Full-catalog blind
  runs (`fit_zmag_max=None`) do not require it.
- **`shape_r`** — missing ⇒ `0` ⇒ the source is rendered as a point source.
  A positive radius makes it a galaxy.
- **`sersic`** — missing ⇒ `1.0` (exponential); only meaningful for galaxies.
- **`shape_e1` / `shape_e2`** — missing ⇒ `0` (round). Normalization turns these
  into `shape_ab` / `shape_phi` via
  {func}`~spherex_photometry.models.ls_shapes_to_ab_phi`, using the Legacy Survey
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
from spherex_photometry import run_photometry, PhotometryConfig

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

## Watch out for fragmented galaxies

Because the catalog is a *prior* on position and shape, its resolution matters.
Optical catalogs such as Legacy Survey resolve a nearby galaxy at 0.26″ and often
split it into several entries — a bulge component, a disc component, a knot.
SPHEREx pixels are 6.15″, so those entries are indistinguishable to the fit: one
PSF's worth of light gets divided between near-identical models, almost
unconstrained by the data. The **sum** over the group is about right; the
individual fluxes are not, and one fragment can come out near zero while another
absorbs everything. See the worked case in {ref}`the gallery <fragmentation>`.

Before trusting a per-source spectrum, check whether the entry has a neighbour
inside one pixel:

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

sc = SkyCoord(cat["ra"], cat["dec"], unit="deg")
_, sep, _ = sc.match_to_catalog_sky(sc, nthneighbor=2)   # nearest OTHER entry
suspect = sep < 6.15 * u.arcsec
```

Then either merge each group into a single entry before fitting, or sum the
group's fitted fluxes afterwards. Distant compact sources are rarely affected;
large nearby galaxies almost always are.

## Fetching Legacy Survey DR10

{func}`~spherex_photometry.fetch_ls_dr10` queries `ls_dr10.tractor` around a sky
position and returns (and optionally writes) an Astropy Table already carrying
the required columns plus the SED bands. It is an **optional** feature:

```bash
pip install 'spherex-photometry[catalog]'      # adds the astro-datalab client
```

It needs **NOIRLab Data Lab credentials**, passed as `user` / `password`
arguments or via the `DATALAB_USER` / `DATALAB_PASSWORD` environment variables;
missing credentials raise `ValueError`.

```python
from spherex_photometry import fetch_ls_dr10

cat = fetch_ls_dr10(150.0, 2.0, out="catalog.parquet")   # creds from env
```

```bash
spherex-phot fetch-catalog --ra 150.0 --dec 2.0 --out catalog.parquet
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

Fitting every faint source in a deep catalog is the blind-production regime, but
you can prune the catalog by z-band depth with `fit_zmag_max` on
{class}`~spherex_photometry.config.PhotometryConfig`:

```python
PhotometryConfig(solver="eigfloor", fit_zmag_max=21.0)   # fit sources with z < 21
```

- `fit_zmag_max=None` (default) fits the **full catalog** — the production blind
  regime, and the only one that needs no `flux_z`.
- A finite value keeps only sources **brighter** than that AB z-mag
  (`apply_depth_cut`). Sources with `flux_z ≤ 0` (undefined z-mag) are dropped;
  the always-kept `target` survives the cut unconditionally.
- A too-aggressive cut removes real flux from blends and biases the fit — cut
  only sources genuinely too faint to matter at SPHEREx S/N. This is distinct
  from `protect_zmag_max`, which does not remove sources but marks bright ones as
  unpenalized reported targets for `lasso` / `eigfloor_prior` (see {doc}`solvers`
  and {doc}`configuration`).

Because positions and shapes are fixed, catalog quality is the dominant input to
the photometry: a missing or mislocated source is never recovered, and a wrong
shape biases the flux. Start from {doc}`quickstart` for the end-to-end flow.
