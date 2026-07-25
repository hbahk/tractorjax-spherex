# Choosing a solver

All four estimators solve the *same* linear forced-photometry system — fixed
source positions/shapes, fluxes free — but regularize it differently. At full
Legacy-Survey depth a SPHEREx cutout has ~22 catalog sources per PSF core, so the
raw system is ill-conditioned and the choice of regularizer *is* the science
decision. Set it with `PhotometryConfig(solver=...)` or `--solver`.

## The four solvers

### `eigfloor` — blind product (default)

A sign-free eigenvalue floor on the Jacobi-normalized normal-equations matrix
(`eig_floor`, default `1e-2`, in units of the largest eigenvalue). It keeps
negative fluxes, applies no selection, and produces **calibrated errors** —
`pull = (f - f_true)/sigma` has NMAD ≈ 0.95–0.99 down to low S/N. This is the
right choice for **blind photometry and photo-z**, where you must not bias the
faint end and you need trustworthy uncertainties.

```python
PhotometryConfig(solver="eigfloor", eig_floor=1e-2)
```

### `lasso` — targeted product

Non-negative LASSO with a per-tile `alpha = sqrt(2 ln p)` penalty
(`lasso_alpha="auto"`) and a sign-free debias step on *protected* bright targets
(`protect_zmag_max`, default 20). It gives the best scatter (σ_NMAD) at moderate
S/N (~3–30), so it is the choice when you care about **specific bright targets**.
Caveats: the non-negativity constraint zeroes the blind faint end (so it is *not*
a blind estimator), and its posteriors under-cover — do not use its errors for
population statistics.

```python
PhotometryConfig(solver="lasso", lasso_alpha="auto", protect_zmag_max=20.0)
```

### `eigfloor_prior` — regularized full-catalog product

`eigfloor` plus per-source Gaussian flux priors on the *penalized* (faint,
`z >= protect_zmag_max`) nuisance sources, ridged toward their Legacy-Survey
SED-predicted flux at each cutout's wavelength. Protected bright sources keep
`lambda = 0` (exactly `eigfloor`, unbiased). Use it when you fit the **full
catalog** and want the faint nuisances constrained by their multi-band SEDs
instead of floating freely. Requires SED band columns in the catalog
(`dered_flux_*` / `flux_*`); sources without a usable SED stay free.

```python
PhotometryConfig(solver="eigfloor_prior", prior_sigma_frac=0.5,
                 prior_sigma_min_ujy=5.0, protect_zmag_max=20.0)
```

The `prior_sigma_min_ujy` floor matters: without it, ultra-faint SED predictions
inflate the relative eigen-floor and crush *all* fluxes (protected included).

### `linear` — plain WLS

Weighted least squares with only a small Jacobi ridge (`rcond`). Correct and fast
for **sparse fields or shallow catalogs**, but at full LS depth the system is
degenerate and its null-space wanders with tile context — flux products shift
with tiling choices. Not a production estimator at depth.

```python
PhotometryConfig(solver="linear")
```

## Decision guide

- **Blind survey photometry / photo-z** → `eigfloor` (default).
- **A few bright targets you care about** → `lasso` (with `protect_zmag_max`).
- **Full-catalog fit with SED-informed faint nuisances** → `eigfloor_prior`.
- **Sparse field or a shallow (bright) catalog** → `linear`.

## Backend × solver support

| solver | JAX backend | `cpu-tractor` backend |
|---|---|---|
| `linear` | ✅ | ✅ |
| `eigfloor` | ✅ | ❌ |
| `eigfloor_prior` | ✅ | ❌ |
| `lasso` | ✅ | ❌ |

The upstream-Tractor CPU backend does a single weighted-least-squares solve and
has no eigenvalue-floor / LASSO / prior estimators, so it supports `linear` only;
`PhotometryConfig` raises a `ConfigError` for the others. For a GPU-free machine,
prefer the JAX backend with `device="cpu"` — it supports every solver. See
{doc}`cpu_backend`.

## Protection and depth knobs

- `fit_zmag_max` — fit only sources brighter than this z-band AB mag (`None` =
  full catalog, the production blind regime). The always-kept target survives the
  cut.
- `protect_zmag_max` — for `lasso` / `eigfloor_prior`, sources brighter than this
  are *protected* (unpenalized, unbiased reported targets); fainter ones are
  penalized nuisances.

## What no solver can fix

The estimator regularizes an under-determined *flux* solve; it cannot repair a
wrong *source list*. Deblending real sources that share a pixel is what the fit
is for — but if the catalog shreds a big galaxy's substructure into separate
entries, every solver dutifully includes those artifact components and divides
the galaxy's light among them in a physically meaningless way (and can bias real
sources fitted alongside them). See {ref}`the gallery <fragmentation>` for a
real example and {doc}`catalogs` for how to detect and clean shredded groups.

## A note on error calibration

The reported `flux_err` is the forward-model 1σ from the solver's Fisher
information. The `eigfloor` calibration (NMAD ≈ 0.95–0.99) was established at
`fp32` on real SPHEREx data; `lasso` posteriors under-cover by construction. If
you need population-level uncertainties, use `eigfloor` (or `eigfloor_prior` on
protected sources).
