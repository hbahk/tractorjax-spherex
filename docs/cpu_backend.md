# CPU backend

Two ways to run without a GPU:

1. **JAX engine on CPU** — `PhotometryConfig(device="cpu")`. NOT a performance path: measured ~3.2x SLOWER than the classic Tractor on one core at full catalog depth (the engine is shaped for accelerators; XLA-on-CPU does not vectorize these kernels well). Use it for numerical cross-checks against the GPU path, not for throughput. Same
   validated engine, every solver, accurate oversampled rendering. No extra
   dependencies beyond the CPU `jax` that ships with `tractor-jax`.
2. **`cpu-tractor` backend** — `PhotometryConfig(backend="cpu-tractor")`. Forced
   photometry on the classic [Tractor](https://github.com/dstndstn/tractor), for
   JAX-free environments or as an independent cross-check. `linear` solver only,
   and **tiled on the same grid as the JAX backend** (see below).

**If you lack a GPU, use option 2**, not option 1: measured at full catalog
depth, the classic Tractor is ~3.2× faster than the JAX engine on CPU, and the
tiled solve widens that further. Option 1 is for numerical cross-checks against
the GPU path and for the solvers option 2 does not have.

## Installing upstream Tractor

Not on PyPI; build from source (needs a C toolchain):

```bash
pip install git+https://github.com/dstndstn/tractor
```

`spherex-photometry` imports it lazily, so it is only required when you actually
select `backend="cpu-tractor"`.

## Accurate low-resolution rendering — and the PixelizedPSF flux bug

SPHEREx is undersampled (6.15″/pixel), so the PSF × source-shape convolution must
be evaluated on an **oversampled** grid and binned to native pixels; doing it at
native resolution biases fluxes. The L2 PSF cube is delivered oversampled (10×),
which we downsample to a 5× stamp and hand to the renderer.

Upstream Tractor *has* a mechanism for an oversampled PSF stamp
(`tractor.psf.PixelizedPSF(stamp, sampling=0.2)`), but it has a **flux
normalization bug**: it point-samples the oversampled PSF onto the native grid
and forgets the `1/sampling**2` pixel-area factor, so a unit-flux point-source
template sums to `sampling**2` (≈ 1/25 at 5×) and forced fluxes come out **~25×
too high**.

This package ships a corrected subclass,
{class}`spherex_photometry.backends.cpu_psf.OversampledPixelizedPSF`, which ports
the fix from the tractor-jax engine: it block-integrates the oversampled PSF to
native pixels (keeping the convolution at oversampled resolution) and applies the
correct flux scale, in both the point-source and Fourier (galaxy) paths. **Do not
pass `sampling < 1` to the stock `tractor.psf.PixelizedPSF`** for SPHEREx — use
`OversampledPixelizedPSF`, which the `cpu-tractor` backend does automatically.

## Agreement with the JAX backend

On isolated point sources the two independent rendering paths (the JAX
oversampled engine vs. upstream Tractor with `OversampledPixelizedPSF`) agree to
**well under 1%** in flux — the packaged tests assert ≤ 1% and observe ≈ 0.1–0.2%.
Run `backend="cpu-tractor", solver="linear"` against `backend="jax",
solver="linear"` on your own field to reproduce it.

## Tiled solve (`cpu_tiling`, on by default)

The backend splits each cutout on the **same tile grid the JAX backend uses** —
15 px cores with a 3 px halo, from the shared
{mod}`spherex_photometry.tiling` module — and runs one upstream
`optimize_forced_photometry` per tile. Each tile is a small, self-contained
`tractor.Tractor` over the tile's data/invvar slice; the upstream engine is not
modified in any way. Tiling here is **orchestration**, not a new estimator.

Two things follow from the geometry:

- **A source is modelled in every tile whose halo box it falls in, and reported
  from the one tile whose *core* box contains it.** Cores tile the cutout exactly
  (the last row/column is clipped at the edge), so every in-cutout source is
  claimed by exactly one tile — halo overlaps can neither double-count nor drop
  it. The halo copies exist so each tile's local deblend is right; their fitted
  values are discarded.
- **Each tile gets one constant PSF**, the zone kernels blended at that tile's
  core centre (or, with `psf_zone_interp=False`, the nearest zone to it) — the
  same piecewise-constant PSF field the JAX backend renders with.

Why it is the default:

- **Speed.** Tens of fluxes per tile instead of thousands per cutout, and the
  small systems converge fast. Measured on real SPHEREx cutouts (~100×100 px,
  49 tiles): **5–10× faster at full catalog depth** (29 s → 5.5 s per cutout,
  ~3800 reported sources) and ~1.9× at `fit_zmag_max=21`.
- **Conditioning.** The whole-cutout system at full LS depth is degenerate; each
  tile's is not. This is why the untiled path needs a depth cut and the tiled one
  does not.
- **Comparability.** With both engines on one geometry, a CPU-vs-GPU comparison
  measures the engines rather than the tiling.

It does not change the answer where the answer is well defined: on the same real
cutouts, restricted to sources the JAX backend detects at S/N > 5, tiled and
whole-cutout fluxes agree to a **median 0.01–0.09 %** (p90 0.15–0.2 % at
`fit_zmag_max=21`). The tails at full depth are large — that is the degeneracy
of the whole-cutout system, which is the thing tiling removes, not a
disagreement about a measurable quantity.

Set `cpu_tiling=False` for the original **whole-cutout** solve: one joint fit over
every source in the cutout. It is kept as the independent "global geometry"
cross-check. At full LS depth that system is degenerate, so keep `fit_zmag_max`
at a sensible depth (≈ 21) when you use it.

### Per-tile background (`cpu_tile_background`, on by default)

`prepare_pixels` subtracts a ZODI+model background fit once per cutout, on both
paths. The JAX backend's tiled solve additionally carries a **free constant per
tile**, solved jointly with the fluxes, which absorbs whatever DC the per-cutout
prefit left behind. The tiled CPU path fits one too (upstream Tractor's
`sky=True`), so the two backends run the *same solve* — same tiles, same
nuisance parameters — and not merely the same geometry.

Measured on real cutouts, it moves the CPU result towards the JAX one, cutting
the median S/N > 5 disagreement by 2–4× (e.g. 0.65 % → 0.19 % and 1.4 % → 0.32 %
on two a2537 cutouts at `fit_zmag_max=21`), for no measurable time cost.

Set `cpu_tile_background=False` to drop the column and rely on the per-cutout
prefit alone. The flag is inert with `cpu_tiling=False` — the whole-cutout solve
has no tiles, and it is deliberately kept as the reproduction of pre-tiling
products — and the backend logs that once per run rather than ignoring it
silently.

## Limitations of the `cpu-tractor` backend

- **`linear` only.** No eigenvalue-floor / LASSO / SED-prior estimators.
- **Resolved galaxies are approximate.** The point-source PSF is oversampled, but
  Tractor renders the galaxy profile itself at native resolution before the
  Fourier convolution, so a well-resolved Sérsic is less accurate than in the JAX
  backend (which renders the whole galaxy at 5×). Most SPHEREx sources are
  unresolved, so this rarely matters; use the JAX backend if it does.
- **Sources with no live pixels report flux 0 and infinite error.** Upstream's
  forced photometry *updates* the current parameters, so a source nothing
  constrains — its whole footprint masked, or its whole tile masked — is never
  stepped and would otherwise report the internal seed value as a measurement.
  Those rows are zeroed and given an infinite error, matching what the JAX
  backend returns; filter on `np.isfinite(flux_err)` if you want only measured
  points. The count is logged per cutout.
- **Error bars are a Fisher diagonal.** Upstream returns `Σ (t·σ⁻¹)²` per source,
  not the diagonal of the inverted normal matrix, so `flux_err` is *not*
  marginalized over co-fit neighbours (nor over the per-tile background when it
  is enabled). The JAX backend's variances are. Expect CPU errors to be
  systematically smaller in blended tiles — this is an estimator difference, not
  a tiling one.
