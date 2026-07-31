# CPU backend

Two ways to run without a GPU:

1. **JAX engine on CPU** — `PhotometryConfig(device="cpu")`. NOT a performance path: measured ~3.2x SLOWER than the classic Tractor on one core at full catalog depth (the engine is shaped for accelerators; XLA-on-CPU does not vectorize these kernels well). Use it for numerical cross-checks against the GPU path, not for throughput. Same
   validated engine, every solver, accurate oversampled rendering. No extra
   dependencies beyond the CPU `jax` that ships with `tractor-jax`.
2. **`cpu-tractor` backend** — `PhotometryConfig(backend="cpu-tractor")`. Forced
   photometry on the classic [Tractor](https://github.com/dstndstn/tractor), for
   JAX-free environments or as an independent cross-check. `linear` solver only.

If you just lack a GPU, use option 1. Option 2 exists for users already in the
Tractor ecosystem and as a second, independent code path.

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

## Limitations of the `cpu-tractor` backend

- **`linear` only.** No eigenvalue-floor / LASSO / SED-prior estimators.
- **Resolved galaxies are approximate.** The point-source PSF is oversampled, but
  Tractor renders the galaxy profile itself at native resolution before the
  Fourier convolution, so a well-resolved Sérsic is less accurate than in the JAX
  backend (which renders the whole galaxy at 5×). Most SPHEREx sources are
  unresolved, so this rarely matters; use the JAX backend if it does.
- **Whole-cutout solve.** No tiling; at full LS depth the single WLS is
  degenerate — keep `fit_zmag_max` at a sensible depth (≈ 21) for this backend.
- **Background** is fit once per cutout (the JAX backend additionally fits a small
  per-tile residual background column).
