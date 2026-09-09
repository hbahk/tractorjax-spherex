# Hardware & performance

Forced photometry over a SPHEREx field is a sequence of small, dense linear
solves — one batched vmap per cutout over its tiles. The build stage runs on the
CPU (catalog geometry, PSF stamps, background) and the solve stage runs on the
accelerator. This page covers how to size GPU memory, when precision matters, and
how to run with no GPU at all. Every knob below lives on
{class}`tractorjax_spherex.config.PhotometryConfig`; see {doc}`configuration` for
the full field table.

## GPU memory

### Preallocation — `gpu_preallocate` / `gpu_mem_fraction`

By default JAX grabs ~75% of the card the first time it touches the GPU, which
OOM-kills any co-running job. The pipeline disables this: with
`gpu_preallocate=False` (the default) it sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, so JAX grows its allocation on demand
instead. When you share a GPU, also cap the ceiling with `gpu_mem_fraction`,
which sets `XLA_PYTHON_CLIENT_MEM_FRACTION` — e.g. `0.45` to stay under half the
card:

```python
PhotometryConfig(device="gpu", gpu_preallocate=False, gpu_mem_fraction=0.45)
```

These are environment variables read by XLA at the first `import jax`, so they
must be applied *before* JAX is imported. `run_photometry` handles this by
calling `setup_device()` up front; a library caller doing its own imports should
call `tractorjax_spherex.device.setup_device(...)` before anything JAX-touching
(the CLI additionally peeks `--device`/mem flags out of `argv`). Setting them
after JAX is imported has no effect.

### `tile_chunk` — bounding peak memory

With `tile_chunk=0` (the default) the solver vmaps over *all* tiles of a cutout
in one dispatch, so peak GPU memory scales with the tile count. On a large,
crowded cutout that batch can be the memory high-water mark. Set `tile_chunk` to
a positive `k` to solve the tiles in fixed chunks of `k` in a Python loop, which
caps peak solve memory at roughly `k / n_tiles` of the monolithic batch. The last
chunk is padded up to `k` so a single compiled shape is reused across chunks; the
cost is more (smaller) dispatches rather than one. Reach for it only when a full
cutout does not fit under your `gpu_mem_fraction`.

### `pad_bucket` vs. fixed caps

Each tile has a different number of point sources and galaxies, so the flux
vector and design matrix have tile-dependent widths. Feeding raw widths to a
jitted solver forces a fresh XLA compile per distinct shape. Two policies tame
this, and they trade **compile count against matrix size**:

- **`pad_bucket`** (default `32`, the "F3" profile) rounds each tile's widths up
  to the nearest multiple. Matrices stay close to their real size (small, fast
  per-solve), at the cost of a handful of distinct bucketed shapes — a few
  compiles instead of one.
- **Fixed caps** (`max_ps_cap` / `max_gal_cap` / `max_mog_k_cap`, the older "F2"
  policy) pad *every* tile to the field-maximum width, so the solver compiles
  exactly once — but every tile then solves an oversized matrix, wasteful on
  tiles with few sources.

The two are mutually exclusive: setting a nonzero `pad_bucket` turns the auto caps
off. To get the single-compile cap policy instead, set `pad_bucket=0` while
fitting the full catalog (`fit_zmag_max=None`), which restores the built-in
full-depth caps. For most workloads the default `pad_bucket=32` is the right
balance.

## Precision — `fp32` vs. `fp64`

The default is `fp32`. Consumer and workstation GPUs run double precision far
slower than single — on an L40S FP64 throughput is about 1/64 of FP32 — so an
`fp64` run is dramatically slower for no accuracy gain in the common case. Keep
`fp32` unless you specifically need it: the `eigfloor` error calibration
(pull NMAD ≈ 0.95–0.99, see {doc}`solvers`) was established at `fp32` on real
SPHEREx data, and reported `flux`/`flux_err` are trustworthy there. Switch to
`precision="fp64"` only when you need exactness rather than accuracy — bit-level
agreement with the CPU/x64 reference path, or invariance to padding and batching
choices for faint near-degenerate fluxes; it enables `jax_enable_x64` and, like
the device settings, must be selected before the first JAX import.

## Prefetch — overlapping CPU build with GPU solve

The build (CPU) and solve (GPU) stages are pipelined. With `prefetch="thread"`
(the default) the JAX backend builds the *next* cutout on a worker thread while
the current one solves on the GPU, hiding the CPU build cost behind GPU work.
Set `prefetch="sync"` to disable the overlap (serial build-then-solve), useful
for debugging or profiling a single stage. Prefetch applies to the JAX backend
only.

## Running without a GPU

Two GPU-free paths, both documented in {doc}`cpu_backend`:

- **`cpu-tractor` backend** — `PhotometryConfig(backend="cpu-tractor")`. **The
  recommended GPU-free path.** Forced photometry on the classic upstream
  Tractor, tiled on the same 15 px core / 3 px halo grid the JAX backend uses
  (one `optimize_forced_photometry` per tile). Measured ~3.2× faster than the
  JAX engine on CPU at full catalog depth even before tiling, and 5–10× faster
  again with it. It supports `linear` only; `PhotometryConfig` raises a
  `ConfigError` for any other solver. See {doc}`cpu_backend` for the tiling
  rules, its rendering caveats, and `cpu_tile_background`.
- **JAX engine on CPU** — `PhotometryConfig(device="cpu")`. NOT a performance
  path (the engine is shaped for accelerators; XLA-on-CPU does not vectorize
  these kernels well). It forces the JAX CPU backend (`JAX_PLATFORMS=cpu`) and
  is the exact same validated engine, supporting *every* solver
  ({doc}`solvers`), with no dependency beyond the CPU `jax` that ships with
  `tractor-jax`. Use it for numerical cross-checks against the GPU path, and for
  the solvers `cpu-tractor` does not have — not for throughput.

On CPU the GPU-memory knobs above (`gpu_preallocate`, `gpu_mem_fraction`,
`tile_chunk`) have no effect, and `fp64` is comparatively cheap — CPU FP64 is not
penalized the way GPU FP64 is — so it is a reasonable choice for a CPU reference
run.
