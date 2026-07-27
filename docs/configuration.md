# Configuration

Every option lives on {class}`spherex_photometry.config.PhotometryConfig`. The
defaults reproduce the **blind-production ("F3") profile**: solver `eigfloor`,
tile 15 / halo 3 / pad-bucket 32, prefetch thread, `fp32`. Change `solver` (and
read {doc}`solvers`) to select a different estimator.

```python
from spherex_photometry import PhotometryConfig
cfg = PhotometryConfig(solver="eigfloor", device="cpu")
```

A config round-trips to YAML/TOML so a run is reproducible from one file:

```python
cfg.to_yaml("run.yaml")
cfg = PhotometryConfig.from_file("run.yaml")
```

On the CLI, `--config run.yaml` loads a file and any explicit flags override it.

## Fields

### Solver

| field | default | meaning |
|---|---|---|
| `solver` | `"eigfloor"` | `linear` / `eigfloor` / `eigfloor_prior` / `lasso` ({doc}`solvers`) |
| `eig_floor` | `1e-2` | relative eigenvalue floor (`eigfloor`, `eigfloor_prior`) |
| `lasso_alpha` | `"auto"` | LASSO penalty; `"auto"` = per-tile `sqrt(2 ln p)` |
| `lasso_n_iter` | `1000` | FISTA iterations (`lasso`) |
| `protect_zmag_max` | `20.0` | bright-source protection tier (`lasso`, `eigfloor_prior`) |
| `prior_sigma_frac` | `0.5` | `sigma_prior = frac * f_SED` (`eigfloor_prior`) |
| `prior_sigma_min_ujy` | `5.0` | floor on `sigma_prior` [µJy] (`eigfloor_prior`) |

### Catalog depth

| field | default | meaning |
|---|---|---|
| `fit_zmag_max` | `None` | fit only sources brighter than this z-mag; `None` = full catalog |

### Tiling / batching

| field | default | meaning |
|---|---|---|
| `tile_size` | `15` | core tile size (native px) |
| `tile_halo` | `3` | halo width for PSF wings (native px) |
| `pad_bucket` | `32` | round batch widths up to this multiple (throughput; replaces caps) |
| `tile_chunk` | `0` | solve tiles in fixed vmap chunks to bound GPU memory (`0` = one batch) |
| `max_ps_cap` / `max_gal_cap` / `max_mog_k_cap` | `None` | fixed batch widths; `None` = auto policy |

:::{warning}
**A tile that overflows a fixed cap loses its whole cutout.** When the caps are
active — full-depth fits (`fit_zmag_max` unset) with `pad_bucket` turned off —
a cutout whose densest tile needs more slots than the cap raises, and the
pipeline logs the traceback and **skips that cutout**, then reports the total in
one `N cutouts failed and were skipped` warning at the end. The run still exits
0 and still writes a parquet, so an incomplete product looks like a successful
one unless you read the log.

The defaults (`MAX_PS_CAP = 112`, `MAX_GAL_CAP = 352`) were sized on one
sparse field and do not generalise. Measured densest-tile occupancies across
eight real SPHEREx fields at `tile_size=15`, `tile_halo=3`:

| field | max point sources | max galaxies |
|---|---|---|
| A1361 | 51 | 110 |
| A2187 | 60 | 102 |
| SpARCS J1613+5649 | 62 | 101 |
| A2537 | 79 | 266 |
| A2055 | 80 | 139 |
| SPT-CL J0546-5345 | **128** | 242 |
| SPT-CL J2145-5644 | **192** | **373** |
| COSMOS | **268** | **450** |

Half the fields exceed a default. Keep the `pad_bucket=32` default (it turns
the caps off and sizes each batch near its natural width), or set the caps
explicitly from your own field's occupancy. **Always check the run log for the
skip warning before using a product.**
:::

### Background

| field | default | meaning |
|---|---|---|
| `bkg_model` | `"photutils"` | `photutils` / `cwave+photutils` / `plane` / `none` ({doc}`backgrounds_systematics`) |
| `bkg_box_size` | `10` | `Background2D` box size |
| `bkg_filter_size` | `3` | `Background2D` filter size |
| `bkg_cwave_nbins` | `48` | wavelength bins for the airglow profile (`cwave+photutils`) |
| `bkg_cwave_min_per_bin` | `20` | minimum pixels per wavelength bin |

### Rendering

| field | default | meaning |
|---|---|---|
| `psf_sampling` | `0.2` | native-pixel size per PSF-stamp pixel (0.2 = 5× oversampled) |
| `fixed_max_factor` | `5.0` | oversampled rendering factor |

### Execution

| field | default | meaning |
|---|---|---|
| `backend` | `"jax"` | `jax` or `cpu-tractor` ({doc}`cpu_backend`) |
| `device` | `"auto"` | `auto` / `gpu` / `cpu` (JAX backend) |
| `precision` | `"fp32"` | `fp32` or `fp64` |
| `prefetch` | `"thread"` | overlap CPU build with GPU solve (`thread`) or not (`sync`) |
| `gpu_mem_fraction` | `None` | `XLA_PYTHON_CLIENT_MEM_FRACTION` (e.g. `0.45` when sharing) |
| `gpu_preallocate` | `False` | if `False`, JAX grows memory on demand instead of grabbing ~75% |

## Validation

`PhotometryConfig` validates on construction and raises `ConfigError` for unknown
enums or the unsupported `cpu-tractor` + non-`linear` combination. See
{doc}`solvers` for the backend × solver matrix.
