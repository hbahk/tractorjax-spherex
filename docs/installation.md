# Installation

`tractorjax-spherex` depends on two packages that are not yet on PyPI — the
`tractor-jax` engine and the `spherex-retrieval` cutout downloader — so install
them from source first, then this package.

```bash
pip install git+https://github.com/hbahk/tractor-jax@v0.3.1         # engine (CPU jax)
pip install git+https://github.com/hbahk/spherex-retrieval@v0.3.2   # L2 cutout downloader
pip install git+https://github.com/hbahk/tractorjax-spherex@v0.3.2  # this package
```

These are the releases this version is developed and tested against
(`tractorjax-spherex` 0.3.2 needs `tractor-jax` >= 0.3.1, whose solvers return
the per-visit fit diagnostics, and refuses to import an older engine). Drop the
`@tag` to follow each repository's `main`.

For local development use an editable install without re-resolving the siblings:

```bash
pip install -e . --no-deps      # from a checkout of tractorjax-spherex
```

Python ≥ 3.11 is required.

## GPU

`tractor-jax` pulls the plain (CPU) `jax`. To run on a GPU, install the CUDA
build:

```bash
pip install "tractorjax-spherex[gpu]"    # jax[cuda12]
```

Everything works on CPU without this — GPU is a performance option, not a
requirement (see {doc}`hardware`).

## Optional extras

```bash
pip install "tractorjax-spherex[catalog]"  # NOIRLab Data Lab (Legacy Survey fetcher)
pip install "tractorjax-spherex[plot]"     # matplotlib for spectrum plots
pip install "tractorjax-spherex[dev]"      # pytest, ruff
pip install "tractorjax-spherex[docs]"     # sphinx toolchain
```

## CPU-only Tractor backend (optional)

The `cpu-tractor` backend uses the upstream Tractor, which is not on PyPI and
needs a C toolchain to build:

```bash
pip install git+https://github.com/dstndstn/tractor
```

It is imported lazily and only needed when you select `backend="cpu-tractor"`.
Most CPU-only users should instead run the JAX engine with `device="cpu"`. See
{doc}`cpu_backend`.

## License note

The package is GPL-3.0-or-later because it links the GPL `tractor-jax` engine.
The optional upstream `tractor` is GPLv2 and is used only as an optional,
user-installed runtime dependency (never vendored).
