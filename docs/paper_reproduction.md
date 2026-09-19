# QR2 settings for the paper

The catalog-depth paper uses the QR2 optical PSF, wavelength calibration,
solid-angle maps, and gains. Current retrieval defaults attach R7 calibration
products even to QR2 images. The paper configuration is recorded in
[`examples/paper_qr2_config.py`](https://github.com/hbahk/tractorjax-spherex/blob/main/examples/paper_qr2_config.py).

The recipe targets `tractor-jax` v0.3.0, `tractorjax-spherex` v0.3.1, and
`spherex-retrieval` v0.3.1. It supplies the three main comparison configurations:

| Name | Solver | Fitted catalog | External flux priors |
|---|---|---|---|
| `baseline` | `eigfloor` | z-band AB magnitude < 21 | None |
| `eigfloor_full` | `eigfloor` | Full catalog | None |
| `eigfloor_prior` | `eigfloor_prior` | Full catalog | Sources at z-band AB magnitude ≥ 21 |

The prior has fractional width 0.15 and minimum width 5 µJy. Protected
targets receive no direct external flux prior. The core shift is explicitly
enabled for the QR2 optical PSF, with 15-pixel tile cores and 3-pixel halos.
`strict=True` stops a run if any cutout fails, so a partial output is not
mistaken for a complete field.

From a checkout of this repository, print the settings without downloading
data or initializing JAX:

```bash
python examples/paper_qr2_config.py
```

Use the recipe with the coordinates, cutout size, and catalog for a paper
field:

```python
import runpy
from tractorjax_spherex import retrieve, run_photometry

paper = runpy.run_path("examples/paper_qr2_config.py")
bundles, summary = retrieve(
    coord, size, output_dir="cutouts_qr2_paper",
    **paper["RETRIEVAL_OPTIONS"],
)
cfg = paper["make_config"]("eigfloor_prior", device="gpu")
phot = run_photometry(
    "cutouts_qr2_paper", catalog_path, cfg, target=target,
)
```

Use a separate output directory for the paper's QR2 inputs rather than
reusing bundles with R7 gain corrections or effective PSFs. The source
catalog, exposure list, and field footprint must also match the paper;
retrieving additional visits changes the input data. This recipe specifies
photometry settings. The spectroscopic sample selection and photometric-
redshift calibration follow Sections 2--3 and Appendix A of the paper.

The unregularized full-depth comparison uses `solver="linear"` with
`precision="fp64"`. The sparse-selection configurations in Appendix C use
`solver="lasso"`, `protect_zmag_max=20.0`, and either fitted catalog depth;
their protection boundary differs from the prior configuration's boundary.
