"""QR2 settings for the paper's main catalog-depth and prior comparisons.

This file specifies retrieval and photometry settings. The paper's field
footprints, exposure lists, and reference catalogs are separate inputs;
photometric-redshift calibration is described in its Appendix A.
Printing this module's configuration does not download data or initialize JAX.
"""

from __future__ import annotations

import json

SOFTWARE_VERSIONS = {
    "tractor-jax": "0.3.0",
    "tractorjax-spherex": "0.3.1",
    "spherex-retrieval": "0.3.1",
}

# Keep the QR2 optical PSF, wavelengths, solid angles, and gains.
RETRIEVAL_OPTIONS = {
    "collections": ("spherex_qr2", "spherex_qr2_deep"),
    "psf_source": "cal",
    "calibration_release": None,
    "gain_correction": False,
    "include_wavelength": True,
    "include_sapm": True,
    "subset_psf": True,
    "zone_margin": 1,
}

COMMON_PHOTOMETRY = {
    "backend": "jax",
    "precision": "fp32",
    "eig_floor": 1e-2,
    "tile_size": 15,
    "tile_halo": 3,
    "pad_bucket": 32,
    "bkg_model": "cwave+photutils",
    "bkg_box_size": 10,
    "bkg_filter_size": 3,
    "bkg_cwave_nbins": 48,
    "bkg_cwave_min_per_bin": 20,
    "psf_sampling": 0.2,
    "fixed_max_factor": 5.0,
    "psf_zone_interp": True,
    "psf_core_shift": True,
    "prefetch": "thread",
    "gpu_preallocate": False,
    "gpu_mem_fraction": 0.45,
    "strict": True,
}

CONFIGURATIONS = {
    "baseline": {"solver": "eigfloor", "fit_zmag_max": 21.0},
    "eigfloor_full": {"solver": "eigfloor", "fit_zmag_max": None},
    "eigfloor_prior": {
        "solver": "eigfloor_prior",
        "fit_zmag_max": None,
        "protect_zmag_max": 21.0,
        "prior_sigma_frac": 0.15,
        "prior_sigma_min_ujy": 5.0,
    },
}


def make_config(name: str, *, device: str = "auto"):
    """Construct one named photometry configuration without running a fit."""
    from tractorjax_spherex import PhotometryConfig

    return PhotometryConfig(
        **COMMON_PHOTOMETRY, **CONFIGURATIONS[name], device=device,
    )


if __name__ == "__main__":
    print(json.dumps({
        "software_versions": SOFTWARE_VERSIONS,
        "retrieval": RETRIEVAL_OPTIONS,
        "photometry": {
            name: {**COMMON_PHOTOMETRY, **options}
            for name, options in CONFIGURATIONS.items()
        },
    }, indent=2))
