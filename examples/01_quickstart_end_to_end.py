"""End-to-end quickstart: retrieve -> catalog -> photometry -> spectra.

Run from anywhere; writes into ./quickstart_out/. Needs the tractor-jax engine
and spherex-retrieval installed (see the README). The catalog step needs the
[catalog] extra, or bring your own catalog with the columns documented in
spherex_photometry.io.catalogs.
"""

from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord

from spherex_photometry import (PhotometryConfig, build_spectra, fetch_ls_dr10,
                                retrieve, run_photometry)
from spherex_photometry.spectra import bin_spectrum, plot_spectrum

RA, DEC = 150.0, 2.0
out = Path("quickstart_out")
out.mkdir(exist_ok=True)

# 1. Download L2 cutouts around the target.
retrieve(SkyCoord(RA * u.deg, DEC * u.deg), 100,
         output_dir=str(out / "cutouts"),
         include_wavelength=True, include_sapm=True)

# 2. Fetch a Legacy Survey DR10 reference catalog (or supply your own parquet).
fetch_ls_dr10(RA, DEC, out=str(out / "catalog.parquet"))

# 3. Forced photometry with the blind-production default (eigfloor).
#    For a shared GPU, add gpu_preallocate=False, gpu_mem_fraction=0.45.
cfg = PhotometryConfig(solver="eigfloor")
phot = run_photometry(out / "cutouts", out / "catalog.parquet", cfg,
                      target=(RA, DEC), output=out / "phot.parquet")
print(f"Photometered {len(phot)} (source, visit) points")

# 4. Assemble and plot the brightest source's spectrum.
spectra = build_spectra(phot)
sid = max(spectra, key=lambda s: spectra[s]["flux"].mean())
spec = spectra[sid]
ax = plot_spectrum(spec, binned=bin_spectrum(spec, dlam=0.05), label=f"id={sid}")
ax.figure.savefig(out / "spectrum.png", dpi=150)
print(f"Wrote {out / 'spectrum.png'} for source {sid}")
