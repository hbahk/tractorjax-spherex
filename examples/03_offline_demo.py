"""Fully self-contained demo: simulate -> photometer -> inspect the fit -> spectrum.

No network, no GPU, no real data needed — the field is simulated with the
package's toy simulator (`spherex_photometry.simulate`), so you can run this
immediately after installing to see the whole pipeline working and to learn the
API. Swap step 1 for `retrieve()` + a real catalog to do the same on real data.

Run:  python examples/03_offline_demo.py     (writes into ./offline_demo_out/)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spherex_photometry import PhotometryConfig, build_spectra, run_photometry
from spherex_photometry.diagnostics import plot_fit
from spherex_photometry.io.catalogs import load_catalog
from spherex_photometry.io.cutouts import read_cutout
from spherex_photometry.simulate import make_synth_catalog, make_synth_field
from spherex_photometry.spectra import bin_spectrum

out = Path("offline_demo_out")
out.mkdir(exist_ok=True)

# --- 1. Define sample sources and simulate a small field -------------------
# Three sources with known truth: two point sources and one small galaxy.
# (x, y) are pixel positions in the 40x40 toy cutouts; fluxes are mJy.
truth = [
    {"x": 12.0, "y": 14.0, "flux_mjy": 5.0},
    {"x": 27.0, "y": 24.0, "flux_mjy": 2.0},
    {"x": 18.0, "y": 30.0, "flux_mjy": 1.2, "shape_r": 1.5, "sersic": 1.0},
]
# 6 cutouts = 6 "visits"; each samples a different wavelength band, like the
# real SPHEREx survey (cwave_base steps per cutout inside make_synth_field).
# noise_mjy_sr=0.02 gives realistic-looking scatter and visible error bars.
make_synth_field(out / "cutouts", n_cutouts=6, sources=truth, seed=42,
                 noise_mjy_sr=0.02)

# The matching reference catalog (positions/shapes the fit will hold fixed).
cutout0 = read_cutout(sorted((out / "cutouts").glob("cutout_*.fits"))[0])
make_synth_catalog(out / "catalog.parquet", truth, cutout0.wcs)

# --- 2. Forced photometry ---------------------------------------------------
# device="cpu" so the demo runs anywhere; on a GPU box just drop that argument.
cfg = PhotometryConfig(solver="eigfloor", device="cpu", prefetch="sync")
phot = run_photometry(out / "cutouts", out / "catalog.parquet", cfg,
                      output=out / "phot.parquet")
print(f"\nPhotometered {len(phot)} (source, visit) measurements")

# --- 3. Inspect the fit: data / model / chi on one cutout -------------------
catalog = load_catalog(out / "catalog.parquet")
fig = plot_fit(cutout0, catalog, phot, cutout_index=0, config=cfg)
fig.savefig(out / "fit_comparison.png", dpi=150)
plt.close(fig)
print(f"Wrote {out / 'fit_comparison.png'}")

# --- 4. Spectra: measured points vs the injected truth ----------------------
spectra = build_spectra(phot)
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(truth)))
for (sid, spec), c, t in zip(sorted(spectra.items()), colors, truth):
    binned = bin_spectrum(spec, dlam=0.3)
    ax.errorbar(spec["central_wavelength"], spec["flux"],
                yerr=spec["flux_err"], fmt="o", ms=4, color=c, alpha=0.8,
                label=f"id={sid} measured")
    ax.axhline(t["flux_mjy"], color=c, ls="--", lw=1,
               label=f"id={sid} truth {t['flux_mjy']:.1f} mJy")
ax.set_xlabel("central wavelength [μm]")
ax.set_ylabel("flux [mJy]")
ax.legend(ncols=3, fontsize="small")
fig.savefig(out / "spectra_vs_truth.png", dpi=150)
plt.close(fig)
print(f"Wrote {out / 'spectra_vs_truth.png'}")

# --- 5. Quick numeric check --------------------------------------------------
print("\nrecovered vs truth (mean over visits):")
for i, t in enumerate(truth, start=1):
    m = phot["id"] == i
    print(f"  id={i}: truth={t['flux_mjy']:.2f}  "
          f"measured={np.mean(phot['flux'][m]):.3f} "
          f"+/- {np.mean(phot['flux_err'][m]):.3f} mJy")
