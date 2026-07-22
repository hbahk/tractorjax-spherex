"""Compare the four solvers on one field (same cutouts, same catalog).

Runs linear / eigfloor / eigfloor_prior / lasso over a prepared field and prints
the main source's flux from each, so you can see how the estimators differ on the
same photons. Point ``CUTOUTS`` and ``CATALOG`` at your own prepared field.
"""

import numpy as np

from spherex_photometry import PhotometryConfig, run_photometry

CUTOUTS = "cutouts"
CATALOG = "catalog.parquet"
TARGET = (150.0, 2.0)

for solver in ("linear", "eigfloor", "eigfloor_prior", "lasso"):
    cfg = PhotometryConfig(solver=solver, device="cpu")   # device="gpu" if available
    phot = run_photometry(CUTOUTS, CATALOG, cfg, target=TARGET, progress=False)
    # summarize the target (nearest source to TARGET is the main source)
    ids, first = np.unique(phot["id"], return_index=True)
    main_id = phot["id"][0]
    m = phot["id"] == main_id
    print(f"{solver:>15}: main-source mean flux = "
          f"{np.nanmean(phot['flux'][m]):.4f} mJy over {m.sum()} visits")
