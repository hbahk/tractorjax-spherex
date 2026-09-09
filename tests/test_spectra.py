import numpy as np
from astropy.table import Table

from tractorjax_spherex.spectra import bin_spectrum, build_spectra, to_ab_mag


def _phot():
    return Table({
        "id": [1, 1, 1, 2, 2],
        "central_wavelength": [2.0, 1.0, 3.0, 2.0, np.nan],
        "flux": [5.0, 4.0, 6.0, 2.0, 2.0],
        "flux_err": [0.1, 0.1, 0.1, 0.5, 0.5],
        "detector": [1, 1, 1, 1, 1],
        "obs_id": ["a", "b", "c", "a", "d"],
        "cutout_index": [0, 1, 2, 0, 1],
    })


def test_build_spectra_sorted_by_wavelength():
    spectra = build_spectra(_phot())
    assert set(spectra) == {1, 2}
    wl = np.asarray(spectra[1]["central_wavelength"])
    assert list(wl) == sorted(wl)          # ascending
    # NaN-wavelength point in id=2 dropped
    assert len(spectra[2]) == 1


def test_build_spectra_min_snr():
    spectra = build_spectra(_phot(), min_snr=10.0)
    # id=2 flux/err = 4 < 10 -> dropped entirely
    assert 2 not in spectra
    assert len(spectra[1]) == 3


def test_bin_spectrum_ivar_weighted():
    spec = Table({"central_wavelength": [1.0, 1.02, 2.0],
                  "flux": [10.0, 20.0, 5.0],
                  "flux_err": [1.0, 1.0, 1.0]})
    b = bin_spectrum(spec, dlam=0.1)
    # first two combine (mean 15), last alone
    assert len(b) == 2
    assert b["flux"][0] == 15.0
    assert b["flux_err"][0] == 1.0 / np.sqrt(2)


def test_bin_spectrum_keeps_top_edge_point():
    # a point landing exactly on the top bin edge must not be silently dropped
    spec = Table({"central_wavelength": [1.5, 3.0],
                  "flux": [10.0, 20.0], "flux_err": [1.0, 1.0]})
    b = bin_spectrum(spec, edges=[1.0, 2.0, 3.0])
    assert int(b["n"].sum()) == 2               # both points retained
    assert 20.0 in list(b["flux"])              # the 3.0 point survived


def test_bin_spectrum_single_wavelength():
    spec = Table({"central_wavelength": [2.0, 2.0, 2.0],
                  "flux": [4.0, 6.0, 5.0], "flux_err": [1.0, 1.0, 1.0]})
    b = bin_spectrum(spec, dlam=0.05)
    assert len(b) == 1 and b["flux"][0] == 5.0


def test_to_ab_mag():
    # 3631 Jy = 3.631e6 mJy = 0 mag
    assert to_ab_mag(3.631e6) == 0.0
