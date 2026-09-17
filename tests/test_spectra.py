import numpy as np
import pytest
from astropy.table import Table

from tractorjax_spherex.spectra import (
    N_CHANNELS,
    bin_spectrum,
    bin_to_channels,
    build_spectra,
    channel_index,
    spherex_channels,
    to_ab_mag,
)


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


def test_spherex_channels_match_the_fiducial_table():
    """Six bands x 17 constant-R channels between the public band edges.
    Reference values are the campaign's 102-channel table (dumped from the
    instrument's channel definition), which this reconstruction reproduces
    to double precision."""
    ch = spherex_channels()
    assert len(ch) == N_CHANNELS == 102
    assert list(ch["channel"]) == list(range(1, 103))
    assert np.all(np.diff(ch["central_wavelength"]) > 0)
    assert ch["lambda_min"][0] == 0.75 and ch["lambda_max"][101] == 5.01
    assert ch["lambda_max"][0] == pytest.approx(0.767901964519022, rel=1e-12)
    assert ch["lambda_min"][17] == 1.10 and ch["detector"][17] == 2
    assert ch["central_wavelength"][99] == pytest.approx(4.916928019996501, rel=1e-12)
    # constant resolving power within a band
    R = ch["central_wavelength"] / (ch["lambda_max"] - ch["lambda_min"])
    assert np.ptp(R[:17]) < 1e-9 and np.ptp(R[85:]) < 1e-9


def test_channel_index_is_detector_aware():
    # 1.11 um lies in band 1's last channel AND band 2's first: the detector decides.
    idx = channel_index([1.11, 1.11, 0.70, np.nan, 5.0], [1, 2, 1, 3, 6])
    assert list(idx) == [16, 17, -1, -1, 101]


def test_bin_to_channels_ivar_weighted_and_sorted():
    spec = Table({"central_wavelength": [0.755, 0.760, 4.99, 1.11, 0.70],
                  "flux": [10.0, 20.0, 5.0, 7.0, 99.0],
                  "flux_err": [1.0, 1.0, 1.0, 1.0, 1.0],
                  "detector": [1, 1, 6, 2, 1]})
    b = bin_to_channels(spec)
    # channel 1 (two visits), channel 18 (one), channel 102 (one); 0.70 um dropped
    assert list(b["channel"]) == [1, 18, 102]
    assert b["flux"][0] == 15.0 and b["flux_err"][0] == pytest.approx(1 / np.sqrt(2))
    assert b["n"][0] == 2 and b["lambda_mean"][0] == pytest.approx(0.7575)
    assert b["central_wavelength"][1] == pytest.approx(spherex_channels()["central_wavelength"][17])
    assert int(b["n"].sum()) == 4
    # empty input -> empty table with the schema intact
    empty = bin_to_channels(spec[:0])
    assert len(empty) == 0 and "channel" in empty.colnames


def test_to_ab_mag():
    # 3631 Jy = 3.631e6 mJy = 0 mag
    assert to_ab_mag(3.631e6) == 0.0
