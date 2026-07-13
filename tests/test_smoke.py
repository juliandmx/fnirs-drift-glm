"""Smoke-Tests fuer die Analyse-Pipeline.

Schnelle Invarianten-Checks (kein voller GLM-Fit). Der Integrationstest laedt einmal die
Ruhedaten und prueft die zentralen Eigenschaften der augmentierten Pipeline.

Aufruf:  conda run -n cedalion python -m pytest tests -q
"""
import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------- schnelle Tests

def _fake_conc(nt=240, nch=3):
    """Minimale NDTimeSeries-artige DataArray (time, channel, chromo) fuer die
    Drift-Bausteine (Cedalion braucht eine raeumliche Dimension)."""
    t = np.linspace(0.0, 24.0, nt)
    return xr.DataArray(
        np.zeros((nt, nch, 2)), dims=("time", "channel", "chromo"),
        coords={"time": t, "channel": [f"S1D{i}" for i in range(nch)],
                "chromo": ["HbO", "HbR"]})


def test_bspline_partition_of_unity():
    """B-Spline-Drift: 6 Regressoren, geklemmte Basis bildet Zerlegung der Eins."""
    import sweep
    dm = sweep._bspline_dm(_fake_conc(), 6)
    common = dm.common
    assert common.sizes["regressor"] == 6
    assert list(common.dims) == ["time", "regressor", "chromo"]
    total = common.sel(chromo="HbO").sum("regressor").values
    assert np.allclose(total, 1.0, atol=1e-6)
    assert all(str(r).startswith("Drift BS ") for r in common.regressor.values)


def test_drift_family_dispatch():
    """drift_dm liefert fuer jede Familie eine DesignMatrix mit erwartetem Praefix."""
    import sweep
    conc = _fake_conc()
    checks = {
        "poly:2": "Drift ", "legendre:3": "Drift LP ", "bspline:5": "Drift BS ",
        "none": "Drift ",
    }
    for fam, prefix in checks.items():
        dm, butter = sweep.drift_dm(fam, conc)
        assert butter is None
        assert any(str(r).startswith(prefix) for r in dm.common.regressor.values)
    # Butterworth: kein Drift-Regressor, aber Cutoff zurueckgegeben
    dm, butter = sweep.drift_dm("butter:0.01", conc)
    assert butter == pytest.approx(0.01)


# --------------------------------------------------- Integration (laedt Daten)

@pytest.fixture(scope="module")
def P():
    import pipeline as pl
    return pl.build(window_s=90.0, seed=0)


def test_beta_true_map_is_spatial_blob(P):
    """Ground-Truth ist raeumlich variabel (Blob), Peak ~ beta_true, HbR invers."""
    btm = P.beta_true_map
    assert set(btm.dims) == {"channel", "chromo"}
    hbo = btm.sel(chromo="HbO").values
    assert np.isclose(np.nanmax(hbo), P.beta_true["HbO"], rtol=0.05)
    assert np.nanstd(hbo) > 0                      # nicht flach -> echter Blob
    # inverse HbO/HbR-Beziehung (gleiches raeumliches Muster, umgekehrtes Vorzeichen)
    hbr = btm.sel(chromo="HbR").values
    assert np.nanmin(hbr) < 0 < np.nanmax(hbo)


def test_hrf_regressor_peak_normalized(P):
    """Der HRF-Regressor ist auf Peak 1 normiert -> beta == injizierte Peak-Amplitude."""
    hrf = P.dm_hrf.common.sel(regressor=P.hrf_names[0])
    assert np.isclose(float(np.abs(hrf).max()), 1.0, atol=1e-6)


def test_injection_is_additive(P):
    """conc_syn = conc + eingespeiste Aktivierung."""
    diff = (P.conc_syn - (P.conc + P.activation))
    assert float(np.abs(diff).max()) < 1e-9
