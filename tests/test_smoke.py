"""Smoke-Tests fuer die Analyse-Pipeline: schnelle Invarianten ohne Daten und ein
Integrationsteil auf den nn22-Ruhedaten (kein voller GLM-Fit).

Aufruf:  conda run -n cedalion python -m pytest tests -q
"""
import numpy as np
import pytest
import xarray as xr


# ---------------------------------------------------------------- schnelle Tests

def _fake_conc(nt=240, nch=3):
    """Leere (time, channel, chromo)-DataArray fuer die Drift-Bausteine."""
    t = np.linspace(0.0, 24.0, nt)
    return xr.DataArray(
        np.zeros((nt, nch, 2)), dims=("time", "channel", "chromo"),
        coords={"time": t, "channel": [f"S1D{i}" for i in range(nch)],
                "chromo": ["HbO", "HbR"]})


def test_bspline_partition_of_unity():
    """B-Spline-Drift: 6 Regressoren, geklemmte Basis bildet Zerlegung der Eins."""
    from drift_glm.analysis import sweep
    dm = sweep._bspline_dm(_fake_conc(), 6)
    common = dm.common
    assert common.sizes["regressor"] == 6
    assert list(common.dims) == ["time", "regressor", "chromo"]
    total = common.sel(chromo="HbO").sum("regressor").values
    assert np.allclose(total, 1.0, atol=1e-6)
    assert all(str(r).startswith("Drift BS ") for r in common.regressor.values)


def test_drift_family_dispatch():
    """drift_dm liefert fuer jede Familie eine DesignMatrix mit erwartetem Praefix."""
    from drift_glm.analysis import sweep
    conc = _fake_conc()
    checks = {
        "poly:2": "Drift ", "legendre:3": "Drift LP ", "bspline:5": "Drift BS ",
        "none": "Drift ",
    }
    for fam, prefix in checks.items():
        dm, filt = sweep.drift_dm(fam, conc)
        assert filt is None
        assert any(str(r).startswith(prefix) for r in dm.common.regressor.values)


def test_filter_families_return_cutoffs_and_no_drift():
    """Filter-Familien liefern Filtergrenzen und nur den Offset als Drift-Regressor."""
    from drift_glm.analysis import sweep
    conc = _fake_conc()
    expected = {
        "butter:0.01": (0.01, 0.0),          # Hochpass (fmax=0)
        "lowpass:0.5": (0.0, 0.5),           # Tiefpass (fmin=0)
        "bandpass:0.01-0.5": (0.01, 0.5),    # Bandpass
    }
    for fam, want in expected.items():
        dm, filt = sweep.drift_dm(fam, conc)
        assert filt == pytest.approx(want)
        # nur der Offset (drift_order=0) -> genau ein Drift-Regressor
        drifts = [r for r in dm.common.regressor.values if str(r).startswith("Drift")]
        assert len(drifts) == 1


# --------------------------------------------------- Integration (laedt Daten)

@pytest.fixture(scope="module")
def rec():
    import cedalion.data
    return cedalion.data.get_nn22_resting_state()


@pytest.fixture(scope="module")
def pre(rec):
    """Preprocessing einmal je Testlauf, Default-Motion-Stufe."""
    from drift_glm.core import preprocess as prep
    return prep.run(rec)


@pytest.fixture(scope="module")
def stage(rec):
    """Erste Haelfte der Kette (Rohamplitude -> OD), einmal je Testlauf."""
    from drift_glm.core import preprocess as prep
    return prep.to_od_stage(rec)


@pytest.fixture(scope="module")
def P(stage):
    from drift_glm.core import pipeline as pl
    return pl.build(window_s=90.0, stage=stage, seed=0)


# ------------------------------------------------------------- Preprocessing

def test_od_roundtrip_is_exact(rec):
    """Ohne Motion Correction gilt od2int(int2od(amp)) == amp."""
    from drift_glm.core import preprocess as prep
    amp = rec["amp"].pint.dequantify().pint.quantify("V")
    amp, _ = prep.gate_positive(amp)
    od, baseline = prep.to_od(amp)
    back = prep.to_amp(od, baseline)
    rel = np.abs(back.pint.dequantify().values - amp.pint.dequantify().values) \
        / np.abs(amp.pint.dequantify().values)
    assert np.nanmax(rel) < 1e-9


def test_wavelet_preserves_drift_band_but_tddr_does_not(rec):
    """Wavelet laesst das Driftband (<0.01 Hz) unveraendert, TDDR daempft es deutlich."""
    from drift_glm.core import preprocess as prep
    amp = rec["amp"].pint.dequantify().pint.quantify("V")
    amp, _ = prep.gate_positive(amp)
    # Kanal-Subset; ein voller TDDR-Lauf waere fuer einen Smoke-Test zu teuer (~2 min).
    od, _ = prep.to_od(amp.isel(channel=slice(0, 40)))
    drift = list(prep.DRIFT_BANDS)[0][0]
    assert prep.band_power_ratio(od, prep.motion_correct(od, "wavelet"))[drift] > 0.95
    assert prep.band_power_ratio(od, prep.motion_correct(od, "tddr"))[drift] < 0.90


def test_quality_masks_drop_dark_and_saturated(rec, pre):
    """mean_amp verwirft dunkle/gesaettigte Kanaele, SNR=3 ist dagegen permissiv."""
    from drift_glm.core import preprocess as prep

    def n_keep(mask):
        keep = mask.all(dim=[d for d in mask.dims if d != "channel"])
        return int(keep.sum())

    n = pre.amp_corr.sizes["channel"]
    assert n_keep(pre.masks["snr"]) > n_keep(pre.masks["mean_amp"])
    assert n_keep(pre.masks["mean_amp"]) < n          # verwirft tatsaechlich etwas
    # Die Kette verwirft Kanaele, behaelt aber die grosse Mehrheit.
    assert pre.conc.sizes["channel"] < rec["amp"].sizes["channel"]
    assert pre.conc.sizes["channel"] > 0.85 * rec["amp"].sizes["channel"]
    assert len(pre.dropped) == rec["amp"].sizes["channel"] - pre.conc.sizes["channel"]


def test_dark_noise_floor_justifies_threshold(pre):
    """Die Untergrenze 1e-3 V liegt weit ueber dem gemessenen Rauschboden."""
    from drift_glm.core import preprocess as prep
    nf = prep.dark_noise_floor(pre.aux)
    assert nf is not None and nf > 0
    assert 1e-3 / nf > 10        # mindestens eine Groessenordnung darueber


def test_beta_true_map_is_spatial_blob(P):
    """Ground Truth ist raeumlich variabel (Blob), Peak ~ beta_true, HbR invers."""
    btm = P.beta_true_map
    assert set(btm.dims) == {"channel", "chromo"}
    hbo = btm.sel(chromo="HbO").values
    assert np.isclose(np.nanmax(hbo), P.beta_true["HbO"], rtol=0.05)
    assert np.nanstd(hbo) > 0                      # nicht flach -> echter Blob
    # inverse HbO/HbR-Beziehung (gleiches raeumliches Muster, umgekehrtes Vorzeichen)
    hbr = btm.sel(chromo="HbR").values
    assert np.nanmin(hbr) < 0 < np.nanmax(hbo)


def test_activation_matches_beta_true_map(P):
    """Zeit-Peak der eingemischten Aktivierung je Kanal ist exakt |beta_true_map|."""
    for c in ("HbO", "HbR"):
        want = np.abs(P.beta_true_map.sel(chromo=c))
        got = np.abs(P.activation.sel(chromo=c)).max("time").sel(channel=want.channel)
        assert np.allclose(got.values, want.values, rtol=1e-6, atol=1e-9), (
            f"{c}: Injektion und GT-Karte kanalweise inkonsistent "
            f"(max |diff| = {float(np.abs(got - want).max()):.4f} µM)")


def test_hrf_regressor_peak_normalized(P):
    """Der HRF-Regressor ist auf Peak 1 normiert -> beta == injizierte Peak-Amplitude."""
    hrf = P.dm_hrf.common.sel(regressor=P.hrf_names[0])
    assert np.isclose(float(np.abs(hrf).max()), 1.0, atol=1e-6)


def test_injection_survives_od_roundtrip_without_correction(stage):
    """Ohne Motion Correction ist conc_syn - conc exakt die eingemischte Aktivierung."""
    from drift_glm.core import pipeline as pl
    P = pl.build(window_s=90.0, stage=stage, motion_method="none", seed=0)
    got = (P.conc_syn - P.conc).transpose(*P.activation.dims)
    want = P.activation
    rel = float(np.abs(got - want).max()) / float(np.abs(want).max())
    assert rel < 1e-9


def test_tddr_attenuates_the_injected_hrf(stage):
    """TDDR daempft die vor der Korrektur eingemischte HRF deutlich, Wavelet nicht."""
    from dataclasses import replace

    from drift_glm.core import pipeline as pl

    # Kanal-Subset; ein voller TDDR-Lauf waere fuer einen Smoke-Test zu teuer (~3 min je Build).
    small = replace(stage,
                    od=stage.od.isel(channel=slice(0, 60)),
                    amp_raw=stage.amp_raw.isel(channel=slice(0, 60)),
                    baseline=stage.baseline.isel(channel=slice(0, 60)))

    def peak_ratio(method):
        P = pl.build(window_s=90.0, stage=small, motion_method=method, seed=0)
        got = (P.conc_syn - P.conc).transpose(*P.activation.dims)
        w = np.asarray(P.activation.sel(chromo="HbO").values, float)
        g = np.asarray(got.sel(chromo="HbO").values, float)
        j = np.unravel_index(np.argmax(np.abs(w)), w.shape)
        return g[j] / w[j]

    assert peak_ratio("wavelet") > 0.95     # laesst die HRF praktisch unangetastet
    assert peak_ratio("tddr") < 0.85        # daempft sie deutlich
