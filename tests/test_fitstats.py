"""Tests fuer die Modellfit-Metriken (fitstats) und die neuen Analyse-Bausteine.

Alle Tests hier sind schnell (synthetische Mini-Daten, kein Datendownload, kein
AR-IRLS) -- die Invarianten sind mathematisch exakt pruefbar.
"""
import numpy as np
import pytest
import xarray as xr


def _ts(nt=200, nch=2, seed=0):
    """Mini-Zeitreihe (time, channel, chromo) mit Rauschen."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 20.0, nt)
    return xr.DataArray(
        rng.normal(size=(nt, nch, 2)), dims=("time", "channel", "chromo"),
        coords={"time": t, "channel": [f"S1D{i}" for i in range(nch)],
                "chromo": ["HbO", "HbR"]})


def _dm_common(ts, k=3):
    """DesignMatrix mit k Regressoren (Offset + Sinus/Cosinus), fuer alle Chromo gleich."""
    import cedalion.models.glm as glm
    t = ts.time.values
    cols = [np.ones_like(t)]
    for i in range(1, k):
        cols.append(np.sin(2 * np.pi * i * t / t[-1]))
    da = xr.DataArray(
        np.stack(cols, axis=1), dims=("time", "regressor"),
        coords={"time": ts.time, "regressor": [f"Reg {i}" for i in range(k)]},
    ).expand_dims({"chromo": ["HbO", "HbR"]}).transpose("time", "regressor", "chromo")
    return glm.design_matrix.DesignMatrix(common=da)


def _betas(dm, ts, values):
    """Beta-DataArray (channel, regressor, chromo) mit konstantem Wert je Regressor."""
    k = dm.common.sizes["regressor"]
    arr = np.tile(np.asarray(values, float)[None, :, None],
                  (ts.sizes["channel"], 1, 2))
    return xr.DataArray(
        arr, dims=("channel", "regressor", "chromo"),
        coords={"channel": ts.channel.values,
                "regressor": dm.common.regressor.values,
                "chromo": ["HbO", "HbR"]})


def test_perfect_fit_has_r2_one_and_zero_residual():
    """Ist y exakt X@beta, muss R^2 = 1 und Residual-RMS = 0 sein."""
    import cedalion.models.glm as glm
    from drift_glm.core import fitstats as fs
    ts = _ts()
    dm = _dm_common(ts)
    betas = _betas(dm, ts, [0.5, 2.0, -1.0])
    y = glm.predict(ts, betas, dm).transpose(*ts.dims)
    y = y.assign_coords({k: ts[k] for k in ts.coords})
    q = fs.fit_metrics(y, betas, dm)
    assert float(q.r2.min()) > 1.0 - 1e-9
    assert float(q.resid_rms.max()) < 1e-9
    assert q.attrs["n_regressors"] == 3


def test_zero_model_has_r2_zero():
    """Sagt das Modell konstant den Mittelwert (hier 0) voraus, ist R^2 ~ 0."""
    from drift_glm.core import fitstats as fs
    ts = _ts()
    ts = ts - ts.mean("time")
    dm = _dm_common(ts)
    q = fs.fit_metrics(ts, _betas(dm, ts, [0.0, 0.0, 0.0]), dm)
    assert abs(float(q.r2.mean())) < 1e-9
    # Residual-RMS == RMS der Daten selbst
    want = float(np.sqrt((ts ** 2).mean("time")).max())
    got = float(q.resid_rms.max())
    assert np.isclose(got, want, rtol=1e-9)


def test_r2_adj_penalizes_more_regressors():
    """adj. R^2 liegt unter R^2, und die Differenz waechst mit der Spaltenzahl."""
    from drift_glm.core import fitstats as fs
    ts = _ts(nt=120)
    gaps = {}
    for k in (2, 30):
        dm = _dm_common(ts, k=k)
        q = fs.fit_metrics(ts, _betas(dm, ts, [0.1] * k), dm)
        gaps[k] = float((q.r2 - q.r2_adj).mean())
        assert gaps[k] > 0
    assert gaps[30] > gaps[2]


def test_n_regressors_counts_channel_wise():
    """Kanalweise Regressoren (short_maxcorr/closest) zaehlen je Kanal einmal."""
    import cedalion.models.glm as glm
    from drift_glm.core import fitstats as fs
    ts = _ts()
    dm = _dm_common(ts, k=3)
    cw = xr.DataArray(
        np.zeros((ts.sizes["time"], 1, 2, ts.sizes["channel"])),
        dims=("time", "regressor", "chromo", "channel"),
        coords={"time": ts.time, "regressor": ["short"],
                "chromo": ["HbO", "HbR"], "channel": ts.channel.values})
    dm2 = glm.design_matrix.DesignMatrix(common=dm.common, channel_wise=[cw])
    assert fs.n_regressors(dm) == 3
    assert fs.n_regressors(dm2) == 4


def test_residuals_dequantify_handles_pint():
    """Quantifizierte Zeitreihe (µM) -> Residuen unitless, Dims wie die Daten."""
    from drift_glm.core import fitstats as fs
    ts = _ts()
    tsq = ts.pint.quantify("micromolar")
    dm = _dm_common(tsq)
    r = fs.residuals(tsq, _betas(dm, ts, [0.0, 0.0, 0.0]), dm)
    assert r.dims == ts.dims
    assert getattr(r, "pint", None) is None or r.pint.units is None
    assert np.allclose(r.values, ts.values)


def test_lowfreq_frac_separates_drift_from_noise():
    """Reines Niederfrequenz-Residuum -> Anteil ~1; reines Hochfrequentes -> ~0."""
    from drift_glm.analysis import residuals as ra
    nt = 4096
    fs_hz = 8.0
    t = np.arange(nt) / fs_hz
    lo = np.sin(2 * np.pi * 0.005 * t)
    hi = np.sin(2 * np.pi * 1.0 * t)
    da = xr.DataArray(
        np.stack([np.stack([lo, hi], axis=1)], axis=1),
        dims=("time", "channel", "chromo"),
        coords={"time": t, "channel": ["S1D1"], "chromo": ["HbO", "HbR"]})
    frac = ra._lowfreq_frac(da, fs_hz).squeeze("channel")
    assert float(frac.sel(chromo="HbO")) > 0.9      # 0.005 Hz liegt im Driftband
    assert float(frac.sel(chromo="HbR")) < 0.1      # 1 Hz liegt weit darueber


def test_figstyle_colors_are_stable_and_ordered():
    """Eine Familie -> immer dieselbe Farbe; order() haelt die kanonische Reihenfolge."""
    from drift_glm import figstyle
    for fam in figstyle.FAMILY_ORDER:
        assert figstyle.family_color(fam) == figstyle.family_color(fam)
    shuffled = ["butter:0.01", "poly:1", "none", "dct:0.01"]
    assert figstyle.order(shuffled) == ["none", "poly:1", "dct:0.01", "butter:0.01"]
    # unbekannte Familien fallen ans Ende statt einen KeyError zu werfen
    assert figstyle.order(["poly:1", "neu:x"]) == ["poly:1", "neu:x"]
    assert figstyle.family_color("neu:x") is not None
