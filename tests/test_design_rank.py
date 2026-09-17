"""Rank, DCT intercept and adjusted-R² regression checks."""
import numpy as np
import pytest
import xarray as xr

import cedalion.models.glm as glm
from cedalion import units
from drift_glm.analysis.sweep import drift_dm
from drift_glm.core.fitstats import design_ranks, fit_metrics, n_regressors


def _series(seconds=180, fs=9.0):
    t = np.arange(int(seconds * fs)) / fs
    ts = xr.DataArray(np.zeros((len(t), 1, 2)), dims=("time", "channel", "chromo"),
                      coords={"time": t, "channel": ["S1D1"], "chromo": ["HbO", "HbR"]})
    ts = ts.assign_coords(samples=("time", np.arange(len(t))))
    ts.time.attrs["units"] = "s"
    return ts


@pytest.mark.parametrize("seconds,cutoff,columns", [(90, .005, 1), (90, .01, 1),
                                                    (180, .01, 3), (368, .02, 14)])
def test_dct_has_exactly_one_intercept_even_when_native_basis_is_empty(seconds, cutoff, columns):
    ts = _series(seconds)
    dm, filt = drift_dm(f"dct:{cutoff}", ts)
    assert filt is None
    assert n_regressors(dm) == columns
    assert int(design_ranks(ts, dm).min()) == columns
    a = dm.common.sel(chromo="HbO").values
    assert np.sum(np.all(a == a[0:1], axis=0)) == 1


def test_rank_is_invariant_to_column_scale_and_duplicate_constant_is_rejected():
    ts = _series()
    dm = glm.design_matrix.drift_regressors(ts, drift_order=5)
    dm.common.values *= np.array([1e-25, 1e-15, 1e-5, 1e5, 1e15, 1e25])[None, :, None]
    assert int(design_ranks(ts, dm).min()) == 6
    duplicate = glm.design_matrix.drift_cosine_regressors(ts, .01 * units.Hz)
    duplicate = duplicate & glm.design_matrix.drift_regressors(ts, drift_order=0)
    with pytest.raises(ValueError, match="Design rank 3 < 4"):
        design_ranks(ts, duplicate)


def test_rank_checks_channel_wise_regressors_too():
    ts = _series()
    dm, _ = drift_dm("none", ts)
    cw = dm.common.copy()
    cw = cw.assign_coords(regressor=["constant short"]).expand_dims(channel=ts.channel)
    dm.channel_wise = [cw]
    with pytest.raises(ValueError, match="Design rank 1 < 2"):
        design_ranks(ts, dm)


def test_removing_duplicate_constant_preserves_hrf_and_predictions():
    ts = _series()
    t = ts.time.values
    hrf = xr.DataArray(np.sin(2 * np.pi * .03 * t), dims="time", coords={"time": ts.time})
    hrf = hrf.expand_dims(regressor=["HRF"], chromo=ts.chromo).transpose(
        "time", "regressor", "chromo")
    task = glm.design_matrix.DesignMatrix(common=hrf)
    old_drift = glm.design_matrix.drift_cosine_regressors(ts, .02 * units.Hz)
    old = task & old_drift & glm.design_matrix.drift_regressors(ts, drift_order=0)
    new = task & drift_dm("dct:0.02", ts)[0]
    y = ts + (1.2 * hrf.sel(regressor="HRF", drop=True) + .2
              + .3 * np.sin(2 * np.pi * .002 * t)[:, None])
    old_b = glm.fit(y, old, noise_model="ols").sm.params
    new_b = glm.fit(y, new, noise_model="ols").sm.params
    np.testing.assert_allclose(old_b.sel(regressor="HRF"), new_b.sel(regressor="HRF"), atol=1e-8)
    np.testing.assert_allclose(glm.predict(y, old_b, old), glm.predict(y, new_b, new), atol=1e-8)
    result = fit_metrics(y, new_b, new)
    assert result.attrs["design_rank_min"] == n_regressors(new)
    expected = 1 - (1 - result.r2) * (len(t) - 1) / (len(t) - n_regressors(new))
    xr.testing.assert_allclose(result.r2_adj, expected)


def test_offline_migration_uses_each_window_and_seed_chromophore_value(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from drift_glm.analysis import sweep
    monkeypatch.setattr(sweep, "RESULTS", tmp_path)
    class Record(dict):
        aux_ts = {}
    monkeypatch.setattr(sweep.cedalion.data, "get_nn22_resting_state",
                        lambda: Record(amp=_series(368)))
    coords = dict(family=["dct:0.005", "dct:0.01"], window_s=[90., 180.],
                  constellation=["baseline", "short_avg"], motion=["wavelet"],
                  noise_model=["ar_irls"], seed=[0, 1], channel=["S1D1"],
                  chromo=["HbO", "HbR"])
    shape = tuple(len(v) for v in coords.values())
    values = np.random.default_rng(10).uniform(-.3, .9, size=shape)
    r2 = xr.DataArray(values, dims=list(coords), coords=coords)
    adj = r2.copy()
    rows = []
    for family in coords["family"]:
        cutoff = float(family.split(":")[1])
        for window in coords["window_s"]:
            k = int(np.floor(2 * window * cutoff))
            n = int(window * 9)
            for constellation in coords["constellation"]:
                old_p = 2 + k + (constellation == "short_avg")
                sel = dict(family=family, window_s=window, constellation=constellation)
                adj.loc[sel] = 1 - (1 - r2.sel(**sel)) * (n - 1) / (n - old_p)
                for chromo in coords["chromo"]:
                    rows.append(dict(**sel, motion="wavelet", noise_model="ar_irls",
                                     chromo=chromo, r2_adj_med=float(adj.sel(**sel, chromo=chromo).median())))
    data = xr.Dataset(dict(r2=r2, r2_adj=adj, bhat=r2.copy()))
    sweep._write_results("sweep", data, pd.DataFrame(rows), dict(config={}))
    counts = sweep.migrate_dct_adjusted_r2()
    revised = xr.load_dataset(tmp_path / "sweep_per_channel.nc")
    summary = pd.read_csv(tmp_path / "sweep_summary.csv")
    for count in counts:
        sel = {key:count[key] for key in ("family", "window_s", "constellation")}
        expected = 1 - (1 - r2.sel(**sel)) * (count["n_samples"] - 1) / (
            count["n_samples"] - count["corrected_columns"])
        xr.testing.assert_allclose(revised.r2_adj.sel(**sel), expected)
    # Native K=0 has no duplicated constant and must remain bit-for-bit unchanged.
    np.testing.assert_array_equal(revised.r2_adj.sel(family="dct:0.005", window_s=90).values,
                                  adj.sel(family="dct:0.005", window_s=90).values)
    xr.testing.assert_identical(revised.bhat, data.bhat)
    for row in summary.to_dict("records"):
        keys = {k:v for k,v in row.items() if k != "r2_adj_med"}
        assert row["r2_adj_med"] == pytest.approx(float(revised.r2_adj.sel(**keys).median()))
    metadata = json.loads((tmp_path / "sweep_meta.json").read_text())
    assert len(metadata["dct_adjusted_r2_migration"]["archive"]) == 3
    with pytest.raises(ValueError, match="already applied"):
        sweep.migrate_dct_adjusted_r2()
