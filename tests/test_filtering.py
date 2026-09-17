"""The control filters the complete design and recovers known coefficients."""
import numpy as np
import pytest
import xarray as xr

import cedalion.models.glm as glm
from cedalion import units
from drift_glm.core.filtering import apply_filter


def _example(channel_wise=False):
    t = np.arange(2048) / 4.0
    hrf = np.sin(2 * np.pi * .006 * t) + .4 * np.sin(2 * np.pi * .04 * t)
    motion = np.cos(2 * np.pi * .018 * t)
    common = xr.DataArray(np.stack([hrf, motion, np.ones_like(t)], axis=1),
                          dims=("time", "regressor"),
                          coords={"time": t, "regressor": ["HRF", "Motion", "Offset"]})
    common = common.expand_dims(chromo=["HbO", "HbR"]).transpose(
        "time", "regressor", "chromo")
    common = common.assign_coords(samples=("time", np.arange(len(t))))
    ts = xr.DataArray(np.zeros((len(t), 2, 2)), dims=("time", "channel", "chromo"),
                      coords={"time": t, "channel": ["S1D1", "S1D2"],
                              "chromo": ["HbO", "HbR"]})
    ts = ts.assign_coords(samples=("time", np.arange(len(t))))
    ts.time.attrs["units"] = "s"
    common.time.attrs["units"] = "s"
    cw = []
    if channel_wise:
        vals = np.stack([np.sin(2 * np.pi * .025 * t),
                         np.cos(2 * np.pi * .03 * t)], axis=1)
        reg = xr.DataArray(vals[:, :, None, None] * np.ones((1, 1, 1, 2)),
                           dims=("time", "channel", "regressor", "chromo"),
                           coords={"time": t, "channel": ts.channel,
                                   "regressor": ["Short"], "chromo": ts.chromo})
        reg = reg.assign_coords(samples=("time", np.arange(len(t))))
        cw = [reg]
    dm = glm.design_matrix.DesignMatrix(common=common, channel_wise=cw)
    values = np.array([1.7, -.3, 0.0] + ([.6] if channel_wise else []))
    betas = xr.DataArray(np.tile(values[None, :, None], (2, 1, 2)),
                         dims=("channel", "regressor", "chromo"),
                         coords={"channel": ts.channel, "regressor": dm.regressors,
                                 "chromo": ts.chromo})
    y = glm.predict(ts, betas, dm).transpose(*ts.dims)
    y = y.assign_coords(samples=("time", np.arange(len(t))))
    y.time.attrs["units"] = "s"
    return y, dm, values


@pytest.mark.parametrize("channel_wise", [False, True])
def test_consistent_filter_recovers_true_hrf_but_data_only_does_not(channel_wise):
    y, dm, truth = _example(channel_wise)
    fy, fdm = apply_filter(y, dm, (.01, 0), filter_design=True)
    estimates = glm.fit(fy, fdm, noise_model="ols").sm.params
    np.testing.assert_allclose(estimates.sel(regressor="HRF"), truth[0], atol=1e-10)
    for i, name in enumerate(dm.regressors):
        np.testing.assert_allclose(estimates.sel(regressor=name), truth[i], atol=1e-10)
    legacy_y, legacy_dm = apply_filter(y, dm, (.01, 0))
    legacy = glm.fit(legacy_y, legacy_dm, noise_model="ols").sm.params
    assert float(abs(legacy.sel(regressor="HRF") - truth[0]).min()) > .5
    xr.testing.assert_identical(legacy_y, y.cd.freq_filter(.01 * units.Hz, 0 * units.Hz, 4))
    assert legacy_dm is dm


def test_all_common_and_channel_specific_regressors_use_identical_filter():
    y, dm, _ = _example(channel_wise=True)
    original = dm.copy()
    _, filtered = apply_filter(y, dm, (.01, 0), filter_design=True)
    for source, result in [(dm.common, filtered.common),
                            (dm.channel_wise[0], filtered.channel_wise[0])]:
        for reg in source.regressor.values:
            old = source.sel(regressor=reg)
            if reg == "Offset":
                xr.testing.assert_identical(old, result.sel(regressor=reg))
                continue
            # Compare directly against Cedalion, using a channel axis for its schema.
            reference = old.copy(deep=True)
            if "channel" not in reference.dims:
                reference = reference.expand_dims(channel=["S1D1"])
            reference.time.attrs["units"] = "s"
            expected = reference.cd.freq_filter(.01 * units.Hz, 0 * units.Hz, 4)
            if "channel" not in old.dims:
                expected = expected.squeeze("channel", drop=True)
            np.testing.assert_allclose(result.sel(regressor=reg), expected.transpose(*old.dims))
    xr.testing.assert_identical(dm.common, original.common)
    xr.testing.assert_identical(dm.channel_wise[0], original.channel_wise[0])


def test_no_filter_returns_inputs_and_consistent_filter_preserves_units():
    y, dm, _ = _example()
    plain_y, plain_dm = apply_filter(y, dm, None)
    assert plain_y is y and plain_dm is dm
    quantified = y.pint.quantify("micromolar")
    filtered, _ = apply_filter(quantified, dm, (.01, 0), filter_design=True)
    assert filtered.pint.units == quantified.pint.units
