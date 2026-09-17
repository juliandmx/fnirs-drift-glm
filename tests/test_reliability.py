"""Split-half regression must model the trials in both halves simultaneously."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def synthetic_joint(monkeypatch):
    import cedalion.models.glm as glm
    from drift_glm.analysis import msglm
    from drift_glm.core import pipeline as pl

    time = np.arange(0.0, 160.0, 0.25)
    ts = xr.DataArray(
        np.zeros((len(time), 6, 2)), dims=("time", "channel", "chromo"),
        coords={"time": time, "channel": [f"S1D{i}" for i in range(6)],
                "chromo": ["HbO", "HbR"]})
    ts = ts.assign_coords(samples=("time", np.arange(len(time))))
    ts.time.attrs["units"] = "s"
    stim = pd.DataFrame(dict(
        onset=5.0 + 5.5 * np.arange(18), duration=5.0, value=1.0,
        trial_type=list(msglm.ms.CONDITIONS) * 6))
    joint = msglm.joint_half_stimuli(stim)
    hrf = glm.design_matrix.hrf_regressors(ts, joint, msglm._basis())
    hrf, _ = pl._normalize_hrf_to_unit_peak(hrf, hrf.regressors)
    dm = hrf & glm.design_matrix.drift_regressors(ts, drift_order=0)
    rng = np.random.default_rng(315)
    values = rng.normal(size=(6, len(dm.regressors), 2))
    beta = xr.DataArray(values, dims=("channel", "regressor", "chromo"),
                        coords={"channel": ts.channel, "regressor": dm.regressors,
                                "chromo": ts.chromo})
    signal = glm.predict(ts, beta, dm).transpose(*ts.dims)
    signal = signal.assign_coords(samples=("time", np.arange(len(time))))
    signal.time.attrs["units"] = "s"
    monkeypatch.setattr(msglm.cedalion.nirs, "split_long_short_channels",
                        lambda conc, _geo, **_kw: (conc, conc.isel(channel=slice(0, 0))))
    return signal, stim, beta, dm


def test_joint_fit_recovers_both_halves_while_omitting_trials_biases_them(synthetic_joint):
    import cedalion.models.glm as glm
    from drift_glm.analysis import msglm
    from drift_glm.core import pipeline as pl

    signal, stim, true_beta, _ = synthetic_joint
    got = msglm.first_level(signal, None, stim, "none", "none", "ols", half="joint")
    for name in got.regressor.values:
        np.testing.assert_allclose(
            got.sel(trial_type=str(name).removeprefix("HRF ")),
            true_beta.sel(regressor=name), atol=1e-10, rtol=1e-10)

    labelled = msglm.joint_half_stimuli(stim)
    omitted = labelled[labelled.trial_type.str.endswith(" [even]")]
    hrf = glm.design_matrix.hrf_regressors(signal, omitted, msglm._basis())
    hrf, _ = pl._normalize_hrf_to_unit_peak(hrf, hrf.regressors)
    dm = hrf & glm.design_matrix.drift_regressors(signal, drift_order=0)
    biased = glm.fit(signal, dm, noise_model="ols").sm.params
    names = hrf.regressors
    error = np.abs(biased.sel(regressor=names) - true_beta.sel(regressor=names))
    assert float(error.max()) > 0.1


def test_joint_split_preserves_all_events_and_condition_correspondence(synthetic_joint):
    from drift_glm.analysis import msglm

    signal, stim, _, _ = synthetic_joint
    shuffled = stim.sample(frac=1, random_state=5)
    before = shuffled.copy(deep=True)
    labelled = msglm.joint_half_stimuli(shuffled)
    pd.testing.assert_frame_equal(shuffled, before)
    assert len(labelled) == len(stim)
    assert labelled.trial_type.nunique() == 6
    for condition in msglm.ms.CONDITIONS:
        rows = labelled[labelled.trial_type.str.startswith(condition)].sort_values("onset")
        assert rows.trial_type.str.endswith(" [even]").tolist() == [True, False] * 3
    joint = msglm.first_level(signal, None, stim, "none", "none", "ols", half="joint")
    even, odd = msglm.split_joint_betas(joint)
    np.testing.assert_array_equal(even.trial_type, odd.trial_type)
    assert even.sizes["trial_type"] == odd.sizes["trial_type"] == 3
    selected = msglm.first_level(signal, None, stim, "none", "none", "ols", half="even")
    xr.testing.assert_allclose(even, selected)


def test_full_fit_is_separate_from_joint_estimation(synthetic_joint):
    """Reliability changes may not change the original three-condition full fit."""
    import cedalion.models.glm as glm
    from drift_glm.analysis import msglm
    from drift_glm.core import pipeline as pl

    signal, stim, _, _ = synthetic_joint
    hrf = glm.design_matrix.hrf_regressors(signal, stim, msglm._basis())
    hrf, _ = pl._normalize_hrf_to_unit_peak(hrf, hrf.regressors)
    dm = hrf & glm.design_matrix.drift_regressors(signal, drift_order=0)
    expected = glm.fit(signal, dm, noise_model="ols").sm.params.sel(regressor=hrf.regressors)
    got = msglm.first_level(signal, None, stim, "none", "none", "ols")
    assert got.sizes["trial_type"] == 3
    np.testing.assert_allclose(got.values, expected.values, rtol=0, atol=0)


def test_reliability_correlates_each_subject_before_taking_median():
    from drift_glm.analysis import msglm

    def beta(values):
        return xr.DataArray(np.asarray(values).reshape(4, 1, 1),
                            dims=("channel", "trial_type", "chromo"),
                            coords={"channel": range(4), "trial_type": ["test"],
                                    "chromo": ["HbO"]})
    pairs = [(beta([1, 2, 3, 4]), beta([1, 2, 3, 4])),
             (beta([10, 20, 30, 40]), beta([40, 30, 20, 10])),
             (beta([1, 2, 3, 4]), beta([1, 3, 2, 4]))]
    got, count = msglm.half_reliability(pairs)
    assert count == 3
    assert got == pytest.approx(0.8)
    assert got != pytest.approx(np.corrcoef(
        np.concatenate([a.values.ravel() for a, _ in pairs]),
        np.concatenate([b.values.ravel() for _, b in pairs]))[0, 1])


def _summary_cell():
    from drift_glm.analysis import msglm
    return pd.DataFrame([
        dict(family="none", systemic="none", noise_model="ols", trial_type=condition,
             chromo=chromo, n_subjects=5, beta_mean=0.25, n_significant=3,
             reliability_r=0.9, n_half_pairs=5, reliability_method=msglm.RELIABILITY_METHOD,
             loc_err_mm=(np.nan if condition == "control" else 15.0),
             loc_err_wrong_mm=(np.nan if condition == "control" else 80.0),
             peak_uM=(np.nan if condition == "control" else 0.1))
        for condition in msglm.ms.CONDITIONS for chromo in ("HbO", "HbR")])


def test_resume_rejects_incomplete_legacy_or_image_incomplete_cells():
    from drift_glm.analysis import msglm
    frame = _summary_cell()
    options = dict(with_halves=True, with_image=True, n_subjects=5)
    assert msglm.complete_cells(frame, **options) == {("none", "none", "ols")}
    assert not msglm.complete_cells(frame.iloc[:5], **options)
    assert not msglm.complete_cells(pd.concat([frame, frame.iloc[:1]]), **options)
    assert not msglm.complete_cells(frame.drop(columns="reliability_method"), **options)
    incomplete = frame.copy()
    incomplete.loc[incomplete.trial_type == "Tapping/Right", "loc_err_mm"] = np.nan
    assert not msglm.complete_cells(incomplete, **options)


def test_reliability_refresh_asserts_unchanged_full_and_image_results():
    from drift_glm.analysis import msglm
    old = _summary_cell().drop(columns="reliability_method")
    updated = old.copy()
    updated["reliability_r"] = -0.2
    updated["reliability_method"] = msglm.RELIABILITY_METHOD
    msglm.assert_full_metrics_unchanged(updated, old)
    for column in ("beta_mean", "n_significant", "loc_err_mm"):
        changed = updated.copy()
        changed.loc[changed.trial_type == "Tapping/Right", column] += 1
        with pytest.raises(AssertionError):
            msglm.assert_full_metrics_unchanged(changed, old)


def test_primary_grid_is_81_cells_and_control_never_runs_ar_irls():
    from drift_glm.analysis import msglm
    grids = [msglm.grid_for(noise, msglm.FAMILIES, msglm.SYSTEMIC)
             for noise in msglm.NOISE_MODELS]
    assert sum(len(f) * len(s) for f, s in grids) == 81
    assert msglm.grid_for("ar_irls", msglm.SUPPLEMENTAL_FAMILIES, msglm.SYSTEMIC)[0] == []
