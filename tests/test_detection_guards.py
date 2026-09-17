"""Invalid inference values are counted and excluded, never silently converted."""
import numpy as np
import xarray as xr

from drift_glm.analysis import detection as det


class _SM:
    def __init__(self, beta, var):
        self._beta, self._var = beta, var

    @property
    def params(self):
        return self._beta

    def regressor_variances(self):
        return self._var


class _Res:
    def __init__(self, beta, var):
        self.sm = _SM(beta, var)


def _arr(values):
    return xr.DataArray(np.asarray(values, float)[:, None, None],
                        dims=("channel", "chromo", "regressor"),
                        coords={"channel": [f"c{i}" for i in range(len(values))],
                                "chromo": ["HbO"], "regressor": ["HRF Stim"]})


def test_negative_zero_and_nonfinite_variances_become_invalid_not_significant():
    res = _Res(_arr([1.0, 1.0, 1.0, 1.0, 1.0]), _arr([0.04, -0.04, 0.0, np.nan, np.inf]))
    t = det._hrf_tvalues(res)
    vals = np.asarray(t.sel(chromo="HbO").values, float)
    assert vals[0] == 5.0
    assert np.isnan(vals[1:]).all()
    assert det.invalid_inference(t) == 4
    p = 2 * det.norm.sf(np.abs(vals))
    m = det._metrics(p, np.array([True, True, False, False, False]))
    assert m["n_invalid"] == 4 and m["n_detected"] == 1 and m["TP"] == 1 and m["FP"] == 0
