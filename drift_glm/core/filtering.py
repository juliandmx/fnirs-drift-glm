"""Concentration-space filtering, including the additional consistent-filter control.

The established filter arms change only y.  ``filter_design=True`` additionally
applies exactly the same filter to every nonconstant design column, including
channel-specific nuisance regressors.  The explicit intercept stays unfiltered.
"""
from __future__ import annotations

import numpy as np
import scipy.signal

from cedalion import units
from cedalion.models.glm.design_matrix import DesignMatrix
from cedalion.sigproc.frequency import sampling_rate


def _filter_columns(da, sos):
    if da is None or not da.sizes.get("regressor", 0):
        return None if da is None else da.copy(deep=True)
    original_units = da.pint.units
    plain = da.pint.dequantify() if original_units is not None else da
    work = plain.transpose("time", ...)
    values = np.asarray(work.values, dtype=float)
    # Test each (regressor, chromophore, channel) time course independently.
    constant = np.all(values == values[0:1], axis=0)
    filtered = scipy.signal.sosfiltfilt(sos, values, axis=0)
    result = work.copy(data=np.where(constant[None, ...], values, filtered))
    result = result.transpose(*da.dims)
    return result.pint.quantify(original_units) if original_units is not None else result


def apply_filter(ts, dm, filt, filter_design=False):
    """Return ``(y, X)`` after optional identical zero-phase filtering.

    ``filt`` is ``None`` or ``(fmin, fmax)`` in Hz, following Cedalion's
    convention: a zero upper/lower bound requests a high/low-pass filter.
    Existing data-only arms are unchanged. Neither input is mutated.
    Consistently filtered designs must retain full column rank.
    """
    if filt is None:
        return ts, dm
    lo, hi = map(float, filt)
    filtered_ts = ts.cd.freq_filter(lo * units.Hz, hi * units.Hz, 4)
    if not filter_design:
        return filtered_ts, dm
    nyquist = float(sampling_rate(ts).to("Hz").magnitude) / 2.0
    if lo == 0:
        sos = scipy.signal.butter(4, hi / nyquist, "low", output="sos")
    elif hi == 0:
        sos = scipy.signal.butter(4, lo / nyquist, "high", output="sos")
    else:
        sos = scipy.signal.butter(4, [lo / nyquist, hi / nyquist],
                                 "bandpass", output="sos")
    filtered_dm = DesignMatrix(
        common=_filter_columns(dm.common, sos),
        channel_wise=[_filter_columns(cw, sos) for cw in dm.channel_wise],
    )
    from drift_glm.core.fitstats import design_ranks
    design_ranks(filtered_ts, filtered_dm, assert_full=True)
    return filtered_ts, filtered_dm
