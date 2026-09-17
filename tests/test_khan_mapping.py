"""Khan et al. (2026) Table 3 hemisphere mapping and its agreement with the SNIRF files."""
import numpy as np
import pytest

from drift_glm.data import realdata as rd


def test_table3_has_24_channels_per_hemisphere_and_consistent_labels():
    t = rd.khan_channel_table()
    assert len(t) == 48 and t.channel.is_unique
    assert (t.hemisphere == "left").sum() == 24 and (t.hemisphere == "right").sum() == 24
    # left-hemisphere 10-10 labels carry odd numbers, right-hemisphere labels even numbers
    for _, row in t.iterrows():
        for lab in (row.source_1010, row.detector_1010):
            digit = int(lab[-1])
            assert (digit % 2 == 1) == (row.hemisphere == "left"), (row.channel, lab)
    np.testing.assert_array_equal(rd.khan_hemisphere(["S1D1", "S16D16", "S8D8", "S9D9"]),
                                  [1, -1, 1, -1])
    with pytest.raises(ValueError):
        rd.khan_hemisphere(["S1D9"])


def test_snirf_channel_order_matches_table3():
    files = rd.find_files()
    if not files:
        pytest.skip("Khan dataset not available locally")
    rec = rd.load(files[0])
    key = "amp" if "amp" in rec.timeseries else list(rec.timeseries)[0]
    channels = [str(c) for c in rec[key].channel.values]
    table = rd.khan_channel_table(channels)
    assert table.channel.tolist() == channels
    assert table.ch_no.tolist() == list(range(1, 49))
    # the two hemispheres form two spatial clusters in the file's coordinate frame
    g = rec.geo3d.pint.dequantify() if rec.geo3d.pint.units is not None else rec.geo3d
    pos = {str(l): np.asarray(v, float) for l, v in zip(g.label.values, g.values)}
    x_left = [pos[s][0] for s in [f"S{i}" for i in range(1, 9)] + [f"D{i}" for i in range(1, 9)]]
    x_right = [pos[s][0] for s in [f"S{i}" for i in range(9, 17)] + [f"D{i}" for i in range(9, 17)]]
    assert min(x_left) > max(x_right) or max(x_left) < min(x_right)
