"""Unit tests for CAN proprio plumbing (GPS resampling).

Synthetic streams so the interpolation is verified without the CAN expansion on
disk; the geometry that consumes these (metric-scale resolution) is tested
separately and is the author's core code, not exercised here.
"""
import numpy as np

from nuslam.data.can_streams import gps_positions_at
from nuslam.types import GpsFix, ProprioStreams


def _linear_stream() -> ProprioStreams:
    # A straight track: xy = (t, 2t) sampled at t = 0,10,20,30 (us).
    gps = [GpsFix(timestamp_us=t, latitude=float("nan"), longitude=float("nan"),
                  global_xy=np.array([float(t), 2.0 * t])) for t in (0, 10, 20, 30)]
    return ProprioStreams(gps=gps)


def test_gps_interp_exact_on_linear_track():
    s = _linear_stream()
    pos, valid = gps_positions_at(s, [0, 5, 15, 30])
    assert valid.all()
    # Linear track -> interpolation is exact.
    np.testing.assert_allclose(pos[:, 0], [0, 5, 15, 30])
    np.testing.assert_allclose(pos[:, 1], [0, 10, 30, 60])


def test_gps_out_of_coverage_is_nan_and_invalid():
    s = _linear_stream()
    pos, valid = gps_positions_at(s, [-5, 10, 35])
    assert list(valid) == [False, True, False]
    assert np.isnan(pos[0]).all() and np.isnan(pos[2]).all()
    np.testing.assert_allclose(pos[1], [10, 20])


def test_gps_unsorted_input_is_handled():
    # Same track logged out of timestamp order must still interpolate correctly.
    s = _linear_stream()
    s.gps.reverse()
    pos, valid = gps_positions_at(s, [5, 25])
    assert valid.all()
    np.testing.assert_allclose(pos[:, 0], [5, 25])
    np.testing.assert_allclose(pos[:, 1], [10, 50])


def test_gps_empty_streams_all_nan():
    pos, valid = gps_positions_at(ProprioStreams(), [1, 2, 3])
    assert pos.shape == (3, 2)
    assert not valid.any()
    assert np.isnan(pos).all()


def test_gps_scalar_timestamp_returns_row():
    s = _linear_stream()
    pos, valid = gps_positions_at(s, 10)
    assert pos.shape == (1, 2) and valid.shape == (1,)
    np.testing.assert_allclose(pos[0], [10, 20])
