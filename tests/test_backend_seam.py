"""The backend is intentionally unimplemented (CLAUDE.md s3). These tests pin the
*seam*: the input/output contract is stable and the estimator raises clearly
until the author writes it. They must NOT be "fixed" by implementing the graph.
"""
import numpy as np
import pytest

from nuslam.backend import MonocularSLAM, SlamEstimate, SlamInputs
from nuslam.backend.factors import GroundPlaneFactor, WheelContactToPlaneFactor
from nuslam.types import CameraCalib, TrackSet
from nuslam.transforms import SE3


def _calib() -> CameraCalib:
    K = np.array([[1200.0, 0, 800], [0, 1200.0, 450], [0, 0, 1]])
    return CameraCalib("CAM_FRONT", K, SE3.identity(), 1600, 900)


def test_factors_are_unimplemented():
    with pytest.raises(NotImplementedError):
        GroundPlaneFactor()
    with pytest.raises(NotImplementedError):
        WheelContactToPlaneFactor()


def test_run_is_unimplemented():
    inputs = SlamInputs(
        calib=_calib(),
        keyframes=[],
        tracks=TrackSet(["a", "b"], np.zeros((2, 3, 2), np.float32), np.ones((2, 3), bool), 0),
    )
    with pytest.raises(NotImplementedError):
        MonocularSLAM(_calib()).run(inputs)


def test_estimate_validates_shape():
    with pytest.raises(ValueError):
        SlamEstimate(tokens=["a"], poses_ego2global=np.zeros((2, 4, 4)))  # length mismatch
    with pytest.raises(ValueError):
        SlamEstimate(tokens=["a"], poses_ego2global=np.zeros((1, 3, 3)))  # wrong shape
    # valid one is accepted
    ok = SlamEstimate(tokens=["a"], poses_ego2global=np.eye(4)[None])
    assert ok.poses_ego2global.shape == (1, 4, 4)
