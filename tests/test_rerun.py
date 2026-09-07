"""Rerun logging smoke tests: exercise every archetype wrapper so a version bump
that changes an archetype signature fails here rather than at viz time. No viewer
or data files needed -- logs into an in-memory recording."""
import numpy as np

from nuslam.viz import log_estimate
from nuslam.viz import rerun_logging as rrlog


def _unit_quats(n, seed):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_logging_wrappers_run():
    rrlog.init("nuslam-test")  # in-memory recording, no sink
    rng = np.random.default_rng(0)

    rrlog.log_trajectory("traj/gt", rng.random((10, 3)), color=(150, 150, 150))
    rrlog.log_points("landmarks", rng.random((20, 3)), colors=(90, 160, 255))

    n = 12
    rrlog.log_gaussians(
        "gaussians",
        means=rng.random((n, 3)),
        scales=np.abs(rng.random((n, 3))) + 0.1,
        quats_wxyz=_unit_quats(n, 1),
        colors_rgba=(rng.random((n, 4)) * 255).astype(np.uint8),
    )


def test_log_estimate_runs():
    rrlog.init("nuslam-test")
    rng = np.random.default_rng(2)
    poses = np.tile(np.eye(4), (6, 1, 1))
    poses[:, :3, 3] = np.cumsum(rng.normal(size=(6, 3)), axis=0)
    landmarks = rng.normal(size=(15, 3))
    is_ground = rng.integers(0, 2, size=15).astype(bool)
    log_estimate(poses, poses, landmarks=landmarks, landmark_is_ground=is_ground, align="sim3")


def test_time_and_quat_helpers():
    rrlog.set_time(3, 1_000_500, 1_000_000)  # frame 3, +0.5 s
    assert rrlog._quat_xyzw([1.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 1.0]  # wxyz -> xyzw
