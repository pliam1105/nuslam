import numpy as np
import pyquaternion

from nuslam.eval.metrics import align_trajectories, evaluate, umeyama


def _poses_from_positions(pos: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, 3] = pos
    return T


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(50, 3))
    R = pyquaternion.Quaternion.random().rotation_matrix
    s, t = 2.5, np.array([3.0, -1.0, 0.5])
    dst = (s * (R @ src.T).T) + t
    s_hat, R_hat, t_hat = umeyama(src, dst, with_scale=True)
    assert np.isclose(s_hat, s, atol=1e-6)
    assert np.allclose(R_hat, R, atol=1e-6)
    assert np.allclose(t_hat, t, atol=1e-6)


def test_sim3_alignment_maps_landmarks_consistently():
    rng = np.random.default_rng(1)
    est = rng.normal(size=(30, 3))
    gt = 1.7 * est + np.array([1.0, 2.0, 3.0])
    aligned, s, T = align_trajectories(est, gt, mode="sim3")
    assert np.isclose(s, 1.7, atol=1e-6)
    # same T applied by hand must equal `aligned`
    by_T = est @ T[:3, :3].T + T[:3, 3]
    assert np.allclose(by_T, aligned, atol=1e-9)
    assert np.allclose(aligned, gt, atol=1e-6)


def test_evaluate_perfect_estimate():
    rng = np.random.default_rng(2)
    gt = _poses_from_positions(np.cumsum(rng.normal(size=(20, 3)), axis=0))
    err = evaluate(gt, gt, align="sim3")
    assert err.ate_rmse < 1e-9
    assert np.isclose(err.scale, 1.0, atol=1e-6)
    assert err.rpe_trans_rmse < 1e-9


def test_evaluate_recovers_scale_of_scaled_estimate():
    rng = np.random.default_rng(3)
    gt_pos = np.cumsum(rng.normal(size=(20, 3)), axis=0)
    est = _poses_from_positions(gt_pos / 3.0)  # metric-scale-off estimate
    err = evaluate(est, _poses_from_positions(gt_pos), align="sim3")
    assert np.isclose(err.scale, 3.0, atol=1e-6)  # sim3 recovers the missing scale
    assert err.ate_rmse < 1e-6                     # ...and aligns to ~0 residual
