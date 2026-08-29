import numpy as np
import pyquaternion

from nuslam.transforms import SE3


def _random_se3(seed: int) -> SE3:
    rng = np.random.default_rng(seed)
    q = pyquaternion.Quaternion.random()
    return SE3(q.rotation_matrix, rng.normal(size=3))


def test_inverse_round_trip():
    T = _random_se3(0)
    I = T @ T.inverse()
    assert np.allclose(I.R, np.eye(3), atol=1e-9)
    assert np.allclose(I.t, np.zeros(3), atol=1e-9)


def test_apply_matches_matrix():
    T = _random_se3(1)
    pts = np.random.default_rng(2).normal(size=(10, 3))
    got = T.apply(pts)
    homog = (T.matrix() @ np.c_[pts, np.ones(10)].T).T[:, :3]
    assert np.allclose(got, homog, atol=1e-9)


def test_compose_associative_with_matrices():
    A, B = _random_se3(3), _random_se3(4)
    assert np.allclose((A @ B).matrix(), A.matrix() @ B.matrix(), atol=1e-9)


def test_quaternion_round_trip():
    T = _random_se3(5)
    R2 = pyquaternion.Quaternion(T.quaternion_wxyz()).rotation_matrix
    assert np.allclose(T.R, R2, atol=1e-9)


def test_from_translation_quaternion():
    q = pyquaternion.Quaternion(axis=[0, 0, 1], angle=np.pi / 2)
    T = SE3.from_translation_quaternion([1, 2, 3], q.elements)
    # rotates +x into +y
    assert np.allclose(T.apply(np.array([1.0, 0, 0])), [1 + 0, 2 + 1, 3], atol=1e-9)
