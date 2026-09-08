"""Metric-upgrade validation harness -- infrastructure (§4).

Forward-simulates a synthetic projective reconstruction with a known rectifier
``H`` and checks that the author-written DAQ solve recovers it. The solver bodies
in ``nuslam.recon.metric_upgrade`` are a seam (NotImplementedError) until written,
so every round-trip test is marked ``xfail`` on NotImplementedError: the suite
stays green now and these activate automatically once the solve lands.

The harness only *generates inputs and checks outputs* -- it does not implement
any part of the solve (that is core, author-written).
"""
import numpy as np
import pytest

from nuslam.recon import (
    build_daq_system,
    metric_upgrade,
    normalized_projective_cameras,
    plane_at_infinity,
    rectifying_homography,
    solve_daq,
)

xfail_unimpl = pytest.mark.xfail(raises=NotImplementedError, strict=False,
                                 reason="core solve not yet written")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))          # fix QR sign ambiguity
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _random_intrinsics(rng: np.random.Generator) -> np.ndarray:
    f = rng.uniform(800, 1600)
    return np.array([[f, 0, rng.uniform(700, 900)],
                     [0, f, rng.uniform(400, 600)],
                     [0, 0, 1.0]])


def _synthetic_reconstruction(rng: np.random.Generator, n: int = 8):
    """Known H = [[M,0],[v^T,s]] and projective cameras P~_i = [R*_i|t*_i] H."""
    K_true = _random_intrinsics(rng)
    K_da3 = _random_intrinsics(rng)
    M = np.linalg.inv(K_true) @ K_da3
    v = rng.standard_normal(3) * 0.1
    s = 1.0
    H = np.eye(4)
    H[:3, :3] = M
    H[3, :3] = v
    H[3, 3] = s

    cameras = np.empty((n, 3, 4))
    for i in range(n):
        Rt = np.hstack([_random_rotation(rng), rng.standard_normal((3, 1))])  # [R*|t*]
        cameras[i] = Rt @ H
    return cameras, M, v, s, H


def _dir(x: np.ndarray) -> np.ndarray:
    """Unit vector with a fixed sign (last nonzero entry positive) for up-to-scale compare."""
    x = x / np.linalg.norm(x)
    for e in x[::-1]:
        if abs(e) > 1e-9:
            return x * np.sign(e)
    return x


# ------------------------------------------------------------------ unit seams

@xfail_unimpl
def test_normalized_projective_cameras_match_M_Rt():
    rng = np.random.default_rng(0)
    K_true = _random_intrinsics(rng)
    n = 4
    K_da3 = np.stack([_random_intrinsics(rng) for _ in range(n)])  # per-frame intrinsics
    R = np.stack([_random_rotation(rng) for _ in range(n)])
    t = rng.standard_normal((n, 3))
    P = normalized_projective_cameras(K_true, K_da3, R, t)
    assert P.shape == (n, 3, 4)
    for i in range(n):
        Mi = np.linalg.inv(K_true) @ K_da3[i]
        assert np.allclose(P[i], Mi @ np.hstack([R[i], t[i, :, None]]))


@xfail_unimpl
def test_daq_system_shape():
    rng = np.random.default_rng(1)
    cameras, *_ = _synthetic_reconstruction(rng, n=6)
    A = build_daq_system(cameras)
    assert A.shape == (5 * 6, 16)


# --------------------------------------------------------------- round trips

@xfail_unimpl
def test_solve_daq_recovers_quadric_nullspace():
    rng = np.random.default_rng(2)
    cameras, M, v, s, H = _synthetic_reconstruction(rng)
    omega = solve_daq(build_daq_system(cameras))
    # smallest singular value tiny => matrix is (numerically) rank 3
    sv = np.linalg.svd(omega, compute_uv=False)
    assert sv[-1] < 1e-6 * sv[0]
    pi = plane_at_infinity(omega)
    assert np.allclose(_dir(pi), _dir(np.concatenate([v, [s]])), atol=1e-5)


@xfail_unimpl
def test_metric_upgrade_recovers_H():
    rng = np.random.default_rng(3)
    cameras, M, v, s, H = _synthetic_reconstruction(rng)
    H_rec = metric_upgrade(cameras, M)
    assert np.allclose(H_rec[:3, :3], M, atol=1e-6)                 # M block is known/exact
    pi_true = np.concatenate([v, [s]])
    assert np.allclose(_dir(H_rec[3, :]), _dir(pi_true), atol=1e-5)  # (v,s) up to scale/sign


def test_metric_point_cloud_shares_backprojection():
    """metric_point_cloud delegates to the shared util -> identical output, sky=None ok."""
    from nuslam.recon import backproject_depth_to_world, metric_point_cloud

    rng = np.random.default_rng(7)
    H, W = 12, 20
    depth = rng.uniform(1, 10, (H, W)).astype(np.float32)
    image = rng.integers(0, 255, (H, W, 3), np.uint8)
    K = np.array([[50, 0, 10], [0, 50, 6], [0, 0, 1]], float)
    T = np.eye(4)
    T[:3, :3] = _random_rotation(rng)
    T[:3, 3] = rng.standard_normal(3)
    pts_a, col_a = metric_point_cloud(depth, image, K, T, stride=4)   # sky/conf None
    pts_b, col_b = backproject_depth_to_world(depth, image, K, T, stride=4)
    assert np.array_equal(pts_a, pts_b) and np.array_equal(col_a, col_b)
    assert pts_a.shape[1] == 3 and len(pts_a) == len(col_a) > 0


def test_decompose_metric_camera_recovers_KRt():
    """cv2 decomposition must return t as a 3-vector world->camera translation."""
    from nuslam.recon import decompose_metric_camera

    rng = np.random.default_rng(11)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    R = _random_rotation(rng)
    t = rng.standard_normal(3)
    P = K @ np.hstack([R, t[:, None]])
    Kd, Rd, td = decompose_metric_camera(P)
    assert td.shape == (3,)
    assert np.allclose(td, t, atol=1e-6)                 # not the homogeneous centre
    assert np.allclose(Rd, R, atol=1e-6)
    assert np.allclose(Kd / Kd[2, 2], K, atol=1e-4)


def test_metric_depth_identity_and_noop():
    """metric_depth on identity transforms returns the input depth (u/v + shapes correct)."""
    from nuslam.recon import metric_depth

    rng = np.random.default_rng(12)
    H, W = 9, 16                                          # H != W catches a u/v swap
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], float)
    depth = rng.uniform(1, 10, (H, W))
    I4 = np.eye(4)
    out = metric_depth(depth, K, I4, I4, I4)
    assert out.shape == (H, W)
    assert np.allclose(out, depth, atol=1e-6)
    # H = I is a no-op even with a non-trivial (shared) pose: depth unchanged.
    E = np.eye(4)
    E[:3, :3] = _random_rotation(rng)
    E[:3, 3] = rng.standard_normal(3)
    assert np.allclose(metric_depth(depth, K, E, I4, E), depth, atol=1e-6)


@xfail_unimpl
def test_upgrade_makes_cameras_metric():
    """After P_metric = P~_i H^{-1}, the left 3x3 is a scaled rotation (calibrated)."""
    rng = np.random.default_rng(4)
    cameras, M, v, s, H = _synthetic_reconstruction(rng)
    H_rec = metric_upgrade(cameras, M)
    Hinv = np.linalg.inv(H_rec)
    for P in cameras:
        A = (P @ Hinv)[:, :3]
        A = A / np.cbrt(abs(np.linalg.det(A)))     # strip per-camera scale
        assert np.allclose(A @ A.T, np.eye(3), atol=1e-4)
