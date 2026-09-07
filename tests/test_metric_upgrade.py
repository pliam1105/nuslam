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
    normalized_projective_camera,
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
def test_normalized_projective_camera_matches_M_Rt():
    rng = np.random.default_rng(0)
    K_true, K_da3 = _random_intrinsics(rng), _random_intrinsics(rng)
    R, t = _random_rotation(rng), rng.standard_normal(3)
    P = normalized_projective_camera(K_true, K_da3, R, t)
    M = np.linalg.inv(K_true) @ K_da3
    assert np.allclose(P, M @ np.hstack([R, t[:, None]]))


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
