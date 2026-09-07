"""Metric upgrade of a DA3 reconstruction -- CORE SCALE-RESOLUTION GEOMETRY.

    ############################################################################
    #  The metric upgrade IS the scale-resolution contribution. The DAQ         #
    #  assembly, the linear solve, the null-vector extraction, and the          #
    #  homography that rectifies the DA3 (projective) reconstruction to metric  #
    #  are derived and written here, not generated. The bodies below are        #
    #  deliberately left unimplemented (NotImplementedError) as a seam for the  #
    #  surrounding plumbing and the validation harness to call.                 #
    ############################################################################

Setting.
  DA3-Base returns, per frame, a rotation/translation ``[R_i | t_i]`` and its own
  intrinsics ``K_da3`` -- an internally consistent reconstruction, but under a
  wrong (intrinsic-agnostic) ``K_da3``. The true camera is ``K_true`` (from
  nuScenes calibration). Interpreting the DA3 cameras with ``K_true`` yields a
  reconstruction that is only *projective* with respect to the true metric scene,
  related to it by a single 4x4 homography ``H``.

The geometry to derive and defend (the reason each function exists).
  * Normalize each DA3 camera by the true intrinsics:
        P~_i = K_true^{-1} K_da3 [R_i | t_i] = M [R_i | t_i],   M = K_true^{-1} K_da3.
    ``M`` is the residual (known) intrinsic mismatch. The metric target camera
    ``[R_i* | t_i*]`` is calibrated (DIAC = I), and  P~_i = [R_i*|t_i*] H  with
        H = [[ M,  0 ],
             [ v^T, s ]].
  * Dual absolute quadric in the projective frame:
        Omega*_proj = H^{-1} diag(1,1,1,0) H^{-T}.
    It satisfies, for every camera and up to a per-camera scale,
        P~_i Omega*_proj P~_i^T = lambda_i^2 I.
    The mismatch ``M`` is absorbed into Omega*_proj (its conic part / H's top-left
    block); the plane at infinity ``pi_inf = (v, s)`` is its null space (H's
    bottom row).
  * Eliminate the nuisance scale lambda_i^2 by encoding "proportional to I"
    (off-diagonals zero + diagonals equal) rather than "equals lambda_i^2 I".
    Each such constraint is linear and homogeneous in Omega*_proj, so it stacks
    into a DLT system solved by the smallest right singular vector.
  * Recover pi_inf as the null vector of Omega*_proj (up to scale/sign), read off
    ``(v, s)``, and assemble ``H`` from the known ``M``. The full
    ``H^{-1} diag(1,1,1,0) H^{-T}`` never has to be inverted or decomposed: the
    conic part of the fitted Omega*_proj should merely *agree* with the known
    ``M`` (a free consistency check on the solve).
  * ``H`` upgrades the reconstruction: a projective world point maps to metric by
    ``X_metric = H X_proj``, and a metric camera is recovered by
    ``P_metric = P~_proj H^{-1}`` (up to per-camera scale). The remaining overall
    scale (the shared length of ``pi_inf``) is the monocular metric scalar,
    resolved separately by the ground / wheel anchor.

Caveat to keep in view.
  A single ``Omega*_proj`` (hence a single ``H``) exists only if a single ``M``
  relates DA3 to the truth. If ``K_da3`` drifts per frame, ``M_i`` varies, no
  single quadric fits, and the DLT's "proportional to I" residual -- together
  with the null-space eigenvalue gap -- will not collapse. Those two numbers are
  the live test of whether the single-homography premise holds on real output.
"""
from __future__ import annotations

import numpy as np


def normalized_projective_camera(
    K_true: np.ndarray, K_da3: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """The DA3 camera in true-normalized coordinates: ``P~ = K_true^{-1} K_da3 [R|t]``.

    Args:
        K_true: (3, 3) true intrinsics (nuScenes calibration).
        K_da3:  (3, 3) DA3-estimated intrinsics for the same frame.
        R:      (3, 3) DA3 rotation.
        t:      (3,)   DA3 translation.

    Returns:
        (3, 4) normalized projective camera ``P~_i = M [R|t]`` with
        ``M = K_true^{-1} K_da3``.
    """
    raise NotImplementedError("core: normalized projective camera assembly (author-written)")


def build_daq_system(cameras: np.ndarray) -> np.ndarray:
    """Assemble the DLT matrix ``A`` for ``P~_i Omega*_proj P~_i^T (prop.) I``.

    Encode "proportional to I" (three off-diagonals zero + two equal-diagonal
    constraints) as outer-product rows in ``vec(Omega*_proj)``; the nuisance scale
    ``lambda_i^2`` drops out.

    Args:
        cameras: (N, 3, 4) normalized projective cameras ``P~_i``.

    Returns:
        (5 N, 16) matrix ``A`` with ``A vec(Omega*_proj) = 0``.
    """
    raise NotImplementedError("core: DAQ constraint assembly (author-written)")


def solve_daq(A: np.ndarray) -> np.ndarray:
    """Solve ``A vec(Omega*) = 0`` for the projective dual absolute quadric.

    Smallest right singular vector, reshape to 4x4, symmetrize, enforce rank-3
    (zero the smallest eigenvalue). Defined up to scale.

    Args:
        A: (5 N, 16) DLT matrix from :func:`build_daq_system`.

    Returns:
        (4, 4) symmetric rank-3 ``Omega*_proj`` (up to scale).
    """
    raise NotImplementedError("core: DAQ solve / rank-3 projection (author-written)")


def plane_at_infinity(omega_star: np.ndarray) -> np.ndarray:
    """The plane at infinity ``pi_inf = (v, s)`` as the null vector of ``Omega*_proj``.

    Up to scale and sign; fix the sign by a convention (e.g. ``s > 0``) before use.

    Args:
        omega_star: (4, 4) ``Omega*_proj`` from :func:`solve_daq`.

    Returns:
        (4,) ``pi_inf = (v_x, v_y, v_z, s)`` (up to scale).
    """
    raise NotImplementedError("core: plane-at-infinity / null-vector extraction (author-written)")


def rectifying_homography(pi_inf: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Assemble the rectifying homography ``H = [[M, 0], [v^T, s]]``.

    ``M`` is known (``K_true^{-1} K_da3``); ``(v, s)`` come from
    :func:`plane_at_infinity`. ``H`` upgrades projective points to metric
    (``X_metric = H X_proj``).

    Args:
        pi_inf: (4,) ``(v, s)`` from :func:`plane_at_infinity`.
        M:      (3, 3) known intrinsic mismatch ``K_true^{-1} K_da3``.

    Returns:
        (4, 4) rectifying homography ``H``.
    """
    raise NotImplementedError("core: rectifying-homography assembly (author-written)")


def metric_upgrade(cameras: np.ndarray, M: np.ndarray) -> np.ndarray:
    """End-to-end: normalized projective cameras + known ``M`` -> rectifier ``H``.

    Ties :func:`build_daq_system` -> :func:`solve_daq` -> :func:`plane_at_infinity`
    -> :func:`rectifying_homography`.

    Args:
        cameras: (N, 3, 4) normalized projective cameras ``P~_i``.
        M:       (3, 3) known intrinsic mismatch ``K_true^{-1} K_da3``.

    Returns:
        (4, 4) rectifying homography ``H`` (projective -> metric).
    """
    raise NotImplementedError("core: metric-upgrade pipeline (author-written)")


# --------------------------------------------------------------------------- #
#  Consuming the upgrade: recover metric cameras and rectify per-frame depth.  #
#  These apply H to the reconstruction (the pose decomposition is a library    #
#  call, like the gsplat calls -- author-written), so they are seams too.      #
# --------------------------------------------------------------------------- #

def metric_cameras(cameras: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Push the projective cameras through the rectifier: ``P_metric = P~ H^{-1}``.

    Each result is a normalized metric camera ``[R_i* | t_i*]`` up to a per-camera
    scale (its left 3x3 becomes a scaled rotation once ``H`` is correct).

    Args:
        cameras: (N, 3, 4) normalized projective cameras ``P~_i``.
        H:       (4, 4) rectifier from :func:`metric_upgrade`.

    Returns:
        (N, 3, 4) metric cameras ``P_metric,i`` (up to per-camera scale).
    """
    raise NotImplementedError("core: apply rectifier to cameras (author-written)")


def decompose_metric_camera(P_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(K, R*, t*)`` from a metric camera (``cv2.decomposeProjectionMatrix``).

    ``cv2.decomposeProjectionMatrix`` factors a 3x4 ``P`` into ``K [R | t]`` with
    ``K`` upper-triangular and ``R`` a rotation; it returns the camera centre in
    homogeneous form, so ``t*`` follows as ``t* = -R* C``. For a correctly
    rectified normalized camera ``K`` comes out proportional to the identity (the
    intrinsics were divided out) -- checking that is a free correctness test on
    ``H``. Fix ``K``'s sign convention (positive diagonal) as OpenCV may return a
    negated factor.

    Args:
        P_metric: (3, 4) metric camera from :func:`metric_cameras`.

    Returns:
        ``(K (3,3), R_star (3,3), t_star (3,))``.
    """
    raise NotImplementedError("core: metric-camera decomposition (author-written)")


def metric_depth(
    depth_da3: np.ndarray, K_da3: np.ndarray, extrinsic_da3: np.ndarray,
    H: np.ndarray, extrinsic_metric: np.ndarray,
) -> np.ndarray:
    """Rectify a DA3 depth map to metric depth in the metric camera frame.

    DA3 depth lives in DA3's camera frame under ``K_da3``; ``H`` is projective, so
    depth does NOT rescale by a constant -- each 3D point must be pushed through
    ``H`` and its metric-camera-frame ``z`` re-read. The chain, per pixel
    ``x = (u, v, 1)`` with DA3 depth ``d``:

        X_cam_da3   = d * K_da3^{-1} x                     # DA3 camera-frame point
        X_world_p   = inv(extrinsic_da3) @ [X_cam_da3; 1]  # DA3 world (projective) frame
        X_world_m   = H @ X_world_p ; dehomogenize         # metric world frame
        X_cam_m     = extrinsic_metric @ [X_world_m; 1]    # metric camera frame
        d_metric    = X_cam_m.z

    The nonlinearity is the per-point homogeneous denominator ``v^T X + s`` inside
    the dehomogenize step -- that is where the affine/metric correction acts.

    Args:
        depth_da3:        (H, W) DA3 depth in the DA3 camera frame.
        K_da3:            (3, 3) DA3 estimated intrinsics (full-res).
        extrinsic_da3:    (4, 4) DA3 world->camera pose for this frame.
        H:                (4, 4) rectifier from :func:`metric_upgrade`.
        extrinsic_metric: (4, 4) metric world->camera pose (from
                          :func:`decompose_metric_camera`, homogenized).

    Returns:
        (H, W) metric depth, paired with the true intrinsics ``K_true``.
    """
    raise NotImplementedError("core: depth rectification through H (author-written)")


def metric_point_cloud(
    depth_metric: np.ndarray, image: np.ndarray, K_true: np.ndarray,
    world_from_cam: np.ndarray, *, stride: int = 8,
    conf: np.ndarray | None = None, conf_thresh: float = 0.0,
    sky: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project metric depth to a world-frame RGB cloud (the visualized map).

    Same back-projection as :func:`nuslam.recon.unproject_depth_to_world`, but on
    the *metric* path: the true intrinsics ``K_true`` and the *recovered* metric
    pose ``world_from_cam`` (camera-to-world, from
    :func:`decompose_metric_camera`), rather than ``K_da3`` / the GT pose. This is
    what the Rerun viz logs as the metric map and what Sim(3) eval scores against
    GT lidar.

    Args:
        depth_metric:   (H, W) metric depth from :func:`metric_depth`.
        image:          (H, W, 3) uint8 RGB for point colors.
        K_true:         (3, 3) true intrinsics.
        world_from_cam: (4, 4) recovered metric camera-to-world pose.
        stride:         pixel decimation for viz density.
        conf, conf_thresh, sky: optional gating, as in ``unproject_depth_to_world``.

    Returns:
        ``(points_world (N, 3) float, colors_rgb (N, 3) uint8)``.
    """
    raise NotImplementedError("core: metric back-projection (author-written)")
