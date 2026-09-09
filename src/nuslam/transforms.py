"""Rigid-body (SE(3)) transform helpers for nuScenes frames.

Plumbing only: a thin, framework-neutral wrapper over 4x4 homogeneous matrices so
the data layer can compose sensor->ego->global chains without every call site
re-deriving the matrix algebra. numpy in, numpy out. Quaternions are nuScenes'
convention: unit ``wxyz``.

This is geometry utility, not estimator design -- the factor graph (see
``nuslam.backend``) is free to use gtsam's own Pose3 type instead; nothing here
prescribes how state is represented in the graph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyquaternion


@dataclass(frozen=True)
class SE3:
    """A rigid transform ``p_parent = R @ p_child + t``.

    Stored as rotation ``R`` (3x3) and translation ``t`` (3,). Construct from a
    nuScenes ``(translation, rotation_wxyz)`` pair, a 4x4 matrix, or compose.
    """

    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    # ---- constructors ----------------------------------------------------

    @staticmethod
    def identity() -> "SE3":
        return SE3(np.eye(3), np.zeros(3))

    @staticmethod
    def from_translation_quaternion(translation, rotation_wxyz) -> "SE3":
        """nuScenes stores calibrated_sensor / ego_pose as (translation, wxyz)."""
        R = pyquaternion.Quaternion(np.asarray(rotation_wxyz, dtype=np.float64)).rotation_matrix
        t = np.asarray(translation, dtype=np.float64).reshape(3)
        return SE3(R, t)

    @staticmethod
    def from_matrix(M: np.ndarray) -> "SE3":
        M = np.asarray(M, dtype=np.float64)
        return SE3(M[:3, :3].copy(), M[:3, 3].copy())

    # ---- conversions -----------------------------------------------------

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    def quaternion_wxyz(self) -> np.ndarray:
        return pyquaternion.Quaternion(matrix=self.R).elements  # wxyz

    # ---- algebra ---------------------------------------------------------

    def __matmul__(self, other: "SE3") -> "SE3":
        """Compose: ``(self @ other)`` maps other's child frame into self's parent."""
        return SE3(self.R @ other.R, self.R @ other.t + self.t)

    def inverse(self) -> "SE3":
        Rt = self.R.T
        return SE3(Rt, -Rt @ self.t)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform an (N, 3) array (or a single (3,) point)."""
        p = np.asarray(points, dtype=np.float64)
        return p @ self.R.T + self.t

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        rpy = pyquaternion.Quaternion(matrix=self.R).yaw_pitch_roll
        return f"SE3(t={np.round(self.t, 3).tolist()}, ypr={np.round(rpy, 3).tolist()})"


def umeyama(src: np.ndarray, dst: np.ndarray, *, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity mapping ``src`` onto ``dst`` (Umeyama 1991).

    ``src``, ``dst`` are (N, 3). Returns ``(scale, R, t)`` minimizing
    ``sum ||dst_i - (scale * R @ src_i + t)||^2``. With ``with_scale=False`` the scale
    is fixed to 1 (pure SE(3) / rigid).

    A framework-neutral geometry primitive: the evaluation harness uses it for Sim(3)
    trajectory alignment, and the metric-scale resolver reuses it as the inner
    rigid/similarity fit inside its lever-arm iteration. It lives here (not in eval) so
    the core resolver need not depend on the scoring package.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1  # reflection fix
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc**2).sum() / n
        scale = float((D * np.diag(S)).sum() / var_s) if var_s > 0 else 1.0
    else:
        scale = 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t
