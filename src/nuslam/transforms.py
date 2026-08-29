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
