"""Trajectory evaluation against nuScenes ground truth (ATE / RPE).

Standard SLAM metrics, plus the Umeyama alignment they need.

On gauge: the monocular ambiguity is a single scalar (the unknown metric scale,
1 DOF). The full 7-DOF Sim(3) freedom of a reconstruction is that scalar plus the
6-DOF SE(3) reference-frame gauge, and the SE(3) part is present in every SLAM
system, metric ones included. Sim(3) enters here only as the *evaluation*
alignment: comparing an estimate to GT must factor out the SE(3) frame offset
always, and the scale scalar too when it is unknown. The recovered scalar is then
the diagnostic -- on a scale-free run it is an arbitrary gauge; on a run that
claims metric scale (build ladder rung 2+), a scalar far from 1.0 means scale did
not lock, and an SE(3) alignment (scale fixed = 1) should already fit well.

Plumbing: this scores the estimate; it makes no estimation decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The Umeyama primitive lives in transforms.py (framework-neutral geometry) so the
# metric-scale resolver can reuse it without the core depending on this scoring
# package; re-exported here to keep ``nuslam.eval.umeyama`` stable.
from ..transforms import umeyama


def align_trajectories(
    est_positions: np.ndarray, gt_positions: np.ndarray, *, mode: str = "sim3"
) -> tuple[np.ndarray, float, np.ndarray]:
    """Align ``est`` positions to ``gt``. ``mode`` in {"none", "se3", "sim3"}.

    Returns ``(aligned_positions, scale, T)`` where ``T`` is the 4x4 matrix such
    that ``aligned = est @ T[:3,:3].T + T[:3,3]`` (its rotation block folds in the
    scale for sim3, so the same ``T`` also maps landmarks consistently).
    """
    est = np.asarray(est_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)
    if mode == "none":
        return est.copy(), 1.0, np.eye(4)
    scale, R, t = umeyama(est, gt, with_scale=(mode == "sim3"))
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    aligned = est @ T[:3, :3].T + T[:3, 3]
    return aligned, scale, T


@dataclass
class TrajErrors:
    ate_rmse: float          # absolute trajectory error, RMSE of position [m]
    ate_median: float
    rpe_trans_rmse: float    # relative pose error, translation [m] over `rpe_delta`
    rpe_rot_rmse_deg: float  # relative pose error, rotation [deg] over `rpe_delta`
    scale: float             # sim3 scale (1.0 if aligned se3/none)
    num_frames: int

    def __str__(self) -> str:  # pragma: no cover - reporting
        return (
            f"ATE(rmse)={self.ate_rmse:.3f} m  ATE(med)={self.ate_median:.3f} m  "
            f"RPE_t={self.rpe_trans_rmse:.3f} m  RPE_r={self.rpe_rot_rmse_deg:.3f} deg  "
            f"scale={self.scale:.4f}  T={self.num_frames}"
        )


def _rotation_angle_deg(R: np.ndarray) -> float:
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def evaluate(
    est_poses: np.ndarray, gt_poses: np.ndarray, *, align: str = "sim3", rpe_delta: int = 1
) -> TrajErrors:
    """Compute ATE and RPE for (T, 4, 4) estimated vs. GT ego->global poses.

    ATE aligns positions (Sim(3) by default -> reports scale) and takes the
    position residual. RPE compares relative motions ``T_i^{-1} T_{i+delta}``
    between estimate and GT, which is alignment-invariant.
    """
    est = np.asarray(est_poses, dtype=np.float64)
    gt = np.asarray(gt_poses, dtype=np.float64)[: len(est)]
    est_p, gt_p = est[:, :3, 3], gt[:, :3, 3]

    aligned, scale, _ = align_trajectories(est_p, gt_p, mode=align)
    ate = np.linalg.norm(aligned - gt_p, axis=1)

    trans_err, rot_err = [], []
    for i in range(len(est) - rpe_delta):
        rel_est = np.linalg.inv(est[i]) @ est[i + rpe_delta]
        rel_gt = np.linalg.inv(gt[i]) @ gt[i + rpe_delta]
        delta = np.linalg.inv(rel_gt) @ rel_est
        trans_err.append(np.linalg.norm(delta[:3, 3]))
        rot_err.append(_rotation_angle_deg(delta[:3, :3]))
    trans_err = np.asarray(trans_err) if trans_err else np.zeros(1)
    rot_err = np.asarray(rot_err) if rot_err else np.zeros(1)

    return TrajErrors(
        ate_rmse=float(np.sqrt(np.mean(ate**2))),
        ate_median=float(np.median(ate)),
        rpe_trans_rmse=float(np.sqrt(np.mean(trans_err**2))),
        rpe_rot_rmse_deg=float(np.sqrt(np.mean(rot_err**2))),
        scale=float(scale),
        num_frames=len(est),
    )
