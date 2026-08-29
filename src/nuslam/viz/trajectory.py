"""Output-visualization path: estimated trajectory & reconstruction vs. GT.

Two renderers, both consuming a :class:`~nuslam.backend.SlamEstimate`:

* :func:`plot_trajectory` -- static matplotlib figure (top-down + height + error),
  estimate overlaid on nuScenes ground truth after optional Sim(3)/SE(3)
  alignment. This is the figure to eyeball after each run.
* :func:`publish_estimate` -- push the estimate (trajectory line, landmark cloud,
  camera pose) to a live :class:`FoxgloveBridge` so the reconstruction can be
  inspected in 3D next to the GT trajectory.

Alignment for plotting only borrows :mod:`nuslam.eval.metrics` so the picture and
the reported ATE agree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..backend import SlamEstimate
from ..eval.metrics import align_trajectories
from .foxglove_bridge import FoxgloveBridge


def _positions(poses: np.ndarray) -> np.ndarray:
    """(T, 4, 4) -> (T, 3) translation."""
    return np.asarray(poses)[:, :3, 3]


def plot_trajectory(
    estimate: SlamEstimate,
    gt_poses: np.ndarray,
    *,
    align: str = "sim3",
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Render estimate vs. GT. ``align`` in {"none", "se3", "sim3"}.

    ``sim3`` (default) also solves the scale factor -- the number that reveals
    whether metric scale resolved (build ladder rung 2). The estimated scale is
    printed in the panel title. Returns the matplotlib figure.
    """
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = _positions(estimate.poses_ego2global)
    gt = _positions(gt_poses)[: len(est)]

    aligned, s, T = align_trajectories(est, gt, mode=align)
    err = np.linalg.norm(aligned - gt, axis=1)
    ate = float(np.sqrt(np.mean(err**2)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 1], "-", color="0.4", lw=2, label="ground truth")
    ax.plot(aligned[:, 0], aligned[:, 1], "-", color="tab:orange", lw=1.6, label="estimate")
    ax.scatter(gt[0, 0], gt[0, 1], c="green", s=40, zorder=5, label="start")
    if estimate.landmarks is not None:
        lm = np.asarray(estimate.landmarks)
        lm_h = lm @ T[:3, :3].T + T[:3, 3]  # same alignment as the trajectory
        ax.scatter(lm_h[:, 0], lm_h[:, 1], s=1, c="tab:blue", alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"top-down  (align={align}, scale={s:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(gt[:, 2], "-", color="0.4", lw=2, label="GT z")
    ax.plot(aligned[:, 2], "-", color="tab:orange", lw=1.6, label="est z")
    ax.set_xlabel("keyframe"); ax.set_ylabel("z [m]")
    ax.set_title("height"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(err, "-", color="tab:red", lw=1.5)
    ax.set_xlabel("keyframe"); ax.set_ylabel("position error [m]")
    ax.set_title(f"ATE (RMSE) = {ate:.3f} m"); ax.grid(alpha=0.3)

    fig.suptitle(title or "monocular SLAM: estimate vs. nuScenes GT", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def publish_estimate(
    bridge: FoxgloveBridge,
    estimate: SlamEstimate,
    gt_poses: np.ndarray,
    timestamp_us: int,
    *,
    frame: str = "global",
    align: str = "sim3",
) -> None:
    """Publish estimate + GT trajectory (and landmarks) to a live bridge.

    The estimate is aligned to GT so both sit in the same ``global`` frame in the
    3D panel: GT in grey, estimate in orange, landmarks as a blue cloud.
    """
    est = _positions(estimate.poses_ego2global)
    gt = _positions(gt_poses)[: len(est)]
    aligned, _, T = align_trajectories(est, gt, mode=align)

    bridge.publish_line("/traj/gt", frame, gt, timestamp_us, color=(0.6, 0.6, 0.6, 1.0), entity_id="gt")
    bridge.publish_line("/traj/est", frame, aligned, timestamp_us, color=(1.0, 0.55, 0.1, 1.0), entity_id="est")

    if estimate.landmarks is not None and len(estimate.landmarks):
        lm = np.asarray(estimate.landmarks, dtype=np.float64)
        lm_h = lm @ T[:3, :3].T + T[:3, 3]  # same alignment as the trajectory
        if estimate.landmark_is_ground is not None:
            g = estimate.landmark_is_ground.astype(bool)
            if g.any():
                bridge.publish_pointcloud("/landmarks/ground", frame, lm_h[g], timestamp_us, rgb=(80, 200, 255))
            if (~g).any():
                bridge.publish_pointcloud("/landmarks/other", frame, lm_h[~g], timestamp_us, rgb=(255, 180, 60))
        else:
            bridge.publish_pointcloud("/landmarks", frame, lm_h, timestamp_us, rgb=(90, 160, 255))
