"""Output-visualization path: estimated trajectory & reconstruction vs. GT.

Two renderers, consuming plain arrays (estimated poses ``(T, 4, 4)`` and optional
landmark points ``(K, 3)``) so they serve any reconstruction output:

* :func:`plot_trajectory` -- static matplotlib figure (top-down + height + error),
  estimate overlaid on nuScenes ground truth after optional Sim(3)/SE(3)
  alignment. The figure to eyeball after a run.
* :func:`log_estimate` -- log the estimate (trajectory line, landmark cloud) to
  Rerun, aligned to GT so both sit in the same world frame in the 3D view.

Alignment borrows :mod:`nuslam.eval.metrics` so the picture and the reported ATE
agree. Visualization infrastructure only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..eval.metrics import align_trajectories


def _positions(poses: np.ndarray) -> np.ndarray:
    """(T, 4, 4) -> (T, 3) translation."""
    return np.asarray(poses)[:, :3, 3]


def plot_trajectory(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    *,
    landmarks: np.ndarray | None = None,
    align: str = "sim3",
    save_path: Path | str | None = None,
    show: bool = False,
    title: str | None = None,
):
    """Render estimate vs. GT. ``align`` in {"none", "se3", "sim3"}.

    ``sim3`` (default) also solves the scale factor -- the scalar that reveals
    whether metric scale resolved. It is printed in the panel title. Returns the
    matplotlib figure.
    """
    import matplotlib

    if save_path is not None and not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    est = _positions(est_poses)
    gt = _positions(gt_poses)[: len(est)]
    aligned, s, T = align_trajectories(est, gt, mode=align)
    err = np.linalg.norm(aligned - gt, axis=1)
    ate = float(np.sqrt(np.mean(err**2)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 1], "-", color="0.4", lw=2, label="ground truth")
    ax.plot(aligned[:, 0], aligned[:, 1], "-", color="tab:orange", lw=1.6, label="estimate")
    ax.scatter(gt[0, 0], gt[0, 1], c="green", s=40, zorder=5, label="start")
    if landmarks is not None:
        lm = np.asarray(landmarks)
        lm_h = lm @ T[:3, :3].T + T[:3, 3]
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

    fig.suptitle(title or "monocular reconstruction: estimate vs. nuScenes GT", fontsize=12)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def log_estimate(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    *,
    landmarks: np.ndarray | None = None,
    landmark_is_ground: np.ndarray | None = None,
    align: str = "sim3",
) -> None:
    """Log estimate + GT trajectory (and landmarks) to Rerun, aligned to GT.

    GT in grey, estimate in orange; landmarks as a blue cloud (ground-flagged
    points cyan when the flag is given). Requires an active Rerun recording
    (:func:`nuslam.viz.rerun_logging.init`).
    """
    from . import rerun_logging as rrlog

    est = _positions(est_poses)
    gt = _positions(gt_poses)[: len(est)]
    aligned, _, T = align_trajectories(est, gt, mode=align)

    rrlog.log_trajectory("traj/gt", gt, color=(150, 150, 150))
    rrlog.log_trajectory("traj/est", aligned, color=(255, 140, 25))

    if landmarks is not None and len(landmarks):
        lm = np.asarray(landmarks, np.float64)
        lm_h = lm @ T[:3, :3].T + T[:3, 3]
        if landmark_is_ground is not None:
            g = np.asarray(landmark_is_ground, bool)
            if g.any():
                rrlog.log_points("landmarks/ground", lm_h[g], colors=(80, 200, 255))
            if (~g).any():
                rrlog.log_points("landmarks/other", lm_h[~g], colors=(255, 180, 60))
        else:
            rrlog.log_points("landmarks", lm_h, colors=(90, 160, 255))
