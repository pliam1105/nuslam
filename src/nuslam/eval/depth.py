"""Score a monocular depth map against sparse lidar depth.

For a RELATIVE depth map (DA3Mono), the informative number is the **recovered
scale**: the median ratio of lidar depth to predicted depth at the projected
lidar pixels. A metric reconstruction's depth should give a scale near the true
metres-per-unit; the standard error metrics (AbsRel, RMSE, delta) are reported
after optionally applying that median scale, so a relative map is not penalised
for its arbitrary global scale.

Plumbing: this scores depth; it makes no modelling decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DepthErrors:
    scale: float          # median(lidar / pred) at valid pixels -- the metric read-out
    abs_rel: float        # mean |pred*scale - lidar| / lidar
    rmse: float           # sqrt(mean (pred*scale - lidar)^2), metres
    delta1: float         # fraction within 1.25x (after scale)
    num_points: int

    def __str__(self) -> str:  # pragma: no cover
        return (f"depth: scale={self.scale:.3f}  AbsRel={self.abs_rel:.3f}  "
                f"RMSE={self.rmse:.3f} m  d<1.25={self.delta1:.3f}  n={self.num_points}")


def sample_depth_at(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-pixel depth lookup at (M, 2) pixel coords."""
    h, w = depth.shape
    x = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    return depth[y, x]


def evaluate_depth(
    depth: np.ndarray, lidar_uv: np.ndarray, lidar_depth: np.ndarray,
    *, align_scale: bool = True, min_depth: float = 0.5, max_depth: float = 80.0,
) -> DepthErrors:
    """Compare a depth map to projected lidar. ``align_scale`` applies the median
    scale before the error metrics (report it for a relative map)."""
    pred = sample_depth_at(np.asarray(depth, np.float64), np.asarray(lidar_uv))
    gt = np.asarray(lidar_depth, np.float64)
    valid = (pred > 1e-6) & (gt > min_depth) & (gt < max_depth)
    pred, gt = pred[valid], gt[valid]
    if pred.size == 0:
        return DepthErrors(scale=float("nan"), abs_rel=float("nan"), rmse=float("nan"),
                           delta1=float("nan"), num_points=0)

    scale = float(np.median(gt / pred))
    p = pred * scale if align_scale else pred
    abs_rel = float(np.mean(np.abs(p - gt) / gt))
    rmse = float(np.sqrt(np.mean((p - gt) ** 2)))
    ratio = np.maximum(p / gt, gt / p)
    delta1 = float(np.mean(ratio < 1.25))
    return DepthErrors(scale=scale, abs_rel=abs_rel, rmse=rmse, delta1=delta1, num_points=int(pred.size))
