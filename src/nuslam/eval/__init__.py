"""Evaluation harness: ATE/RPE (Sim(3) alignment) and depth-vs-lidar scoring."""
from .depth import DepthErrors, evaluate_depth, sample_depth_at
from .metrics import TrajErrors, align_trajectories, evaluate, umeyama

__all__ = [
    "evaluate", "TrajErrors", "align_trajectories", "umeyama",
    "evaluate_depth", "DepthErrors", "sample_depth_at",
]
