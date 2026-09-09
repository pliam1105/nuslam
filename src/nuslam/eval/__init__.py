"""Evaluation harness: ATE/RPE (Sim(3) alignment), depth-vs-lidar, rendered-view scoring."""
from .daq import DAQDiagnostics, evaluate_daq
from .depth import DepthErrors, evaluate_depth, sample_depth_at
from .metrics import TrajErrors, align_trajectories, evaluate, umeyama
from .render import RenderErrors, evaluate_render, holdout_indices
from .rung_log import format_rungs, load_rungs, record_rung

__all__ = [
    "evaluate", "TrajErrors", "align_trajectories", "umeyama",
    "evaluate_depth", "DepthErrors", "sample_depth_at",
    "evaluate_daq", "DAQDiagnostics",
    "evaluate_render", "RenderErrors", "holdout_indices",
    "record_rung", "load_rungs", "format_rungs",
]
