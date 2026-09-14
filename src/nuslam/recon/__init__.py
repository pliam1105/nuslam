"""Reconstruction core -- written by hand (representation, init, optimization, loss).

This package holds the Gaussian-splat reconstruction: the representation and its
initialization, the gsplat calls, the pose refinement, the losses (photometric +
metric anchor), and the scale resolution. It is core substance, built here, not
generated. Only the depth back-projection seam is stubbed so the surrounding viz
and evaluation plumbing have something to call.
"""
from .depth_init import backproject_depth_to_world, unproject_depth_to_world
from .gaussians import train_gaussians
from .ground_anchor import (
    GroundAnchor,
    GroundAnchorInputs,
    apply_ground_anchor,
    fit_ground_anchor,
    ground_anchor_inputs,
    ground_anchor_residual,
)
from .scale import ScaleResult, resolve_scale_gps
from .metric_upgrade import (
    build_daq_system,
    daq_m_prior_rows,
    decompose_metric_camera,
    metric_cameras,
    metric_depth,
    metric_point_cloud,
    metric_upgrade,
    normalized_projective_cameras,
    plane_at_infinity,
    rectifying_homography,
    solve_daq,
)

__all__ = [
    "unproject_depth_to_world",
    "backproject_depth_to_world",
    "normalized_projective_cameras",
    "build_daq_system",
    "daq_m_prior_rows",
    "solve_daq",
    "plane_at_infinity",
    "rectifying_homography",
    "metric_upgrade",
    "metric_cameras",
    "decompose_metric_camera",
    "metric_depth",
    "metric_point_cloud",
    "resolve_scale_gps",
    "ScaleResult",
    "train_gaussians",
    "GroundAnchor",
    "GroundAnchorInputs",
    "ground_anchor_inputs",
    "fit_ground_anchor",
    "ground_anchor_residual",
    "apply_ground_anchor",
]
