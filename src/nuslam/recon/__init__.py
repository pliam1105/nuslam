"""Reconstruction core -- written by hand (representation, init, optimization, loss).

This package holds the Gaussian-splat reconstruction: the representation and its
initialization, the gsplat calls, the pose refinement, the losses (photometric +
metric anchor), and the scale resolution. It is core substance, built here, not
generated. Only the depth back-projection seam is stubbed so the surrounding viz
and evaluation plumbing have something to call.
"""
from .depth_init import backproject_depth_to_world, unproject_depth_to_world
from .metric_upgrade import (
    build_daq_system,
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
    "solve_daq",
    "plane_at_infinity",
    "rectifying_homography",
    "metric_upgrade",
    "metric_cameras",
    "decompose_metric_camera",
    "metric_depth",
    "metric_point_cloud",
]
