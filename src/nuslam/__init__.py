"""nuslam -- monocular metric reconstruction (SLAM) on nuScenes with GTSAM.

Package layout:
  nuslam.data      nuScenes monocular keyframe source + CAN proprio streams (plumbing)
  nuslam.frontend  offline CoTracker tracking + CLIPSeg road segmentation (delegated)
  nuslam.backend   the factor graph -- core estimator substance; seam only here
  nuslam.viz       Rerun logging (images/frusta/points/splats) + trajectory figures (plumbing)
  nuslam.eval      ATE/RPE against nuScenes GT (plumbing)
  nuslam.pipeline  data -> frontend -> [graph] -> eval + viz orchestration
"""
from . import backend, data, eval, frontend, viz  # noqa: F401
from .transforms import SE3
from .types import (
    CameraCalib,
    GroundMask,
    Keyframe,
    ProprioStreams,
    TrackSet,
)

__all__ = [
    "SE3",
    "CameraCalib",
    "Keyframe",
    "TrackSet",
    "GroundMask",
    "ProprioStreams",
    "backend",
    "data",
    "frontend",
    "viz",
    "eval",
]
