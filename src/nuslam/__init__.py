"""nuslam -- monocular metric reconstruction (SLAM) on nuScenes with GTSAM.

Package layout:
  nuslam.data      nuScenes monocular keyframe source + CAN proprio streams (plumbing)
  nuslam.frontend  offline CoTracker tracking + CLIPSeg road segmentation (delegated)
  nuslam.backend   the §14 Sim(3) factor graph (COLMAP + DA3) -- core estimator substance
  nuslam.recon     metric upgrade (DAQ), ground anchor, and 3D Gaussian-splat reconstruction
  nuslam.viz       Rerun logging (images/frusta/points/splats) + trajectory figures (plumbing)
  nuslam.eval      ATE/RPE against nuScenes GT (plumbing)
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
