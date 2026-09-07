"""Depth back-projection -- CORE INITIALIZATION, written by hand.

    ############################################################################
    #  Seeding the reconstruction from monocular depth is core representation/  #
    #  initialization substance. The pixel -> 3D back-projection, the strided   #
    #  grid, the confidence/sky gating, the scale choice (kept deliberately     #
    #  wrong), and the transform into the world/submap frame are derived and    #
    #  written here, not generated.                                            #
    ############################################################################

The surrounding plumbing calls :func:`unproject_depth_to_world` to render the
depth as a point cloud next to lidar (``scripts/visualize_depth.py``) and, later,
to seed Gaussian means/colors. Implement it once; both consume it.

Sketch of what it does (the geometry to derive and defend):
  - pick pixels on a strided grid; drop ``depth_map.sky`` and low ``depth_map.conf``;
  - back-project ``X_cam = depth * K^{-1} [u, v, 1]^T`` with ``K = keyframe.calib.intrinsic``;
  - transform to the world frame with ``keyframe.ego2global_gt @ keyframe.calib.sensor2ego``
    (nuScenes camera is RDF: x right, y down, z forward);
  - colors from ``keyframe.image()[v, u]``.
Return ``(points_world (N, 3) float, colors_rgb (N, 3) uint8)``.
"""
from __future__ import annotations

import numpy as np

from ..types import DepthMap, Keyframe


def unproject_depth_to_world(
    depth_map: DepthMap, keyframe: Keyframe, *, stride: int = 8,
    conf_thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to a world-frame RGB point cloud. Written by hand.

    See module banner. Left unimplemented by design.
    """
    raise NotImplementedError(
        "unproject_depth_to_world is core initialization (the monocular-depth "
        "back-projection). It is derived and written here, not generated. Fill in "
        "the pixel->3D unprojection + world transform; the viz and init both call it."
    )
