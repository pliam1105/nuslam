"""Depth back-projection -- CORE INITIALIZATION.

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
    """Back-project a depth map to a world-frame RGB point cloud. Written by hand."""
    img = keyframe.image() # (H,W,3)
    vv, uu = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    img_coords = np.stack([uu, vv, np.ones_like(uu)], axis=-1) # (H,W,3)
    depth = depth_map.depth.reshape(img.shape[0], img.shape[1], 1, 1)
    img_coords = img_coords.reshape(img.shape[0], img.shape[1], 3, 1)
    pts_camera = (depth * np.linalg.inv(keyframe.calib.intrinsic).reshape(1,1,3,3) @ img_coords).reshape(-1,3,1) # (N,3,1)
    pts_camera_norm = np.concatenate([pts_camera, np.ones_like(pts_camera)[:,0:1,:]], axis=1) # (N,4,1)
    pts_world_norm = keyframe.ego2global_gt.matrix().reshape(1,4,4) @ keyframe.calib.sensor2ego.matrix().reshape(1,4,4) @ pts_camera_norm
    pts_world = pts_world_norm[:,:3,0]/pts_world_norm[:,3:,0] # (N,3)
    colors_rgb = img.reshape(-1,3)
    pts_mask = (((depth_map.conf is None) or (depth_map.conf > conf_thresh)) & (~depth_map.sky) & ((uu % stride) == 0) & ((vv % stride) == 0)).reshape(-1)
    return pts_world[pts_mask], colors_rgb[pts_mask]