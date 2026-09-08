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
to seed Gaussian means/colors. The pixel->3D back-projection itself lives in
:func:`backproject_depth_to_world`, a util shared with the metric-upgrade path
(``nuslam.recon.metric_upgrade.metric_point_cloud``): same geometry, different
``(depth, K, pose)`` -- GT pose + true K here, recovered metric pose + true K there.

Sketch of what it does (the geometry to derive and defend):
  - pick pixels on a strided grid; drop ``sky`` and low ``conf``;
  - back-project ``X_cam = depth * K^{-1} [u, v, 1]^T``;
  - transform to the world frame with a 4x4 ``world_from_cam``
    (nuScenes camera is RDF: x right, y down, z forward);
  - colors from ``image[v, u]``.
Return ``(points_world (N, 3) float, colors_rgb (N, 3) uint8)``.
"""
from __future__ import annotations

import numpy as np

from ..types import DepthMap, Keyframe


def backproject_depth_to_world(
    depth: np.ndarray, image: np.ndarray, K: np.ndarray, world_from_cam: np.ndarray,
    *, stride: int = 8, conf: np.ndarray | None = None, conf_thresh: float = 0.0,
    sky: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a depth map to a world-frame RGB point cloud (written by hand).

    Shared by :func:`unproject_depth_to_world` (GT pose) and the metric-upgrade
    path (recovered metric pose). ``world_from_cam`` is a 4x4 camera->world; ``K``
    the intrinsics paired with ``depth``. ``conf``/``sky`` (either may be None,
    e.g. DA3-Base has no sky mask) gate which pixels survive, along with the
    strided grid.
    """
    H, W = image.shape[0], image.shape[1]
    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    img_coords = np.stack([uu, vv, np.ones_like(uu)], axis=-1).reshape(H, W, 3, 1) # (H,W,3,1)
    depth = depth.reshape(H, W, 1, 1)
    pts_camera = (depth * np.linalg.inv(K).reshape(1,1,3,3) @ img_coords).reshape(-1,3,1) # (N,3,1)
    pts_camera_norm = np.concatenate([pts_camera, np.ones_like(pts_camera)[:,0:1,:]], axis=1) # (N,4,1)
    pts_world_norm = world_from_cam.reshape(1,4,4) @ pts_camera_norm
    pts_world = pts_world_norm[:,:3,0]/pts_world_norm[:,3:,0] # (N,3)
    colors_rgb = image.reshape(-1,3)
    conf_ok = True if conf is None else (conf > conf_thresh)
    sky_ok = np.ones_like(uu, dtype=bool) if sky is None else (~sky)
    depth_ok = depth.reshape(H, W) > 0
    pts_mask = (conf_ok & sky_ok & depth_ok & ((uu % stride) == 0) & ((vv % stride) == 0)).reshape(-1)
    return pts_world[pts_mask], colors_rgb[pts_mask]


def unproject_depth_to_world(
    depth_map: DepthMap, keyframe: Keyframe, *, stride: int = 8,
    conf_thresh: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a keyframe's depth to a world cloud using its GT pose + true K."""
    world_from_cam = keyframe.ego2global_gt.matrix() @ keyframe.calib.sensor2ego.matrix()
    return backproject_depth_to_world(
        depth_map.depth, keyframe.image(), keyframe.calib.intrinsic, world_from_cam,
        stride=stride, conf=depth_map.conf, conf_thresh=conf_thresh, sky=depth_map.sky)