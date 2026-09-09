"""Generic point-cloud utilities (framework-neutral, numpy in/out).

Data plumbing shared by the reconstruction seeding and viz. ``voxel_downsample``
thins a dense back-projected cloud to at most one point per occupied voxel, so the
Gaussian count (and the GPU) stays bounded before the cloud is seeded or logged. It is
a generic reduction and makes no decision about how Gaussians are seeded from the cloud;
it only offers a metric voxel knob, wrapping Open3D's ``voxel_down_sample``.
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    *,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None] | np.ndarray:
    """Keep one (centroid) point per occupied voxel of side ``voxel_size`` (scene units).

    Args:
        points: (N, 3) cloud.
        colors: optional (N, C) per-point attributes (e.g. RGB), averaged per voxel
            alongside the points; None to downsample positions only.
        voxel_size: voxel side length in the cloud's units (metres, once the cloud is
            at metric scale). Larger -> fewer points.

    Returns:
        ``points_ds (M, 3)`` if ``colors`` is None, else ``(points_ds, colors_ds)``,
        with ``M`` the number of occupied voxels.

    Implementation note: wrap ``open3d.geometry.PointCloud.voxel_down_sample`` --
    build a PointCloud from ``points`` (and ``colors`` as float [0, 1]), call
    ``voxel_down_sample(voxel_size)``, and read ``.points`` / ``.colors`` back to numpy.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64)/255.0)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(downsampled_pcd.points) if colors is None else (np.asarray(downsampled_pcd.points), np.asarray((np.asarray(downsampled_pcd.colors)*255.0).round(), dtype=np.uint8))


def nn_distances(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance from each point to its ``k`` nearest neighbours (scene units).

    Used to size Gaussians to the local point density at initialization: a scale of
    roughly the neighbour spacing keeps each Gaussian about as big as its neighbourhood
    (instead of an arbitrary fixed size). Returns an ``(N,)`` float array. Degenerate
    inputs (fewer than 2 points) return zeros. This computes a geometric quantity; the
    decision to seed scales from it lives in the reconstruction init.
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return np.zeros(n, dtype=np.float64)
    kk = min(k, n - 1)
    d, _ = cKDTree(pts).query(pts, k=kk + 1)  # +1 for the point itself (distance 0)
    return np.atleast_2d(d)[:, 1:].mean(axis=1)
