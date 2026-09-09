"""Contract tests for the voxel downsampler (wraps open3d ``voxel_down_sample``).

Pin the numpy-in/numpy-out contract the callers (seeding, viz) rely on: voxel
collapse, bounded count, and uint8 colour round-trip.
"""
import numpy as np

from nuslam.pointcloud import voxel_downsample


def test_collapses_points_in_same_voxel():
    # Two tight clusters, mid-voxel and not an integer number of voxels apart, so each
    # lands cleanly in one voxel -> two occupied voxels.
    pts = np.concatenate([
        0.3 + np.random.default_rng(0).normal(0, 0.001, (50, 3)),
        3.7 + np.random.default_rng(1).normal(0, 0.001, (50, 3)),
    ])
    ds = voxel_downsample(pts, voxel_size=1.0)
    assert ds.shape == (2, 3)
    ds = ds[np.argsort(ds[:, 0])]
    np.testing.assert_allclose(ds[0], [0.3, 0.3, 0.3], atol=0.05)
    np.testing.assert_allclose(ds[1], [3.7, 3.7, 3.7], atol=0.05)


def test_reduces_count_and_bounds_it():
    pts = np.random.default_rng(2).uniform(0, 10, (10000, 3))
    ds = voxel_downsample(pts, voxel_size=1.0)
    assert len(ds) < len(pts)
    assert len(ds) <= 11 * 11 * 11  # at most ~one point per 1 m voxel in a 10 m cube


def test_colors_travel_with_points():
    pts = np.array([[0.3, 0.3, 0.3], [0.35, 0.35, 0.35], [3.7, 3.7, 3.7]])
    cols = np.array([[10, 10, 10], [30, 30, 30], [200, 200, 200]], dtype=np.uint8)
    ds, dc = voxel_downsample(pts, cols, voxel_size=1.0)
    assert ds.shape == (2, 3) and dc.shape == (2, 3)
