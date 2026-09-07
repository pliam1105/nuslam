"""Depth plumbing: cache round-trip, lidar projection (real data), depth scoring."""
import numpy as np

from nuslam.eval import evaluate_depth
from nuslam.frontend import cache
from nuslam.types import DepthMap


def test_depth_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    depths = [
        DepthMap("t0", rng.random((12, 20)).astype(np.float32) * 5,
                 is_metric=False, conf=rng.random((12, 20)).astype(np.float32),
                 sky=rng.integers(0, 2, (12, 20)).astype(bool)),
        DepthMap("t1", rng.random((12, 20)).astype(np.float32) * 5,
                 is_metric=False, conf=rng.random((12, 20)).astype(np.float32),
                 sky=np.zeros((12, 20), bool)),
    ]
    cache.save_depth(tmp_path, "scene-x", depths)
    back = cache.load_depth(tmp_path, "scene-x")
    assert set(back) == {"t0", "t1"}
    assert np.allclose(back["t0"].depth, depths[0].depth, atol=1e-2)  # float16
    assert np.allclose(back["t0"].conf, depths[0].conf, atol=1.0 / 255)
    assert np.array_equal(back["t0"].sky, depths[0].sky)
    assert back["t0"].is_metric is False


def test_depth_cache_no_conf_sky(tmp_path):
    depths = [DepthMap("t", np.ones((8, 8), np.float32))]
    cache.save_depth(tmp_path, "s", depths)
    back = cache.load_depth(tmp_path, "s")
    assert back["t"].conf is None and back["t"].sky is None
    assert back["t"].extrinsic is None and back["t"].intrinsic is None


def test_depth_cache_recon_pose_intrinsics(tmp_path):
    # DA3-Base carries a world->camera pose and estimated K alongside depth.
    rng = np.random.default_rng(3)
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    ext[:3, 3] = rng.standard_normal(3)
    K = np.array([[900, 0, 800], [0, 900, 450], [0, 0, 1]], np.float32)
    depths = [DepthMap("t0", rng.random((10, 16)).astype(np.float32),
                       conf=rng.random((10, 16)).astype(np.float32),
                       extrinsic=ext, intrinsic=K)]
    cache.save_depth(tmp_path, "recon", depths)
    back = cache.load_depth(tmp_path, "recon")["t0"]
    assert np.allclose(back.extrinsic, ext, atol=1e-5)
    assert np.allclose(back.intrinsic, K, atol=1e-3)


def test_missing_depth_cache_returns_none(tmp_path):
    assert cache.load_depth(tmp_path, "nope") is None


def test_evaluate_depth_recovers_scale():
    # relative depth off by a constant factor: scale must come out as that factor
    depth = np.full((100, 100), 2.0)          # predicted (relative)
    uv = np.array([[10, 10], [50, 50], [90, 20]], float)
    lidar_depth = np.array([6.0, 6.0, 6.0])   # metric GT -> ratio 3.0
    err = evaluate_depth(depth, uv, lidar_depth)
    assert np.isclose(err.scale, 3.0)
    assert err.abs_rel < 1e-6
    assert err.delta1 == 1.0
    assert err.num_points == 3


def test_evaluate_depth_no_valid_points():
    err = evaluate_depth(np.ones((10, 10)), np.empty((0, 2)), np.empty((0,)))
    assert err.num_points == 0 and np.isnan(err.scale)


def test_project_lidar_to_camera(dataroot):
    from nuslam.data import NuScenesMonoSource, project_lidar_to_camera

    source = NuScenesMonoSource(dataroot, "v1.0-mini", camera="CAM_FRONT")
    kf = next(source.stream(source.list_scenes()[0][1]))
    uv, depth = project_lidar_to_camera(source, kf)
    assert len(uv) == len(depth) and len(uv) > 100      # front cam sees plenty of lidar
    assert (uv[:, 0] >= 0).all() and (uv[:, 0] < kf.calib.width).all()
    assert (uv[:, 1] >= 0).all() and (uv[:, 1] < kf.calib.height).all()
    assert (depth > 0).all()


def test_lidar_points_global(dataroot):
    from nuslam.data import NuScenesMonoSource, lidar_points_global

    source = NuScenesMonoSource(dataroot, "v1.0-mini")
    kf = next(source.stream(source.list_scenes()[0][1]))
    xyz = lidar_points_global(source, kf)
    assert xyz.shape[1] == 3 and len(xyz) > 1000
    # points sit near the ego's global position (a lidar sweep spans ~100 m)
    ego = kf.ego2global_gt.t
    assert np.linalg.norm(xyz.mean(0) - ego) < 100.0
