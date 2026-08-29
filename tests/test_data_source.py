"""Integration tests against the real nuScenes mini split (skip if absent)."""
import numpy as np

from nuslam.data import NuScenesMonoSource
from nuslam.transforms import SE3


def test_stream_yields_ordered_keyframes(dataroot):
    source = NuScenesMonoSource(dataroot, "v1.0-mini", camera="CAM_FRONT")
    scenes = source.list_scenes()
    assert scenes
    keyframes = list(source.stream(scenes[0][1]))
    assert len(keyframes) >= 2
    # temporally ordered, 0-based contiguous frame indices
    ts = [kf.timestamp_us for kf in keyframes]
    assert ts == sorted(ts)
    assert [kf.frame_index for kf in keyframes] == list(range(len(keyframes)))


def test_calibration_is_sane(dataroot):
    source = NuScenesMonoSource(dataroot, "v1.0-mini")
    kf = next(source.stream(source.list_scenes()[0][1]))
    K = kf.calib.intrinsic
    assert K.shape == (3, 3)
    assert kf.calib.fx > 100 and kf.calib.fy > 100  # plausible focal length in px
    assert 0 < kf.calib.cx < kf.calib.width
    assert 0 < kf.calib.cy < kf.calib.height
    assert isinstance(kf.calib.sensor2ego, SE3)


def test_gt_trajectory_moves(dataroot):
    source = NuScenesMonoSource(dataroot, "v1.0-mini")
    kfs = source.load_scene(source.list_scenes()[0][1])
    traj = source.gt_trajectory(kfs)
    assert traj.shape == (len(kfs), 4, 4)
    dist = np.linalg.norm(traj[-1, :3, 3] - traj[0, :3, 3])
    assert dist > 1.0  # the ego vehicle drives at least a metre over a scene


def test_image_decodes_to_calib_size(dataroot):
    source = NuScenesMonoSource(dataroot, "v1.0-mini")
    kf = next(source.stream(source.list_scenes()[0][1]))
    img = kf.image()
    assert img.shape == (kf.calib.height, kf.calib.width, 3)
    assert img.dtype == np.uint8
