"""KLT (OpenCV) tracker integration test on the real mini split (skip if absent).

Kept small (few frames, CPU) so it runs in the normal suite. CoTracker is not
tested here -- it needs GPU + a weight download; run it via scripts/run_frontend.py.
"""
import numpy as np

from nuslam.data import NuScenesMonoSource
from nuslam.frontend import KLTFrontend, SeedConfig, TrackConfig, make_tracker


def _keyframes(dataroot, n=6):
    src = NuScenesMonoSource(dataroot, "v1.0-mini", camera="CAM_FRONT")
    return src.load_scene(src.list_scenes()[0][1], max_frames=n)


def test_klt_tracks_scene(dataroot):
    kfs = _keyframes(dataroot)
    cfg = TrackConfig(backend="klt", cell_size_px=140, downscale_long_side=480,
                      seed=SeedConfig(method="shitomasi"))
    tracks = make_tracker(cfg).track_scene(kfs)
    assert tracks.num_tracks > 20
    assert tracks.num_frames == len(kfs)
    # seed frames within range; tracks invisible before their seed
    assert tracks.seed_frame is not None
    assert tracks.seed_frame.min() >= 0 and tracks.seed_frame.max() < len(kfs)
    for k in range(tracks.num_tracks):
        s = int(tracks.seed_frame[k])
        assert not tracks.visible[:s, k].any()
        assert tracks.visible[s, k]  # visible at its seed frame
    # points stay within the full-res image
    w, h = kfs[0].calib.width, kfs[0].calib.height
    vis_pts = tracks.points[tracks.visible]
    assert (vis_pts[:, 0] >= -1).all() and (vis_pts[:, 0] <= w + 1).all()
    assert (vis_pts[:, 1] >= -1).all() and (vis_pts[:, 1] <= h + 1).all()


def test_klt_replenish_beats_single_seed(dataroot):
    kfs = _keyframes(dataroot, n=6)
    common = dict(backend="klt", cell_size_px=140, downscale_long_side=480)
    single = make_tracker(TrackConfig(replenish=False, **common)).track_scene(kfs)
    repl = make_tracker(TrackConfig(replenish=True, **common)).track_scene(kfs)
    # replenishment keeps later-frame density up vs. decaying single seeding
    assert repl.visible[-1].sum() >= single.visible[-1].sum()
    assert repl.num_tracks >= single.num_tracks


def test_klt_ground_labels(dataroot):
    from nuslam.types import GroundMask

    kfs = _keyframes(dataroot, n=4)
    # a fake "road" mask: bottom third of the image is ground
    masks = {}
    for kf in kfs:
        m = np.zeros((kf.calib.height, kf.calib.width), bool)
        m[int(kf.calib.height * 0.66):] = True
        masks[kf.token] = GroundMask(kf.token, m)
    tracks = make_tracker(TrackConfig(backend="klt", downscale_long_side=480)).track_scene(kfs, masks=masks)
    assert tracks.is_ground is not None
    assert tracks.is_ground.shape == (tracks.num_tracks,)
