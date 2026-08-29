import numpy as np

from nuslam.frontend import cache
from nuslam.types import GroundMask, TrackSet


def test_tracks_round_trip(tmp_path):
    tracks = TrackSet(
        frame_tokens=["a", "b", "c"],
        points=np.random.default_rng(0).normal(size=(3, 40, 2)).astype(np.float32),
        visible=np.random.default_rng(1).integers(0, 2, size=(3, 40)).astype(bool),
        query_frame=0,
        is_ground=np.random.default_rng(2).integers(0, 2, size=40).astype(bool),
        seed_frame=np.random.default_rng(3).integers(0, 3, size=40).astype(np.int64),
    )
    cache.save_tracks(tmp_path, "scene-x", tracks)
    back = cache.load_tracks(tmp_path, "scene-x")
    assert back.frame_tokens == tracks.frame_tokens
    assert np.allclose(back.points, tracks.points)
    assert np.array_equal(back.visible, tracks.visible)
    assert back.query_frame == 0
    assert np.array_equal(back.is_ground, tracks.is_ground)
    assert np.array_equal(back.seed_frame, tracks.seed_frame)


def test_tracks_round_trip_no_ground(tmp_path):
    tracks = TrackSet(["a", "b"], np.zeros((2, 5, 2), np.float32), np.ones((2, 5), bool), 0, None)
    cache.save_tracks(tmp_path, "s", tracks)
    back = cache.load_tracks(tmp_path, "s")
    assert back.is_ground is None
    assert back.seed_frame is None


def test_masks_round_trip(tmp_path):
    masks = [
        GroundMask("t0", np.random.default_rng(0).integers(0, 2, (12, 20)).astype(bool),
                   np.random.default_rng(1).random((12, 20)).astype(np.float32)),
        GroundMask("t1", np.zeros((12, 20), bool), np.zeros((12, 20), np.float32)),
    ]
    cache.save_masks(tmp_path, "scene-y", masks)
    back = cache.load_masks(tmp_path, "scene-y")
    assert set(back) == {"t0", "t1"}
    assert np.array_equal(back["t0"].mask, masks[0].mask)
    assert np.allclose(back["t0"].prob, masks[0].prob, atol=1.0 / 255)  # uint8-quantized


def test_missing_cache_returns_none(tmp_path):
    assert cache.load_tracks(tmp_path, "nope") is None
    assert cache.load_masks(tmp_path, "nope") is None


def test_ground_mask_sample_at():
    mask = np.zeros((10, 10), bool)
    mask[5, 3] = True
    gm = GroundMask("t", mask)
    pts = np.array([[3, 5], [0, 0], [3.4, 4.9]])  # (x, y); third rounds to (3,5)
    got = gm.sample_at(pts)
    assert list(got) == [True, False, True]
