"""Subpixel cornerSubPix refinement tests (OpenCV, no tracker model)."""
import numpy as np

from nuslam.frontend.refine import RefineConfig, refine_points_on_frame


def _corner_image(h=80, w=80, cx=40, cy=40):
    """Black image with a white lower-right quadrant -> a sharp corner at (cx, cy)."""
    img = np.zeros((h, w), np.uint8)
    img[cy:, cx:] = 255
    return img


def test_refine_snaps_toward_true_corner():
    img = _corner_image(cx=40, cy=40)
    off = np.array([[42.0, 38.5]], np.float32)  # ~2.5 px off the true corner
    refined, accepted, shift = refine_points_on_frame(img, off, RefineConfig(max_shift_px=4.0))
    assert accepted[0]
    # moved closer to the true corner (40, 40) than it started
    assert np.linalg.norm(refined[0] - [40, 40]) < np.linalg.norm(off[0] - [40, 40])
    assert shift[0] > 0


def test_large_move_rejected_keeps_original():
    img = _corner_image(cx=40, cy=40)
    off = np.array([[43.0, 37.0]], np.float32)  # ~4.2 px off
    refined, accepted, _ = refine_points_on_frame(img, off, RefineConfig(max_shift_px=1.0))
    assert not accepted[0]                      # would move > max_shift -> rejected
    assert np.allclose(refined[0], off[0])      # original kept


def test_border_points_untouched():
    img = _corner_image()
    pts = np.array([[1.0, 1.0], [79.0, 2.0]], np.float32)  # inside the border window
    refined, accepted, _ = refine_points_on_frame(img, pts, RefineConfig(win=7))
    assert not accepted.any()
    assert np.allclose(refined, pts)


def test_empty_input():
    refined, accepted, shift = refine_points_on_frame(
        _corner_image(), np.empty((0, 2), np.float32), RefineConfig())
    assert refined.shape == (0, 2) and accepted.shape == (0,) and shift.shape == (0,)


def test_refine_tracks_updates_only_points(dataroot):
    """End-to-end on the real KLT tracker: refinement preserves shape/visibility."""
    from dataclasses import replace

    from nuslam.data import NuScenesMonoSource
    from nuslam.frontend import TrackConfig, make_tracker
    from nuslam.frontend.refine import refine_tracks_subpixel

    src = NuScenesMonoSource(dataroot, "v1.0-mini", camera="CAM_FRONT")
    kfs = src.load_scene(src.list_scenes()[0][1], max_frames=4)
    # track without auto-refine so we can compare pre/post
    raw = make_tracker(TrackConfig(backend="klt", downscale_long_side=480,
                                   refine=replace(TrackConfig().refine, enabled=False))).track_scene(kfs)
    refined, stats = refine_tracks_subpixel(raw, kfs, RefineConfig())
    assert refined.points.shape == raw.points.shape
    assert np.array_equal(refined.visible, raw.visible)          # visibility unchanged
    assert np.array_equal(refined.seed_frame, raw.seed_frame)    # seed frames unchanged
    assert stats.n_points > 0
    # invisible entries must be left exactly as they were
    inv = ~raw.visible
    assert np.array_equal(refined.points[inv], raw.points[inv])
