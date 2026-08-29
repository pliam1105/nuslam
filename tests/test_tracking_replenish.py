"""Unit tests for the grid-replenishment cell logic (pure, no tracker model)."""
import numpy as np

from nuslam.frontend.seeding import empty_viable_cells


def test_empty_viable_cell_is_reseeded():
    # 2x2 grid of 10px cells over 20x20, T=3. cell(15,5)=ci 1, cell(5,5)=ci 0.
    cell, W, H, T = 10, 20, 20, 3
    pts = np.zeros((T, 2, 2), np.float32)
    vis = np.zeros((T, 2), bool)
    pts[:, 0] = (5, 5); vis[:, 0] = True                 # persistent in ci 0
    pts[:, 1] = (15, 5); vis[0, 1] = True; vis[2, 1] = True  # persistent ci 1, empty at t=1
    cells = empty_viable_cells(pts, vis, cell, W, H, set())
    assert (1, 1) in cells                                # ci 1 empty at t=1 -> reseed
    assert not any(ci == 0 for ci, _ in cells)            # ci 0 always occupied


def test_non_viable_cell_never_reseeded():
    # a track visible on a single frame (sky-like) makes no cell viable.
    cell, W, H, T = 10, 20, 20, 3
    pts = np.full((T, 1, 2), 15.0, np.float32)
    vis = np.zeros((T, 1), bool)
    vis[0, 0] = True
    assert empty_viable_cells(pts, vis, cell, W, H, set()) == []


def test_attempted_dedup_converges():
    cell, W, H, T = 10, 20, 20, 4
    pts = np.zeros((T, 2, 2), np.float32)
    vis = np.zeros((T, 2), bool)
    pts[:, 0] = (5, 5); vis[:, 0] = True
    pts[:, 1] = (15, 5); vis[[0, 3], 1] = True
    attempted = set()
    first = empty_viable_cells(pts, vis, cell, W, H, attempted)
    assert first
    assert empty_viable_cells(pts, vis, cell, W, H, attempted) == []  # same state, deduped
