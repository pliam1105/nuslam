"""Shi-Tomasi seeding + cell-picking tests (uses OpenCV, no tracker model)."""
import numpy as np

from nuslam.frontend import seeding
from nuslam.frontend.seeding import SeedConfig


def _checkerboard(h=80, w=120, sq=10):
    ys, xs = np.mgrid[0:h, 0:w]
    board = (((xs // sq) + (ys // sq)) % 2 * 255).astype(np.uint8)
    return board


def test_cell_of_and_dims():
    ncols, nrows = seeding.grid_dims(120, 80, 40)
    assert (ncols, nrows) == (3, 2)
    assert seeding.cell_of(0, 0, 40, ncols, nrows) == 0
    assert seeding.cell_of(119, 79, 40, ncols, nrows) == nrows * ncols - 1  # clamps to last cell


def test_shi_tomasi_finds_corners_on_texture():
    corners = seeding.shi_tomasi_corners(_checkerboard(), SeedConfig())
    assert len(corners) > 10          # checkerboard is corner-rich
    assert corners.shape[1] == 2


def test_shi_tomasi_empty_on_flat_image():
    flat = np.full((60, 60), 128, np.uint8)
    assert len(seeding.shi_tomasi_corners(flat, SeedConfig())) == 0


def test_point_in_cell_shitomasi_skips_empty_cell():
    ncols = 3
    corner_cells = {0: (5.0, 5.0)}  # only cell 0 has a corner
    got = seeding.point_in_cell(0, 40, ncols, 120, 80, "shitomasi", corner_cells)
    assert got == (5.0, 5.0)
    # cell 1 has no corner -> None (untrackable, skipped)
    assert seeding.point_in_cell(1, 40, ncols, 120, 80, "shitomasi", corner_cells) is None


def test_point_in_cell_grid_uses_center():
    got = seeding.point_in_cell(0, 40, 3, 120, 80, "grid", None)
    assert got == (20.0, 20.0)  # center of first 40px cell


def test_corners_by_cell_keeps_strongest_first():
    # two corners in the same cell; the first (strongest) must win
    corners = np.array([[3.0, 3.0], [8.0, 8.0]], np.float32)
    by_cell = seeding.corners_by_cell(corners, 40, 3, 2)
    assert by_cell[0] == (3.0, 3.0)
