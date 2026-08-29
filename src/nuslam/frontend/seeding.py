"""Shared seeding + grid-cell logic for the tracking frontends.

Both trackers (CoTracker, KLT) seed points on a pixel grid and replenish empty
cells as tracks are lost. Where a point goes *inside* a target cell is chosen by
the Shi-Tomasi "good features to track" criterion (``cv2.goodFeaturesToTrack``,
min-eigenvalue of the local structure tensor) so seeds land on trackable texture
rather than arbitrary centres; a ``grid`` method (cell centre) is kept as a
baseline. Textureless cells (sky) yield no corner, so Shi-Tomasi seeding also
gates out regions that cannot be tracked.

All coordinates here are in the *processing* resolution the tracker runs at.
Pure/CPU helpers -- no torch, unit-testable without a model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SeedConfig:
    method: str = "shitomasi"   # "shitomasi" | "grid"
    quality: float = 0.01       # goodFeaturesToTrack qualityLevel
    min_distance: int = 7       # min spacing between corners (proc px)
    block_size: int = 7         # structure-tensor window


def grid_dims(proc_w: int, proc_h: int, cell: int) -> tuple[int, int]:
    return math.ceil(proc_w / cell), math.ceil(proc_h / cell)  # (ncols, nrows)


def cell_of(x: float, y: float, cell: int, ncols: int, nrows: int) -> int:
    c = min(int(x // cell), ncols - 1)
    r = min(int(y // cell), nrows - 1)
    return r * ncols + c


def cell_center(ci: int, cell: int, ncols: int, proc_w: int, proc_h: int) -> tuple[float, float]:
    r, c = divmod(ci, ncols)
    return (min((c + 0.5) * cell, proc_w - 1), min((r + 0.5) * cell, proc_h - 1))


def shi_tomasi_corners(gray: np.ndarray, cfg: SeedConfig, max_corners: int = 4096) -> np.ndarray:
    """Shi-Tomasi corners of a (H, W) uint8 image, strongest first: (M, 2) x,y."""
    import cv2

    pts = cv2.goodFeaturesToTrack(
        gray, maxCorners=max_corners, qualityLevel=cfg.quality,
        minDistance=cfg.min_distance, blockSize=cfg.block_size,
    )
    if pts is None:
        return np.empty((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)  # already sorted by quality


def corners_by_cell(corners: np.ndarray, cell: int, ncols: int, nrows: int) -> dict[int, tuple[float, float]]:
    """Strongest corner per grid cell (first wins, since corners are quality-sorted)."""
    out: dict[int, tuple[float, float]] = {}
    for x, y in corners:
        ci = cell_of(x, y, cell, ncols, nrows)
        if ci not in out:
            out[ci] = (float(x), float(y))
    return out


def point_in_cell(
    ci: int, cell: int, ncols: int, proc_w: int, proc_h: int,
    method: str, corner_cells: dict[int, tuple[float, float]] | None,
) -> tuple[float, float] | None:
    """Pick the seed point for cell ``ci``: a Shi-Tomasi corner or the centre.

    Returns ``None`` when ``method=="shitomasi"`` and the cell holds no corner --
    the caller skips that cell (untrackable / textureless).
    """
    if method == "grid":
        return cell_center(ci, cell, ncols, proc_w, proc_h)
    if corner_cells is not None and ci in corner_cells:
        return corner_cells[ci]
    return None


def empty_viable_cells(
    pts: np.ndarray, vis: np.ndarray, cell: int, proc_w: int, proc_h: int,
    attempted: set[tuple[int, int]], *, min_persist: int = 2,
) -> list[tuple[int, int]]:
    """``(cell, t)`` pairs that are empty at frame t but sit in a *viable* cell.

    A cell is *occupied* at frame t if a visible track is in it. It is *viable*
    if a persisting track (visible in >= ``min_persist`` frames) passes through
    it -- so cells whose seeds die after one frame (sky) are never re-seeded.
    Each ``(cell, t)`` is recorded in ``attempted`` and returned at most once, so
    an outer loop over this function converges. Used by the CoTracker frontend
    (KLT gates on Shi-Tomasi corners directly instead).
    """
    T, _, _ = pts.shape
    ncols, nrows = grid_dims(proc_w, proc_h, cell)
    persists = vis.sum(axis=0) >= min_persist
    occupancy = [set() for _ in range(T)]
    viable: set[int] = set()
    for t in range(T):
        for k in np.nonzero(vis[t])[0]:
            x, y = pts[t, k]
            if 0 <= x < proc_w and 0 <= y < proc_h:
                ci = cell_of(x, y, cell, ncols, nrows)
                occupancy[t].add(ci)
                if persists[k]:
                    viable.add(ci)
    out: list[tuple[int, int]] = []
    for ci in viable:
        for t in range(T):
            if ci in occupancy[t] or (ci, t) in attempted:
                continue
            attempted.add((ci, t))
            out.append((ci, t))
    return out
