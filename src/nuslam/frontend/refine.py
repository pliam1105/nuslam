"""Subpixel refinement of track coordinates via ``cv2.cornerSubPix``.

Both trackers run on a downscaled copy (CoTracker also caps internally at
384x512), so their points, scaled back to full resolution, carry ~2-5 px of
localization error. This pass re-localizes each *visible* track point to subpixel
accuracy on the FULL-resolution grayscale image, where the real detail lives --
tightening the reprojection observations the graph will consume.

Safety: ``cornerSubPix`` iterates to where local gradients vanish, which is well
posed at a genuine corner but ill-posed on an edge (it slides along it) or a flat
patch. So a refinement is **accepted only if it moves the point less than
``max_shift_px``**; a larger move means the window latched onto different
structure, and the original tracker location is kept. That guard doubles as an
implicit "is this really a corner" test, so the pass is safe to run on every
point rather than only detected corners.

Pure post-process on coordinates -- visibility, seed frames and ground labels are
untouched. Nothing here concerns the factor graph -- frontend plumbing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from ..types import Keyframe, TrackSet

log = logging.getLogger(__name__)


@dataclass
class RefineConfig:
    enabled: bool = True
    win: int = 7             # cornerSubPix search half-window (winSize=(win, win))
    max_shift_px: float = 2.0  # reject (keep original) refinements moving more than this
    iters: int = 40
    eps: float = 0.01


@dataclass
class RefineStats:
    n_points: int        # visible observations considered
    n_refined: int       # accepted subpixel adjustments
    n_rejected: int      # rejected (moved > max_shift) -> kept original
    mean_shift_px: float # mean accepted subpixel shift (full-res px)
    median_shift_px: float

    def __str__(self) -> str:  # pragma: no cover
        frac = 0.0 if not self.n_points else self.n_refined / self.n_points
        return (f"subpixel refine: {self.n_refined}/{self.n_points} obs ({frac:.0%}) refined, "
                f"{self.n_rejected} rejected; shift mean={self.mean_shift_px:.3f} "
                f"median={self.median_shift_px:.3f} px")


def refine_points_on_frame(
    gray: np.ndarray, pts_xy: np.ndarray, cfg: RefineConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine (N, 2) full-res pixel points on one grayscale frame.

    Returns ``(refined_xy, accepted_mask, shift)``: points that moved more than
    ``max_shift_px`` (or sit within a window of the border) are left unchanged
    with ``accepted=False``. Pure -> unit-testable without keyframes.
    """
    import cv2

    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    h, w = gray.shape
    n = len(pts_xy)
    refined = np.asarray(pts_xy, dtype=np.float32).copy()
    accepted = np.zeros(n, dtype=bool)
    shift = np.zeros(n, dtype=np.float32)
    if n == 0:
        return refined, accepted, shift

    b = cfg.win + 1  # cornerSubPix needs a full window inside the image
    inside = ((refined[:, 0] >= b) & (refined[:, 0] < w - b)
              & (refined[:, 1] >= b) & (refined[:, 1] < h - b))
    idx = np.nonzero(inside)[0]
    if len(idx) == 0:
        return refined, accepted, shift

    corners = refined[idx].reshape(-1, 1, 2).copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, cfg.iters, cfg.eps)
    cv2.cornerSubPix(gray, corners, (cfg.win, cfg.win), (-1, -1), criteria)
    moved = corners.reshape(-1, 2)
    d = np.linalg.norm(moved - refined[idx], axis=1)
    ok = d <= cfg.max_shift_px
    sel = idx[ok]
    refined[sel] = moved[ok]
    accepted[sel] = True
    shift[sel] = d[ok]
    return refined, accepted, shift


def refine_tracks_subpixel(
    tracks: TrackSet, keyframes: Sequence[Keyframe], cfg: RefineConfig | None = None
) -> tuple[TrackSet, RefineStats]:
    """Refine every visible track point on its full-resolution keyframe image.

    ``keyframes`` must cover the tracks' ``frame_tokens``. Returns a new TrackSet
    (same visibility/seed/ground) plus :class:`RefineStats`.
    """
    cfg = cfg or RefineConfig()
    kf_by_token = {kf.token: kf for kf in keyframes}
    pts = tracks.points.copy()

    n_points = n_refined = n_rejected = 0
    shifts: list[float] = []
    for t, tok in enumerate(tracks.frame_tokens):
        vis_k = np.nonzero(tracks.visible[t])[0]
        if len(vis_k) == 0:
            continue
        kf = kf_by_token.get(tok)
        if kf is None:
            continue
        import cv2

        gray = cv2.cvtColor(kf.image(), cv2.COLOR_RGB2GRAY)
        refined, accepted, shift = refine_points_on_frame(gray, pts[t, vis_k], cfg)
        pts[t, vis_k] = refined
        n_points += len(vis_k)
        n_refined += int(accepted.sum())
        n_rejected += int((~accepted).sum())
        shifts.extend(shift[accepted].tolist())

    shifts_arr = np.asarray(shifts) if shifts else np.zeros(1)
    stats = RefineStats(
        n_points=n_points, n_refined=n_refined, n_rejected=n_rejected,
        mean_shift_px=float(shifts_arr.mean()), median_shift_px=float(np.median(shifts_arr)),
    )
    return replace(tracks, points=pts.astype(np.float32)), stats
