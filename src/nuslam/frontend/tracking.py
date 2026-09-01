"""Offline point tracking for the SLAM frontend -- two selectable backends.

Both produce a :class:`~nuslam.types.TrackSet` of long-range pixel correspondences
(the reprojection-factor input) from a scene's keyframes, using Shi-Tomasi seeding
(see :mod:`nuslam.frontend.seeding`) and grid replenishment so density is held up
across the scene instead of decaying from a single frame-0 seeding:

* ``"cotracker"`` -- Meta's CoTracker (learned, transformer). Robust through
  occlusion; run offline via ``torch.hub``. Replenishment iterates the offline
  model with a growing set of timestamped ``(t, x, y)`` queries.
* ``"klt"`` -- classic pyramidal Lucas-Kanade (``cv2.calcOpticalFlowPyrLK``) with
  forward-backward error gating. Single forward pass; empty cells are re-seeded
  with fresh Shi-Tomasi corners each frame. A fast, fully-classical baseline to
  compare against the learned tracker.

Both are delegated frontend: correspondence quality feeds the graph but neither
tracker is estimator substance. Choose with ``TrackConfig.backend`` or
:func:`make_tracker`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..types import GroundMask, Keyframe, TrackSet
from . import seeding
from .refine import RefineConfig, refine_tracks_subpixel
from .seeding import SeedConfig

log = logging.getLogger(__name__)

_HUB_REPO = "facebookresearch/co-tracker"
_HUB_MODEL = "cotracker3_offline"


@dataclass
class TrackConfig:
    backend: str = "cotracker"       # "cotracker" | "klt"
    cell_size_px: int = 120          # grid cell size in FULL-RES pixels
    replenish: bool = True           # re-seed empty cells as tracks drift away
    min_visible_frames: int = 2      # drop tracks visible in fewer frames than this
    max_tracks: int = 4000           # hard cap on total seeded tracks
    downscale_long_side: int = 640   # track on this resolution; 0 = full res
    device: str | None = None        # cotracker: None -> cuda if available else cpu
    query_frame: int = 0             # frame for the initial full-grid seeding
    seed: SeedConfig = field(default_factory=SeedConfig)  # Shi-Tomasi vs grid centre
    refine: RefineConfig = field(default_factory=RefineConfig)  # subpixel cornerSubPix pass
    # cotracker-only
    max_replenish_iters: int = 4     # safety cap on offline re-runs (converges in ~2-4)
    # klt-only
    klt_win: int = 21                # LK search window (proc px)
    klt_levels: int = 3              # pyramid levels
    klt_fb_thresh: float = 1.0       # forward-backward consistency threshold (proc px)


# ---- shared helpers ------------------------------------------------------

def _scale(cfg: TrackConfig, w: int, h: int) -> float:
    return 1.0 if not cfg.downscale_long_side else min(1.0, cfg.downscale_long_side / max(w, h))


def _load_proc_frames(keyframes, proc_h, proc_w):
    """Decode + resize keyframes once -> (rgb list, gray list) at proc resolution."""
    import cv2

    rgb, gray = [], []
    for kf in keyframes:
        img = kf.image()
        if img.shape[:2] != (proc_h, proc_w):
            img = cv2.resize(img, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        rgb.append(np.ascontiguousarray(img))
        gray.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return rgb, gray


def _corner_cells_per_frame(gray, cfg: TrackConfig, cell, ncols, nrows):
    """Precompute {cell -> strongest corner} per frame (for Shi-Tomasi seeding)."""
    if cfg.seed.method != "shitomasi":
        return None
    return [seeding.corners_by_cell(seeding.shi_tomasi_corners(g, cfg.seed), cell, ncols, nrows)
            for g in gray]


def _label_ground(keyframes, masks, pts_full, seed_frame):
    if masks is None:
        return None
    n = pts_full.shape[1]
    is_ground = np.zeros(n, dtype=bool)
    for k in range(n):
        s = int(seed_frame[k])
        gm = masks.get(keyframes[s].token)
        if gm is not None:
            is_ground[k] = bool(gm.sample_at(pts_full[s, k][None])[0])
    return is_ground


def _finalize(keyframes, masks, pts_proc, vis, seed_frame, scale, cfg, T, proc_w, proc_h, cell):
    """Shared post-processing: rescale, drop short tracks, ground-label, pack."""
    pts_full = pts_proc / scale
    keep = vis.sum(axis=0) >= cfg.min_visible_frames
    pts_full, vis, seed_frame = pts_full[:, keep], vis[:, keep], seed_frame[keep]
    is_ground = _label_ground(keyframes, masks, pts_full, seed_frame)
    log.info("tracked %d points across %d frames (proc %dx%d, cell %dpx, %s, seed=%s, replenish=%s)",
             pts_full.shape[1], T, proc_w, proc_h, cell, cfg.backend, cfg.seed.method, cfg.replenish)
    tracks = TrackSet(
        frame_tokens=[kf.token for kf in keyframes],
        points=pts_full.astype(np.float32),
        visible=vis,
        query_frame=cfg.query_frame,
        is_ground=is_ground,
        seed_frame=seed_frame.astype(np.int64),
    )
    if cfg.refine.enabled:
        tracks, stats = refine_tracks_subpixel(tracks, keyframes, cfg.refine)
        log.info("%s", stats)
    return tracks


# ---- CoTracker backend ---------------------------------------------------

class CoTrackerFrontend:
    """Learned tracking via CoTracker, with Shi-Tomasi seeding + replenishment."""

    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig()
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch

        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading CoTracker %s (%s) on %s", _HUB_MODEL, _HUB_REPO, self._device)
        self._model = torch.hub.load(_HUB_REPO, _HUB_MODEL).to(self._device).eval()

    def track_scene(self, keyframes: Sequence[Keyframe], *,
                    masks: dict[str, GroundMask] | None = None) -> TrackSet:
        self._ensure_loaded()
        import torch

        cfg = self.config
        if len(keyframes) < 2:
            raise ValueError("need >= 2 keyframes to track")
        full_h, full_w = keyframes[0].calib.height, keyframes[0].calib.width
        scale = _scale(cfg, full_w, full_h)
        proc_w, proc_h = round(full_w * scale), round(full_h * scale)
        cell = max(1, round(cfg.cell_size_px * scale))
        ncols, nrows = seeding.grid_dims(proc_w, proc_h, cell)

        rgb, gray = _load_proc_frames(keyframes, proc_h, proc_w)
        corner_cells = _corner_cells_per_frame(gray, cfg, cell, ncols, nrows)
        video = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float()[None].to(self._device)
        T = video.shape[1]

        # initial seeds: every cell on the query frame
        init_pairs = [(ci, cfg.query_frame) for ci in range(ncols * nrows)]
        queries = self._pairs_to_queries(init_pairs, corner_cells, cell, ncols, proc_w, proc_h)
        if len(queries) == 0:
            raise RuntimeError("no seed points found on the query frame")
        pts, vis = self._run(video, queries)

        if cfg.replenish:
            attempted: set[tuple[int, int]] = set()
            for it in range(cfg.max_replenish_iters):
                cells = seeding.empty_viable_cells(pts, vis, cell, proc_w, proc_h, attempted)
                new = self._pairs_to_queries(cells, corner_cells, cell, ncols, proc_w, proc_h)
                room = cfg.max_tracks - len(queries)
                if len(new) == 0 or room <= 0:
                    break
                if len(new) > room:
                    new = new[:room]
                queries = np.concatenate([queries, new], axis=0)
                pts, vis = self._run(video, queries)
                log.info("replenish iter %d: +%d tracks -> %d total", it + 1, len(new), len(queries))

        seed_frame = queries[:, 0].astype(np.int64)
        return _finalize(keyframes, masks, pts, vis, seed_frame, scale, cfg, T, proc_w, proc_h, cell)

    @staticmethod
    def _pairs_to_queries(pairs, corner_cells, cell, ncols, proc_w, proc_h) -> np.ndarray:
        """(cell, t) pairs -> (M, 3) (t, x, y) queries, skipping cells with no seed."""
        method = "shitomasi" if corner_cells is not None else "grid"
        out = []
        for ci, t in pairs:
            cc = corner_cells[t] if corner_cells is not None else None
            p = seeding.point_in_cell(ci, cell, ncols, proc_w, proc_h, method, cc)
            if p is not None:
                out.append((float(t), p[0], p[1]))
        return np.asarray(out, dtype=np.float32).reshape(-1, 3)

    def _run(self, video, queries: np.ndarray):
        import torch

        q = torch.from_numpy(np.ascontiguousarray(queries, np.float32))[None].to(self._device)
        with torch.no_grad():
            tracks, vis = self._model(video, queries=q)
        return tracks[0].cpu().numpy().astype(np.float32), vis[0].cpu().numpy().astype(bool)


# ---- KLT (OpenCV) backend ------------------------------------------------

class KLTFrontend:
    """Classic pyramidal Lucas-Kanade tracking with Shi-Tomasi seed + replenishment.

    Single forward pass: propagate active points frame-to-frame with
    ``cv2.calcOpticalFlowPyrLK``, drop a track when LK fails, the point leaves the
    image, or the forward-backward check exceeds ``klt_fb_thresh``, then re-seed
    any empty cell that holds a Shi-Tomasi corner. The corner detector is the
    viability gate, so untextured cells (sky) are never seeded.
    """

    def __init__(self, config: TrackConfig | None = None) -> None:
        self.config = config or TrackConfig(backend="klt")

    def track_scene(self, keyframes: Sequence[Keyframe], *,
                    masks: dict[str, GroundMask] | None = None) -> TrackSet:
        import cv2

        cfg = self.config
        if len(keyframes) < 2:
            raise ValueError("need >= 2 keyframes to track")
        full_h, full_w = keyframes[0].calib.height, keyframes[0].calib.width
        scale = _scale(cfg, full_w, full_h)
        proc_w, proc_h = round(full_w * scale), round(full_h * scale)
        cell = max(1, round(cfg.cell_size_px * scale))
        ncols, nrows = seeding.grid_dims(proc_w, proc_h, cell)
        _, gray = _load_proc_frames(keyframes, proc_h, proc_w)
        T = len(gray)

        method = cfg.seed.method
        corner_cells = _corner_cells_per_frame(gray, cfg, cell, ncols, nrows)
        lk = dict(winSize=(cfg.klt_win, cfg.klt_win), maxLevel=cfg.klt_levels,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        # per-track records grown as tracking proceeds
        tracks: list[dict] = []          # {seed, xs:{t:(x,y)}}
        active: list[int] = []           # indices into tracks currently alive
        pos = np.empty((0, 2), np.float32)  # current positions of active tracks

        def seed_empty(t: int):
            nonlocal pos
            occupied = set()
            for xy in pos:
                occupied.add(seeding.cell_of(xy[0], xy[1], cell, ncols, nrows))
            new_pts = []
            for ci in range(ncols * nrows):
                if ci in occupied or len(tracks) >= cfg.max_tracks:
                    continue
                cc = corner_cells[t] if corner_cells is not None else None
                p = seeding.point_in_cell(ci, cell, ncols, proc_w, proc_h, method, cc)
                if p is None:
                    continue
                idx = len(tracks)
                tracks.append({"seed": t, "xs": {t: (float(p[0]), float(p[1]))}})
                active.append(idx)
                new_pts.append(p)
            if new_pts:
                pos = np.concatenate([pos, np.asarray(new_pts, np.float32)], axis=0)

        seed_empty(0)  # frame 0
        for t in range(1, T):
            if len(active):
                p0 = pos.reshape(-1, 1, 2)
                p1, st, _ = cv2.calcOpticalFlowPyrLK(gray[t - 1], gray[t], p0, None, **lk)
                p0b, _, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - 1], p1, None, **lk)
                fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
                p1 = p1.reshape(-1, 2)
                good = (st.flatten() == 1) & (fb < cfg.klt_fb_thresh) \
                    & (p1[:, 0] >= 0) & (p1[:, 0] < proc_w) & (p1[:, 1] >= 0) & (p1[:, 1] < proc_h)
                survivors, surv_pos = [], []
                for a, ok, xy in zip(active, good, p1):
                    if ok:
                        tracks[a]["xs"][t] = (float(xy[0]), float(xy[1]))
                        survivors.append(a)
                        surv_pos.append(xy)
                active = survivors
                pos = np.asarray(surv_pos, np.float32).reshape(-1, 2)
            if cfg.replenish or t == 0:
                seed_empty(t)

        # pack ragged per-track records into (T, N, 2) + visibility
        n = len(tracks)
        pts = np.zeros((T, n, 2), np.float32)
        vis = np.zeros((T, n), bool)
        seed_frame = np.zeros(n, np.int64)
        for k, tr in enumerate(tracks):
            seed_frame[k] = tr["seed"]
            for t, xy in tr["xs"].items():
                pts[t, k] = xy
                vis[t, k] = True
        return _finalize(keyframes, masks, pts, vis, seed_frame, scale, cfg, T, proc_w, proc_h, cell)


def make_tracker(config: TrackConfig | None = None):
    """Return the tracker frontend selected by ``config.backend``."""
    config = config or TrackConfig()
    if config.backend == "klt":
        return KLTFrontend(config)
    if config.backend == "cotracker":
        return CoTrackerFrontend(config)
    raise ValueError(f"unknown tracker backend {config.backend!r} (want 'cotracker' or 'klt')")
