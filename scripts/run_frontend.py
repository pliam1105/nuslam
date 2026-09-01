#!/usr/bin/env python3
"""Run the delegated frontend offline and cache its output.

    python scripts/run_frontend.py --dataroot data/nuscenes --scene scene-0061 \
        --preview out/frontend

Runs prompt-based road segmentation (CLIPSeg) and point tracking (CoTracker) over
a scene's CAM_FRONT keyframes and writes ``tracks.npz`` / ``masks.npz`` under the
cache root, ready for ``scripts/run_slam.py``. With ``--preview DIR`` it also
writes a few overlay PNGs so the mask and tracks can be verified visually before
the graph is built on them.

First run downloads the CLIPSeg (~150 MB) and CoTracker weights.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.frontend import (  # noqa: E402
    RefineConfig, RoadSegmenter, SeedConfig, SegConfig, TrackConfig, cache, make_tracker,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--tracker", choices=["cotracker", "klt"], default="cotracker",
                   help="tracking backend: CoTracker (learned) or KLT (OpenCV Lucas-Kanade)")
    p.add_argument("--seed-method", choices=["shitomasi", "grid"], default="shitomasi",
                   help="seed point within each cell: Shi-Tomasi corner or cell centre")
    p.add_argument("--cell-size", type=int, default=120,
                   help="grid cell size in full-res px (seed + replenish density)")
    p.add_argument("--no-replenish", action="store_true",
                   help="disable grid replenishment (single frame-0 seeding)")
    p.add_argument("--min-visible", type=int, default=2, help="drop tracks visible in fewer frames")
    p.add_argument("--max-tracks", type=int, default=4000)
    p.add_argument("--st-quality", type=float, default=0.01, help="Shi-Tomasi qualityLevel")
    p.add_argument("--st-min-distance", type=int, default=7, help="Shi-Tomasi min corner spacing (proc px)")
    p.add_argument("--no-refine", action="store_true", help="disable subpixel cornerSubPix refinement")
    p.add_argument("--refine-win", type=int, default=7, help="cornerSubPix search half-window (full-res px)")
    p.add_argument("--refine-max-shift", type=float, default=2.0,
                   help="reject refinements moving more than this (full-res px)")
    p.add_argument("--prompt", default="road", help="segmentation text prompt")
    p.add_argument("--seg-threshold", type=float, default=0.35)
    p.add_argument("--skip-tracks", action="store_true")
    p.add_argument("--skip-masks", action="store_true")
    p.add_argument("--preview", type=Path, default=None, help="dir for overlay PNGs")
    return p.parse_args()


def _preview(out_dir: Path, keyframes, masks, tracks, n: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    mask_by_tok = {m.token: m for m in masks} if masks else {}
    idxs = np.linspace(0, len(keyframes) - 1, min(n, len(keyframes))).astype(int)
    for i in idxs:
        kf = keyframes[i]
        img = kf.image()
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.imshow(img); ax.set_axis_off()
        gm = mask_by_tok.get(kf.token)
        if gm is not None:
            overlay = np.zeros((*gm.mask.shape, 4))
            overlay[gm.mask] = (0.1, 0.9, 0.3, 0.35)
            ax.imshow(overlay)
        if tracks is not None:
            pts = tracks.points[i]           # (N, 2)
            vis = tracks.visible[i]          # (N,)
            g = tracks.is_ground
            c = np.where((g if g is not None else np.zeros(len(pts), bool)), "cyan", "orange")
            ax.scatter(pts[vis, 0], pts[vis, 1], s=6, c=c[vis], alpha=0.8)
        ax.set_title(f"{kf.scene_name} kf{kf.frame_index}: road mask + tracks "
                     f"(cyan=ground)", fontsize=10)
        path = out_dir / f"frontend_{kf.scene_name}_kf{kf.frame_index:03d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name
    print(f"scene {scene_name!r}: {len(keyframes)} keyframes")

    masks = []
    mask_dict = None
    if not args.skip_masks:
        seg = RoadSegmenter(SegConfig(prompt=args.prompt, threshold=args.seg_threshold))
        masks = seg.segment_scene(keyframes)
        cache.save_masks(args.cache_root, scene_name, masks)
        mask_dict = {m.token: m for m in masks}
        cover = np.mean([m.mask.mean() for m in masks])
        print(f"masks: saved {len(masks)} (mean road coverage {cover:.1%})")
    else:
        mask_dict = cache.load_masks(args.cache_root, scene_name)  # reuse cached for ground labels
        if mask_dict:
            masks = list(mask_dict.values())
            print(f"masks: reusing {len(mask_dict)} cached for ground labeling")

    tracks = None
    if not args.skip_tracks:
        tracker = make_tracker(TrackConfig(
            backend=args.tracker,
            cell_size_px=args.cell_size,
            replenish=not args.no_replenish,
            min_visible_frames=args.min_visible,
            max_tracks=args.max_tracks,
            seed=SeedConfig(method=args.seed_method, quality=args.st_quality,
                            min_distance=args.st_min_distance),
            refine=RefineConfig(enabled=not args.no_refine, win=args.refine_win,
                                max_shift_px=args.refine_max_shift),
        ))
        tracks = tracker.track_scene(keyframes, masks=mask_dict)
        cache.save_tracks(args.cache_root, scene_name, tracks)                       # canonical (run_slam)
        cache.save_tracks(args.cache_root, scene_name, tracks, variant=args.tracker)  # tagged copy
        ng = "n/a" if tracks.is_ground is None else int(tracks.is_ground.sum())
        seeds = "1 (frame 0)" if tracks.seed_frame is None else f"{len(np.unique(tracks.seed_frame))} frames"
        per_frame = tracks.visible.sum(axis=1)
        print(f"tracks: saved {tracks.num_tracks} across {tracks.num_frames} frames "
              f"(ground-flagged {ng}, seeded over {seeds}); "
              f"visible/frame min={per_frame.min()} mean={per_frame.mean():.0f} max={per_frame.max()}")

    if args.preview is not None:
        print("writing previews...")
        _preview(args.preview, keyframes, masks, tracks)
    print(f"\ncache -> {args.cache_root / scene_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
