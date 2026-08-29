#!/usr/bin/env python3
"""Render a video of the cached frontend output (road mask + tracks) for a scene.

    python scripts/render_frontend_video.py --scene scene-0061 --out out/frontend_scene-0061.mp4

Reads the cached ``masks.npz`` / ``tracks.npz`` (produced by run_frontend.py) and
draws, per keyframe: the road mask as a green overlay and each visible track as a
point with a short motion trail -- cyan for ground-flagged tracks, orange
otherwise. Pure visualization plumbing; no models are run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.frontend import cache  # noqa: E402

# BGR (cv2) colors.
_GROUND = (255, 200, 0)    # cyan-ish
_OTHER = (40, 140, 255)    # orange
_MASK = (60, 200, 60)      # green overlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--tracker", default="", help="load a tagged tracks variant (e.g. cotracker, klt); '' = canonical")
    p.add_argument("--out", type=Path, default=None, help="output .mp4 (default out/frontend_<scene>.mp4)")
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--trail", type=int, default=6, help="track trail length in frames")
    p.add_argument("--width", type=int, default=1280, help="output width (px); height scales")
    p.add_argument("--mask-alpha", type=float, default=0.4)
    return p.parse_args()


def _draw_frame(img_bgr, mask, pts, vis, is_ground, i, trail, alpha):
    """Draw mask overlay + track trails/points onto a copy of img_bgr."""
    out = img_bgr.copy()
    if mask is not None:
        overlay = out.copy()
        overlay[mask] = _MASK
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    n = pts.shape[1]
    for k in range(n):
        color = _GROUND if (is_ground is not None and is_ground[k]) else _OTHER
        # trail over the last `trail` frames where the track was visible
        j0 = max(0, i - trail)
        prev = None
        for j in range(j0, i + 1):
            if not vis[j, k]:
                prev = None
                continue
            x, y = pts[j, k]
            cur = (int(round(x)), int(round(y)))
            if prev is not None:
                cv2.line(out, prev, cur, color, 1, cv2.LINE_AA)
            prev = cur
        if vis[i, k]:
            x, y = pts[i, k]
            cv2.circle(out, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    return out


def _banner(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene)
    scene_name = keyframes[0].scene_name

    tracks = cache.load_tracks(args.cache_root, scene_name, variant=args.tracker)
    if tracks is None:
        print(f"no cached tracks (variant={args.tracker!r}) for {scene_name!r} under "
              f"{args.cache_root} -- run scripts/run_frontend.py --scene {scene_name} first",
              file=sys.stderr)
        return 1
    masks = cache.load_masks(args.cache_root, scene_name) or {}

    out_path = args.out or (Path("out") / f"frontend_{scene_name}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h0, w0 = keyframes[0].calib.height, keyframes[0].calib.width
    scale = args.width / w0
    out_w, out_h = args.width, int(round(h0 * scale))

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (out_w, out_h))
    if not writer.isOpened():
        print("cv2.VideoWriter failed to open (mp4v codec unavailable)", file=sys.stderr)
        return 2

    ng = "n/a" if tracks.is_ground is None else int(tracks.is_ground.sum())
    print(f"{scene_name}: {len(keyframes)} frames, {tracks.num_tracks} tracks "
          f"({ng} ground) -> {out_path} @ {args.fps} fps")
    for i, kf in enumerate(keyframes):
        img = cv2.cvtColor(kf.image(), cv2.COLOR_RGB2BGR)
        gm = masks.get(kf.token)
        frame = _draw_frame(img, gm.mask if gm else None, tracks.points, tracks.visible,
                            tracks.is_ground, i, args.trail, args.mask_alpha)
        vis_n = int(tracks.visible[i].sum())
        _banner(frame, f"{scene_name}  kf {i:02d}/{len(keyframes)-1}   "
                       f"road mask (green) + tracks: cyan=ground orange=other   visible={vis_n}")
        writer.write(cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA))
    writer.release()
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
