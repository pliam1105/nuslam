#!/usr/bin/env python3
"""Visualize one monocular nuScenes scene in Rerun.

    python scripts/visualize_scene.py --scene scene-0061            # opens the viewer
    python scripts/visualize_scene.py --scene scene-0061 --save out/scene-0061.rrd

Logs, per keyframe: the ego + camera poses (transform tree), the camera pinhole
and image, and the growing ground-truth trajectory. With cached road masks
(--masks, from scripts/run_frontend.py) it also overlays the ground segmentation.
Open a saved recording later with:  .venv/bin/rerun out/scene-0061.rrd

Raw-data visualization; the reconstruction's own output is logged by the modelling
code via nuslam.viz.rerun_logging / log_estimate.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.viz import rerun_logging as rrlog  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--save", type=Path, default=None, help="write a .rrd recording instead of spawning a viewer")
    p.add_argument("--connect", default=None, help="gRPC url of a running Rerun viewer to attach to")
    p.add_argument("--masks", action="store_true", help="overlay cached road/ground masks")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--realtime", action="store_true",
                   help="pace a live viewer by keyframe timestamps (default: log as fast as possible)")
    p.add_argument("--rate", type=float, default=1.0,
                   help="playback speed multiplier when --realtime (2 = 2x, 0.5 = half)")
    p.add_argument("--loop", action="store_true", help="replay the scene until interrupted (live only)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name
    print(f"scene {scene_name!r}: {len(keyframes)} keyframes on {args.camera}")

    spawn = args.save is None and args.connect is None
    live = args.save is None  # spawned or connected viewer
    paced = live and args.realtime  # real-time pacing is opt-in
    rrlog.init(f"nuslam-{scene_name}", spawn=spawn, save=args.save, connect=args.connect)

    masks = cache.load_masks(args.cache_root, scene_name) if args.masks else None
    if args.masks and not masks:
        print(f"no cached masks under {args.cache_root}/{scene_name} -- run scripts/run_frontend.py first")

    t0 = keyframes[0].timestamp_us
    gt_xyz = source.gt_trajectory(keyframes)[:, :3, 3]

    def play_once() -> None:
        prev_t = None
        for i, kf in enumerate(keyframes):
            if paced and prev_t is not None and args.rate > 0:
                dt = (kf.timestamp_us - prev_t) / 1e6 / args.rate  # pace by keyframe timestamps
                if dt > 0:
                    time.sleep(dt)
            prev_t = kf.timestamp_us
            gm = masks[kf.token].mask if (masks and kf.token in masks) else None
            rrlog.log_keyframe(kf, t0, ground_mask=gm)
            rrlog.log_trajectory("traj/gt", gt_xyz[: i + 1], color=(150, 150, 150))

    try:
        play_once()
        while live and args.loop:
            print("  (loop) replaying scene")
            play_once()
    except KeyboardInterrupt:
        print("\nstopped")
    print(f"logged {len(keyframes)} keyframes"
          + (f" -> {args.save}" if args.save else " to the Rerun viewer"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
