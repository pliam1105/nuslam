#!/usr/bin/env python3
"""Stream one monocular nuScenes scene to Foxglove over a live WebSocket bridge.

    python scripts/visualize_scene.py --dataroot data/nuscenes --scene scene-0061

Open Foxglove and connect to ws://localhost:8765. Add an Image panel on
/camera/CAM_FRONT and a 3D panel (it composes the transform tree and shows the
ground-truth trajectory and annotation boxes). Playback is paced by keyframe
timestamps and loops until Ctrl-C; --once plays a single pass.

Raw-data visualization plumbing -- the estimator's own output is streamed by
scripts/run_slam.py --live instead.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.viz import FoxgloveBridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None)
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--rate", type=float, default=1.0, help="playback speed multiplier")
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene)
    print(f"scene {scene!r}: {len(keyframes)} keyframes")
    gt_xyz = source.gt_trajectory(keyframes)[:, :3, 3]

    with FoxgloveBridge(port=args.port) as bridge:
        print(f"Foxglove live on ws://localhost:{args.port} — connect the app, Ctrl-C to stop")
        try:
            while True:
                prev_t = None
                for i, kf in enumerate(keyframes):
                    if prev_t is not None and args.rate > 0:
                        dt = (kf.timestamp_us - prev_t) / 1e6 / args.rate
                        if dt > 0:
                            time.sleep(dt)
                    prev_t = kf.timestamp_us
                    t = kf.timestamp_us
                    ego = kf.ego2global_gt
                    cam = kf.calib
                    bridge.publish_transform("/tf", "global", "ego",
                                             ego.t, ego.quaternion_wxyz(), t)
                    bridge.publish_transform("/tf", "ego", args.camera,
                                             cam.sensor2ego.t, cam.sensor2ego.quaternion_wxyz(), t)
                    bridge.publish_compressed_image(f"/camera/{args.camera}", args.camera,
                                                    kf.image_bytes(), t)
                    bridge.publish_line("/traj/gt", "global", gt_xyz[: i + 1], t,
                                        color=(0.6, 0.6, 0.6, 1.0), entity_id="gt")
                    print(f"  kf{i:3d}  t={t}")
                if args.once:
                    break
                print("  (loop) restarting scene")
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
