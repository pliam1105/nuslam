#!/usr/bin/env python3
"""Accumulate the DA3 depth cloud (RGB) vs lidar in Rerun.

    python scripts/visualize_depth.py --scene scene-0061
    python scripts/visualize_depth.py --scene scene-0061 --save out/depth_cloud.rrd

Per keyframe it logs, in the global frame, an accumulating point cloud (each
frame under its own entity, so they build up over the timeline):
  * world/lidar/NNN  -- LIDAR_TOP points, grey (metric reference);
  * world/depth/NNN  -- the DA3 depth back-projected to RGB points.

The depth cloud needs the pixel->3D back-projection, which is core initialization
(nuslam.recon.depth_init.unproject_depth_to_world) and is written by hand -- until
it is implemented this logs lidar only and says so. Run scripts/run_depth.py first
to cache the depth maps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource, lidar_points_global  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.recon import unproject_depth_to_world  # noqa: E402
from nuslam.viz import rerun_logging as rrlog  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--save", type=Path, default=None, help="write a .rrd instead of spawning a viewer")
    p.add_argument("--connect", default=None, help="gRPC url of a running Rerun viewer")
    p.add_argument("--stride", type=int, default=8, help="depth pixel stride (viz density)")
    p.add_argument("--max-frames", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name
    print(f"scene {scene_name!r}: {len(keyframes)} keyframes")

    depth = cache.load_depth(args.cache_root, scene_name)
    if depth is None:
        print(f"no cached depth under {args.cache_root}/{scene_name} -- run scripts/run_depth.py first")

    spawn = args.save is None and args.connect is None
    rrlog.init(f"nuslam-depth-{scene_name}", spawn=spawn, save=args.save, connect=args.connect)

    t0 = keyframes[0].timestamp_us
    depth_ok = True
    warned = False
    for i, kf in enumerate(keyframes):
        rrlog.set_time(kf.frame_index, kf.timestamp_us, t0)
        lidar_xyz = lidar_points_global(source, kf)
        if len(lidar_xyz):
            rrlog.log_points(f"lidar/{i:03d}", lidar_xyz, colors=(190, 190, 190), radii=0.03)
        if depth is not None and depth_ok and kf.token in depth:
            try:
                pts, cols = unproject_depth_to_world(depth[kf.token], kf, stride=args.stride)
                rrlog.log_points(f"depth/{i:03d}", pts, colors=cols, radii=0.05)
            except NotImplementedError as exc:
                depth_ok = False
                if not warned:
                    print(f"\n[depth cloud skipped] {exc}\n")
                    warned = True
    print(f"logged {len(keyframes)} keyframes"
          + (f" -> {args.save}" if args.save else " to the Rerun viewer")
          + ("  (lidar only until the back-projection is written)" if not depth_ok else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
