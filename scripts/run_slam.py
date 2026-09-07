#!/usr/bin/env python3
"""Step-0 entrypoint: run the full pipeline for one scene.

    python scripts/run_slam.py --dataroot data/nuscenes --scene scene-0061

Assembles the scene's SlamInputs (keyframes + cached CoTracker tracks + road
masks + IMU/wheel/GPS streams), calls the factor graph (``MonocularSLAM.run``),
and -- when it returns an estimate -- evaluates ATE/RPE against nuScenes GT and
writes the trajectory figure (``--out``), optionally logging the reconstruction
to Rerun (``--live``).

Until the backend is written this reports exactly what reached the seam and
exits cleanly, so the whole pipeline is verifiable from day one. Run
``scripts/run_frontend.py`` first to populate the cache.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.pipeline import PipelineConfig, run_scene  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--align", default="sim3", choices=["none", "se3", "sim3"])
    p.add_argument("--out", type=Path, default=Path("out/trajectory.png"))
    p.add_argument("--live", action="store_true", help="also log the estimate to a Rerun viewer")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    from nuslam.data import NuScenesMonoSource

    scene = args.scene or NuScenesMonoSource(args.dataroot, args.version).list_scenes()[0][1]
    config = PipelineConfig(
        dataroot=args.dataroot, scene=scene, version=args.version, camera=args.camera,
        cache_root=args.cache_root, max_frames=args.max_frames, align=args.align,
    )

    result = run_scene(config)

    if not result.backend_ready:
        print("\n" + "=" * 74)
        print("Pipeline reached the factor-graph seam. MonocularSLAM.run is unimplemented")
        print("(core estimator substance, left unimplemented by design).")
        print("Everything above is live: the inputs it printed are exactly what the")
        print("graph receives. Implement nuslam.backend.graph.MonocularSLAM.run,")
        print("then re-run this script for eval + the trajectory figure.")
        print("=" * 74)
        return 0

    import numpy as np

    from nuslam.viz import log_estimate, plot_trajectory
    est = result.estimate
    gt = np.stack([kf.ego2global_gt.matrix() for kf in result.inputs.keyframes])
    print(f"\neval: {result.errors}")

    plot_trajectory(est.poses_ego2global, gt, landmarks=est.landmarks, align=args.align,
                    save_path=args.out, title=f"{scene}: monocular SLAM vs. GT")
    print(f"wrote {args.out}")

    if args.live:
        from nuslam.viz import rerun_logging as rrlog
        rrlog.init(f"nuslam-{scene}", spawn=True)
        log_estimate(est.poses_ego2global, gt, landmarks=est.landmarks,
                     landmark_is_ground=est.landmark_is_ground, align=args.align)
        print("logged estimate to the Rerun viewer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
