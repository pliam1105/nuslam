#!/usr/bin/env python3
"""Run Depth Anything 3 over a scene and cache the depth maps.

    python scripts/run_depth.py --scene scene-0061 --preview out/depth_preview

Runs DA3Mono-Large per CAM_FRONT keyframe, writes ``depth.npz`` under the cache
root, and (with --preview) writes colormapped depth PNGs. If lidar is present it
also scores the recovered metric scale (median lidar/pred), which is the number
the reconstruction's anchor must reproduce. First run downloads DA3 weights.

DA3 is used only to initialize -- its depth is cached here; the back-projection of
that depth into Gaussians is written by hand elsewhere.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource, project_lidar_to_camera  # noqa: E402
from nuslam.eval import evaluate_depth  # noqa: E402
from nuslam.frontend import DepthConfig, DepthEstimator, cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--preview", type=Path, default=None, help="dir for colormapped depth PNGs")
    return p.parse_args()


def _preview(out_dir: Path, source, keyframes, depths, n: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_tok = {d.token: d for d in depths}
    idxs = np.linspace(0, len(keyframes) - 1, min(n, len(keyframes))).astype(int)
    for i in idxs:
        kf = keyframes[i]
        dm = by_tok[kf.token]
        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
        ax[0].imshow(kf.image()); ax[0].set_axis_off(); ax[0].set_title(f"{kf.scene_name} kf{kf.frame_index}")
        im = ax[1].imshow(dm.depth, cmap="turbo"); ax[1].set_axis_off()
        ax[1].set_title(f"DA3 depth (relative, metric={dm.is_metric})")
        fig.colorbar(im, ax=ax[1], fraction=0.03)
        path = out_dir / f"depth_{kf.scene_name}_kf{kf.frame_index:03d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {path}")
        uv, ld = project_lidar_to_camera(source, kf)
        if len(ld):
            print(f"    kf{kf.frame_index}: {evaluate_depth(dm.depth, uv, ld)}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name
    print(f"scene {scene_name!r}: {len(keyframes)} keyframes")

    est = DepthEstimator(DepthConfig(process_res=args.process_res))
    depths = est.estimate_scene(keyframes)
    cache.save_depth(args.cache_root, scene_name, depths)
    rng = [(float(d.depth.min()), float(d.depth.max())) for d in depths]
    print(f"depth: saved {len(depths)} (metric={depths[0].is_metric}, "
          f"range ~{np.mean([r[0] for r in rng]):.2f}..{np.mean([r[1] for r in rng]):.2f})")

    if args.preview is not None:
        print("writing previews...")
        _preview(args.preview, source, keyframes, depths)
    print(f"\ncache -> {args.cache_root / scene_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
