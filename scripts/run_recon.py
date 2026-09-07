#!/usr/bin/env python3
"""Run the DA3-Base full reconstruction over a scene and cache it.

    python scripts/run_recon.py --scene scene-0061 --preview out/recon_preview

Unlike ``run_depth.py`` (per-frame DA3Mono relative depth), this runs DA3-Base
*jointly* over all keyframes and caches, per frame, the depth PLUS DA3's own
world->camera pose and estimated intrinsics ``K_da3`` (rescaled to full image
resolution). These are the inputs the metric upgrade consumes:

    P~_i = K_true^{-1} K_da3 [R_i | t_i]      (normalized projective cameras)

DA3-Base is intrinsic-agnostic -- the true nuScenes intrinsics are NOT passed in;
DA3 estimates its own (typically wrong) K, and the metric upgrade rectifies it
against the true ``calib.intrinsic``. First run downloads DA3-Base weights.

The cache is written to the same ``depth.npz`` used by ``run_depth.py`` (the
extra ``extrinsic``/``intrinsic`` fields are simply populated), so downstream
loaders read it identically; this overwrites any DA3Mono depth for the scene.
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
from nuslam.frontend import DA3ReconEstimator, ReconConfig, cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap frames (whole scene reconstructed jointly -- lower this if OOM)")
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--model-id", default=None, help="override DA3 model id (default: DA3-Base)")
    p.add_argument("--preview", type=Path, default=None, help="dir for colormapped depth PNGs")
    return p.parse_args()


def _preview(out_dir: Path, keyframes, depths, n: int = 3) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    by_tok = {d.token: d for d in depths}
    idxs = np.linspace(0, len(keyframes) - 1, min(n, len(keyframes))).astype(int)
    for i in idxs:
        kf = keyframes[i]
        dm = by_tok[kf.token]
        f_da3, f_true = float(dm.intrinsic[0, 0]), float(kf.calib.intrinsic[0, 0])
        fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))
        ax[0].imshow(kf.image()); ax[0].set_axis_off()
        ax[0].set_title(f"{kf.scene_name} kf{kf.frame_index}")
        im = ax[1].imshow(dm.depth, cmap="turbo"); ax[1].set_axis_off()
        ax[1].set_title(f"DA3-Base depth  (f_da3={f_da3:.0f} vs f_true={f_true:.0f}, "
                        f"{100 * (f_da3 / f_true - 1):+.0f}%)")
        fig.colorbar(im, ax=ax[1], fraction=0.03)
        path = out_dir / f"recon_{kf.scene_name}_kf{kf.frame_index:03d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {path}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name
    print(f"scene {scene_name!r}: {len(keyframes)} keyframes (reconstructed jointly)")

    cfg = ReconConfig(process_res=args.process_res)
    if args.model_id:
        cfg.model_id = args.model_id
    est = DA3ReconEstimator(cfg)
    depths = est.estimate_scene(keyframes)
    cache.save_depth(args.cache_root, scene_name, depths)

    # Quick read-out: DA3's focal vs the true focal (the mismatch the upgrade fixes).
    f_da3 = np.mean([d.intrinsic[0, 0] for d in depths])
    f_true = np.mean([kf.calib.intrinsic[0, 0] for kf in keyframes])
    print(f"recon: saved {len(depths)} frames with pose+K  "
          f"(mean f_da3={f_da3:.0f} vs f_true={f_true:.0f}, {100 * (f_da3 / f_true - 1):+.0f}%)")

    if args.preview is not None:
        print("writing previews...")
        _preview(args.preview, keyframes, depths)
    print(f"\ncache -> {args.cache_root / scene_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
