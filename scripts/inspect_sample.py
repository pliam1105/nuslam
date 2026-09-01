#!/usr/bin/env python3
"""Smoke-test the data plumbing and verify calibration on one keyframe.

    python scripts/inspect_sample.py --dataroot data/nuscenes --scene scene-0061 \
        --index 0 --out out/calib_check.png

Prints keyframe count, intrinsics and per-frame GT pose deltas, then (with --out)
projects the keyframe's ground-truth 3D annotation boxes into the CAM_FRONT image
using the calibration extracted by ``nuslam.data`` + ``nuslam.transforms``. If the
box wireframes land on the objects, the intrinsics and the sensor->ego->global
chain are correct -- run this before trusting anything downstream.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.transforms import SE3  # noqa: E402


def project(K: np.ndarray, cam_from_global: SE3, pts_global: np.ndarray):
    """Project (N,3) global points to pixels; returns (pixels (M,2), mask front)."""
    p_cam = cam_from_global.apply(pts_global)  # (N, 3)
    front = p_cam[:, 2] > 0.1
    uv = (K @ p_cam[front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv, front


# 12 edges of a nuScenes box, indexing its 8 corners (Box.corners() order).
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None, help="scene name/token (default: first)")
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--index", type=int, default=0, help="keyframe index within scene")
    p.add_argument("--out", type=Path, help="write the calibration-overlay PNG here")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera, verbose=True)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene)
    print(f"\nscene {scene!r}: {len(keyframes)} keyframes on {args.camera}")

    k0 = keyframes[0].calib
    print(f"intrinsic  fx={k0.fx:.1f} fy={k0.fy:.1f} cx={k0.cx:.1f} cy={k0.cy:.1f} "
          f"({k0.width}x{k0.height})")
    print(f"sensor->ego  {k0.sensor2ego}")

    gt = source.gt_trajectory(keyframes)
    steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
    print(f"GT path: total {steps.sum():.1f} m, step {steps.mean():.2f}+-{steps.std():.2f} m")

    kf = keyframes[args.index % len(keyframes)]
    print(f"\nkeyframe {kf.frame_index}  token={kf.token}  t={kf.timestamp_us}")

    # boxes in the GLOBAL frame from the devkit, projected with our own calib
    sample = source.nusc.get("sample", kf.token)
    sd_token = sample["data"][args.camera]
    boxes = source.nusc.get_boxes(sd_token)  # global frame
    cam_from_global = kf.calib.sensor2ego.inverse() @ kf.ego2global_gt.inverse()
    print(f"annotation boxes in view check: {len(boxes)} boxes in scene")

    if args.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        img = kf.image()
        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.imshow(img)
        ax.set_axis_off()
        drawn = 0
        for box in boxes:
            corners = box.corners().T  # (8, 3) global
            p_cam = cam_from_global.apply(corners)
            if np.all(p_cam[:, 2] <= 0.1):
                continue
            uv = (kf.calib.intrinsic @ p_cam.T).T
            uv = uv[:, :2] / uv[:, 2:3]
            if np.all(p_cam[:, 2] > 0.1):  # draw only fully-in-front boxes cleanly
                for a, b in _EDGES:
                    ax.plot([uv[a, 0], uv[b, 0]], [uv[a, 1], uv[b, 1]], color="lime", lw=1.2)
                drawn += 1
        ax.set_xlim(0, kf.calib.width); ax.set_ylim(kf.calib.height, 0)
        ax.set_title(f"{scene} kf{kf.frame_index} — {drawn} GT boxes projected via our calib", fontsize=10)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"\nwrote {args.out}  ({drawn} boxes drawn) — verify wireframes sit on objects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
