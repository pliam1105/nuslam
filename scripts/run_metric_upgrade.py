#!/usr/bin/env python3
"""Drive the metric upgrade from a cached DA3-Base reconstruction.

    python scripts/run_metric_upgrade.py --scene scene-0061

Loads the DA3-Base cache (depth + per-frame pose + estimated ``K_da3``), builds
the normalized projective cameras

    P~_i = K_true^{-1} K_da3_i [R_i | t_i],

solves for the rectifier ``H`` (``nuslam.recon.metric_upgrade``), recovers the
metric cameras, and wires the results downstream:

  * Sim(3) evaluation of the recovered trajectory against nuScenes GT (the scale
    read-out says whether it is metric);
  * Rerun logging of the recovered poses (frusta with the true intrinsics), the
    estimated vs GT trajectories, and the metric depth cloud (``--rerun`` / ``--save``).

It also reports the per-frame focal spread of ``K_da3`` -- the single-homography
premise holds only if a single mismatch ``M`` relates DA3 to the truth, so a large
spread predicts a poor fit.

The solve itself is core (author-written); until those seams are implemented this
prints how far it got and what is pending, without failing. Run
``scripts/run_recon.py`` first to populate the cache.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource  # noqa: E402
from nuslam.eval import evaluate  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.recon import (  # noqa: E402
    decompose_metric_camera,
    metric_cameras,
    metric_depth,
    metric_point_cloud,
    metric_upgrade,
    normalized_projective_camera,
)
from nuslam.viz import rerun_logging as rrlog  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None)
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--rerun", action="store_true", help="spawn a Rerun viewer with the result")
    p.add_argument("--save", type=Path, default=None, help="write a .rrd instead of spawning")
    p.add_argument("--stride", type=int, default=8, help="depth cloud pixel stride")
    return p.parse_args()


def _cam_to_world(P_metric: np.ndarray) -> np.ndarray:
    """Recovered [R*|t*] (world->camera, up to scale) -> a 4x4 camera->world pose.

    Decompose to (K, R*, t*); the camera->world pose is R*^T for rotation and the
    centre C = -R*^T t*. Pose bookkeeping only -- the decomposition is the core call.
    """
    K, R, t = decompose_metric_camera(P_metric)
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ t
    return T, K


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name

    depth = cache.load_depth(args.cache_root, scene_name)
    if depth is None:
        print(f"no cached reconstruction under {args.cache_root}/{scene_name} -- run run_recon.py first")
        return 1
    frames = [(kf, depth[kf.token]) for kf in keyframes if kf.token in depth]
    if not frames or frames[0][1].extrinsic is None:
        print("cache has no DA3-Base pose/intrinsics -- run run_recon.py (not run_depth.py) first")
        return 1

    # Per-frame mismatch M_i = K_true^{-1} K_da3_i. These enter the DAQ per camera
    # (each folded into its own P~_i) and NEED NOT be equal -- DA3's per-frame focal
    # is expected to drift. The block form H = [[M,0],[v^T,s]] uses one M: the
    # REFERENCE (first) camera's, in the gauge where camera 0 is canonical (the
    # other cameras' K_da3 live in their P~_i). Feasibility hinges on DA3 being
    # projectively self-consistent -- read off the DAQ residual / Sim(3) below, not
    # the focal spread.
    K_true = frames[0][0].calib.intrinsic
    Ms = [np.linalg.inv(K_true) @ dm.intrinsic for _, dm in frames]
    f_da3 = np.array([dm.intrinsic[0, 0] for _, dm in frames])
    print(f"scene {scene_name!r}: {len(frames)} frames")
    print(f"K_da3 focal: mean={f_da3.mean():.0f}  std={f_da3.std():.0f}  "
          f"spread={100 * f_da3.std() / f_da3.mean():.1f}%   (drift expected; per-camera known)")
    M = Ms[0]  # reference-camera M for the block form; the M choice is a core design call

    def stage(name, fn):
        try:
            return fn(), None
        except NotImplementedError as exc:
            print(f"[pending] {name}: {exc}")
            return None, exc

    # Build normalized projective cameras P~_i (author seam).
    cams, err = stage("normalized_projective_camera", lambda: np.stack([
        normalized_projective_camera(K_true, dm.intrinsic, dm.extrinsic[:3, :3], dm.extrinsic[:3, 3])
        for _, dm in frames]))
    if err:
        return 0

    H, err = stage("metric_upgrade", lambda: metric_upgrade(cams, M))
    if err:
        return 0
    print(f"H =\n{np.array2string(H, precision=4, suppress_small=True)}")
    print(f"plane at infinity (v, s) = {H[3, :]}")

    Pm, err = stage("metric_cameras", lambda: metric_cameras(cams, H))
    if err:
        return 0

    # Recover camera->world poses (and check K ~ proportional to I -- a free test).
    poses, err = stage("decompose_metric_camera", lambda: [_cam_to_world(P) for P in Pm])
    if err:
        return 0
    world_from_cam = [T for T, _ in poses]
    for i, (_, K) in enumerate(poses[: min(3, len(poses))]):
        Kn = K / K[2, 2]
        aniso = abs(Kn[0, 0] - Kn[1, 1]) / abs(Kn[0, 0])
        print(f"  cam {i}: recovered K diag={np.diag(Kn)}  aniso={aniso:.3f}  (want ~[c,c,1])")

    # ---- Sim(3) evaluation of the recovered trajectory vs nuScenes GT ----
    est = np.stack(world_from_cam)                                  # (N,4,4) camera->world (metric frame)
    gt = np.stack([kf.ego2global_gt.matrix() for kf, _ in frames])  # (N,4,4) ego->global GT
    err3 = evaluate(est, gt, align="sim3")
    print(f"\nSim(3) vs GT: {err3}")
    print(f"  -> recovered scale {err3.scale:.4f} (≈1.0 once metric); a large value = scale not locked")

    # ---- Rerun: recovered poses + trajectories + metric depth cloud ----
    if args.rerun or args.save is not None:
        rrlog.init(f"nuslam-metric-{scene_name}", spawn=args.rerun, save=args.save)
        t0 = frames[0][0].timestamp_us
        rrlog.log_trajectory("traj/gt", gt[:, :3, 3], color=(120, 120, 120))
        rrlog.log_trajectory("traj/est", est[:, :3, 3], color=(80, 200, 120))
        for i, (kf, dm) in enumerate(frames):
            rrlog.set_time(kf.frame_index, kf.timestamp_us, t0)
            T = world_from_cam[i]
            rrlog.log_estimated_camera(f"est/{i:03d}", T, kf.calib.intrinsic,
                                       kf.calib.width, kf.calib.height, channel=args.camera)
            dmetric, e = stage("metric_depth", lambda kf=kf, dm=dm, T=T: metric_depth(
                dm.depth, dm.intrinsic, dm.extrinsic, H, np.linalg.inv(T)))
            if e:
                continue
            pts, cols = metric_point_cloud(dmetric, kf.image(), kf.calib.intrinsic, T,
                                           stride=args.stride, conf=dm.conf, sky=dm.sky)
            rrlog.log_points(f"depth/{i:03d}", pts, colors=cols, radii=0.05)
        print("logged recovered poses + metric cloud to Rerun"
              + (f" -> {args.save}" if args.save else ""))
    print("\nmetric upgrade complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
