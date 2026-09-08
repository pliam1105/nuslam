#!/usr/bin/env python3
"""Drive the metric upgrade from a cached DA3-Base reconstruction.

    python scripts/run_metric_upgrade.py --scene scene-0061

Loads the DA3-Base cache (depth + per-frame pose + estimated ``K_da3``), builds
the normalized projective cameras

    P~_i = K_true^{-1} K_da3_i [R_i | t_i],

solves for the rectifier ``H`` (``nuslam.recon.metric_upgrade``) and recovers a
reconstruction that is **metric up to a single global scale** -- correct camera
angles / shape, with one unresolved scalar (the monocular scale, later fixed by
the ground/wheel anchor). It then:

  * visualizes the metric-up-to-scale reconstruction in Rerun -- recovered poses
    (frusta with the true intrinsics), the trajectory, and the corrected-intrinsics
    depth cloud (``--rerun`` / ``--save``).

The single global scale is NOT resolved here and is NEVER taken from GT (that would
be cheating). It is resolved downstream, without GT, by either: road/ground +
wheel-contact on the unprojected point cloud, or a Sim(3) fit of these poses to a
GPS(+IMU) trajectory. ``--eval-gt`` (oracle only) Sim(3)-aligns to nuScenes GT to
print ATE and the scale those legitimate methods should reproduce -- GT is a
scoring oracle, never a scale source.

It also reports the per-frame focal spread of ``K_da3`` (drift is expected and
harmless -- each ``K_da3_i`` is known and folded into its own ``P~_i``); the live
test of the single-homography premise is the DAQ residual / recovered-K
anisotropy, not the focal spread.

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

from nuslam.data import NuScenesMonoSource, lidar_points_global  # noqa: E402
from nuslam.eval import align_trajectories, evaluate, evaluate_daq  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.recon import (  # noqa: E402
    build_daq_system,
    decompose_metric_camera,
    metric_cameras,
    metric_depth,
    metric_point_cloud,
    normalized_projective_cameras,
    plane_at_infinity,
    rectifying_homography,
    solve_daq,
)
from nuslam.types import MetricUpgrade  # noqa: E402
from nuslam.viz import rerun_logging as rrlog  # noqa: E402

# The reconstruction lives in camera-0's frame (RDF: x right, y down, z forward),
# so its "up" is -y and it renders tilted/sideways in a Z-up viewer. R_VIZ rotates
# that frame to a Z-up, upright, horizontal world for viewing only:
#   forward (cam +z) -> world +X,  up (cam -y) -> world +Z,  right (cam +x) -> world -Y.
R_VIZ = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
T_VIZ = np.eye(4)
T_VIZ[:3, :3] = R_VIZ


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
    p.add_argument("--point-size", type=float, default=0.05,
                   help="metric depth-cloud point radius; >0 = metres, <0 = screen pixels")
    p.add_argument("--lidar-size", type=float, default=0.03,
                   help="lidar point radius (--eval-gt overlay); metres if >0")
    p.add_argument("--frustum-frac", type=float, default=0.2,
                   help="camera frustum length as a fraction of that frame's median cloud depth")
    p.add_argument("--eval-gt", action="store_true",
                   help="ORACLE ONLY: Sim(3)-align to nuScenes GT and print ATE/scale for scoring. "
                        "GT is never used to resolve scale (that would be cheating); scale is "
                        "resolved by road/ground or GPS(+IMU) downstream. Off by default.")
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

    # Build normalized projective cameras P~_i (author seam), batched over frames.
    K_da3 = np.stack([dm.intrinsic for _, dm in frames])            # (N,3,3) per-frame
    R = np.stack([dm.extrinsic[:3, :3] for _, dm in frames])        # (N,3,3)
    t = np.stack([dm.extrinsic[:3, 3] for _, dm in frames])         # (N,3)
    cams, err = stage("normalized_projective_cameras",
                      lambda: normalized_projective_cameras(K_true, K_da3, R, t))
    if err:
        return 0

    # Solve the DAQ via the sub-functions so A and Omega* are exposed for diagnostics.
    A = build_daq_system(cams)
    omega = solve_daq(A)
    p_inf = plane_at_infinity(omega)
    H = rectifying_homography(p_inf, M)
    print(f"H =\n{np.array2string(H, precision=4, suppress_small=True)}")
    print(f"plane at infinity (v, s) = {H[3, :]}")

    Pm = metric_cameras(cams, H)

    # Recover camera->world poses + K for ALL frames (not just 3).
    poses = [_cam_to_world(P) for P in Pm]
    world_from_cam = np.stack([T for T, _ in poses])
    K_recovered = np.stack([K for _, K in poses])

    # ---- DAQ solve diagnostics: how well / how consistent the fit was ----
    diag = evaluate_daq(cams, A, omega, Pm)
    print(f"\n{diag}")
    Kn = K_recovered / K_recovered[:, 2:3, 2:3]
    aniso = np.abs(Kn[:, 0, 0] - Kn[:, 1, 1]) / np.abs(Kn[:, 0, 0])
    skew = np.abs(Kn[:, 0, 1]) / np.abs(Kn[:, 0, 0])
    print(f"  recovered-K aniso  mean={aniso.mean():.3f} median={np.median(aniso):.3f} max={aniso.max():.3f}")
    print(f"  recovered-K skew   mean={skew.mean():.3f} median={np.median(skew):.3f} max={skew.max():.3f}")

    # ---- Cache the Stage-1 result (no re-solve, no DA3 rerun for the next stage) ----
    mu = MetricUpgrade(
        tokens=[kf.token for kf, _ in frames], H=H, omega_star=omega, plane_at_infinity=p_inf,
        world_from_cam=world_from_cam, metric_cameras=np.asarray(Pm), K_recovered=K_recovered,
        K_true=K_true, diagnostics=diag.as_dict(),
    )
    cache.save_metric_upgrade(args.cache_root, scene_name, mu)
    print(f"cached metric upgrade -> {args.cache_root / scene_name / 'metric_upgrade.npz'}")

    # The reconstruction is now metric UP TO ONE GLOBAL SCALE. That scalar is NOT
    # resolved here and is NEVER taken from GT (that would be cheating). It is
    # resolved downstream, without GT, by either: (a) road/ground + wheel-contact on
    # the unprojected point cloud, or (b) a Sim(3) fit of these poses to a GPS(+IMU)
    # trajectory. See the module docstring.
    est = np.stack(world_from_cam)  # (N,4,4) camera->world, metric up to a global scale
    print("\nreconstruction is metric up to ONE global scale (unresolved here; not taken from GT).")
    print("resolve downstream: road/ground on the unprojected cloud, or Sim(3)-fit these poses to GPS(+IMU).")

    est_aligned, gt, T_sim = None, None, None
    if args.eval_gt:
        # ORACLE ONLY -- scoring, not a scale source. Sim(3)-align to GT to read ATE
        # and the scale the legitimate (road / GPS+IMU) methods should reproduce.
        gt = np.stack([kf.ego2global_gt.matrix() for kf, _ in frames])
        err3 = evaluate(est, gt, align="sim3")
        est_aligned, _, T_sim = align_trajectories(est[:, :3, 3], gt[:, :3, 3], mode="sim3")
        print(f"\n[oracle · GT · scoring only] Sim(3) fit: {err3}")
        print(f"  -> scale {err3.scale:.4f} is the value the road/GPS methods should hit; ATE = residual after removing it")

    # ---- Rerun: horizontal, upright metric-up-to-scale reconstruction ----
    # Everything is rotated by R_VIZ into a Z-up world before logging, so the road
    # sits horizontal and the scene stands upright in the viewer.
    if args.rerun or args.save is not None:
        rrlog.init(f"nuslam-metric-{scene_name}", spawn=args.rerun, save=args.save)
        t0 = frames[0][0].timestamp_us
        rrlog.log_trajectory("traj/est", est[:, :3, 3] @ R_VIZ.T, color=(80, 200, 120))
        if args.eval_gt:  # oracle overlay, in GT's own Z-up frame: GT + Sim(3)-aligned est
            rrlog.log_trajectory("traj/gt_oracle", gt[:, :3, 3], color=(120, 120, 120))
            rrlog.log_trajectory("traj/est_gtaligned", est_aligned, color=(200, 160, 60))
        for i, (kf, dm) in enumerate(frames):
            rrlog.set_time(kf.frame_index, kf.timestamp_us, t0)
            T = world_from_cam[i]
            dmetric, e = stage("metric_depth", lambda kf=kf, dm=dm, T=T: metric_depth(
                dm.depth, dm.intrinsic, dm.extrinsic, H, np.linalg.inv(T)))
            # frustum length ~ a fraction of this frame's cloud depth, so it matches
            # the points instead of dwarfing them (the cloud scale is arbitrary here).
            valid = dmetric[np.isfinite(dmetric) & (dmetric > 0)] if e is None else np.array([])
            plane_dist = float(args.frustum_frac * np.median(valid)) if valid.size else None
            rrlog.log_estimated_camera(f"est/{i:03d}", T_VIZ @ T, kf.calib.intrinsic,
                                       kf.calib.width, kf.calib.height, channel=args.camera,
                                       image_plane_distance=plane_dist)
            if e:
                continue
            pts, cols = metric_point_cloud(dmetric, kf.image(), kf.calib.intrinsic, T,
                                           stride=args.stride, conf=dm.conf, sky=dm.sky)
            rrlog.log_points(f"depth/{i:03d}", pts @ R_VIZ.T, colors=cols, radii=args.point_size)
            if args.eval_gt:  # aligned cloud + lidar in GT frame -> a metric reference to eyeball
                rrlog.log_points(f"oracle/depth/{i:03d}", pts @ T_sim[:3, :3].T + T_sim[:3, 3],
                                 colors=cols, radii=args.point_size)
                lidar_xyz = lidar_points_global(source, kf)
                if len(lidar_xyz):
                    rrlog.log_points(f"oracle/lidar/{i:03d}", lidar_xyz,
                                     colors=(190, 190, 190), radii=args.lidar_size)
        print("logged metric-up-to-scale reconstruction (poses + cloud) to Rerun"
              + ("  [+ GT/lidar oracle overlay]" if args.eval_gt else "")
              + (f" -> {args.save}" if args.save else ""))
    print("\nmetric upgrade complete (reconstruction is metric up to a global scale).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
