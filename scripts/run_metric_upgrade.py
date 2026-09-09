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

from nuslam.data import (  # noqa: E402
    NuScenesMonoSource,
    gps_positions_at,
    lidar_points_global,
    load_proprio_streams,
)
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
    resolve_scale_gps,
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
    p.add_argument("--m-weight", type=float, default=1.0,
                   help="soft-constrain Omega*[:3,:3] to the known reference intrinsics M_0 in "
                        "the DAQ solve (0 = off; default 1.0). Corrects the forward-motion "
                        "degeneracy that under-constrains the plane at infinity -- on scene-0061 "
                        "it drops the conic-block vs M_0 deviation from 0.33 to 0.04 and roughly "
                        "halves oracle ATE. Pass 0 to reproduce the unconstrained solve.")
    p.add_argument("--eval-gt", action="store_true",
                   help="ORACLE ONLY: Sim(3)-align to nuScenes GT and print ATE/scale for scoring. "
                        "GT is never used to resolve scale (that would be cheating); scale is "
                        "resolved by road/ground or GPS(+IMU) downstream. Off by default.")
    p.add_argument("--route-b", action="store_true",
                   help="Resolve the global metric scale from GPS (nuScenes-CAN 'pose') via the "
                        "lever-arm Umeyama fit (nuslam.recon.resolve_scale_gps). GT-free; needs the "
                        "CAN expansion. With --eval-gt, prints Route-B vs oracle scale side by side.")
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
    # --m-weight softly pins Omega*[:3,:3] to the known reference intrinsics M_0.
    try:
        A = build_daq_system(cams, m_prior=(M if args.m_weight > 0 else None),
                             m_weight=args.m_weight)
    except NotImplementedError as exc:
        print(f"[pending] M-prior rows (daq_m_prior_rows): {exc}")
        print("          re-run without --m-weight, or implement the seam.")
        return 0
    omega = solve_daq(A)
    p_inf = plane_at_infinity(omega)
    H = rectifying_homography(p_inf, M)
    if args.m_weight > 0:
        print(f"DAQ solved with M_0 soft prior, weight={args.m_weight:g}")
    print(f"H =\n{np.array2string(H, precision=4, suppress_small=True)}")
    print(f"plane at infinity (v, s) = {H[3, :]}")

    # How far the (unconstrained) fit's conic block sits from the known W_0 = M_0^-1 M_0^-T.
    # A large gap here is exactly what the M prior corrects.
    W0 = np.linalg.inv(M) @ np.linalg.inv(M).T
    blk = omega[:3, :3]
    W0n, blkn = W0 / np.linalg.norm(W0), blk / np.linalg.norm(blk)
    blkn *= np.sign(np.vdot(W0n, blkn))  # match sign (Omega* is up to sign)
    print(f"conic-block vs known W_0: relative deviation "
          f"{np.linalg.norm(blkn - W0n) / np.linalg.norm(W0n):.3f}  (0 = fit agrees with M_0)")

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
    diag_out = diag.as_dict()
    diag_out["m_weight"] = float(args.m_weight)
    mu = MetricUpgrade(
        tokens=[kf.token for kf, _ in frames], H=H, omega_star=omega, plane_at_infinity=p_inf,
        world_from_cam=world_from_cam, metric_cameras=np.asarray(Pm), K_recovered=K_recovered,
        K_true=K_true, diagnostics=diag_out,
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
        # Compare like with like: the est poses are the CAMERA, so bring GT into the
        # camera frame via the known sensor->ego extrinsic (the ego origin sits ~1.51 m
        # below / 1.70 m behind CAM_FRONT). Aligning recon camera centres to GT *ego*
        # centres would bake that offset into the Sim(3) and push the aligned cloud off
        # the lidar (which is logged in the true global frame).
        s2e = frames[0][0].calib.sensor2ego.matrix()  # static per channel: camera->ego
        gt = np.stack([kf.ego2global_gt.matrix() @ s2e for kf, _ in frames])  # GT camera->global
        err3 = evaluate(est, gt, align="sim3")
        est_aligned, _, T_sim = align_trajectories(est[:, :3, 3], gt[:, :3, 3], mode="sim3")
        print(f"\n[oracle · GT · scoring only] Sim(3) fit: {err3}")
        print(f"  -> scale {err3.scale:.4f} is the value the road/GPS methods should hit; ATE = residual after removing it")

    # ---- Route B: resolve the global scale from GPS (GT-free) ----
    res, gps_ref_xy, gps_valid = None, None, None
    if args.route_b:
        streams = load_proprio_streams(args.dataroot, scene_name, version=args.version)
        if not streams.gps:
            print("\n[route B] no GPS stream (nuScenes-CAN expansion missing) -- skipping.")
        else:
            gps_ref_xy, gps_valid = gps_positions_at(streams, [kf.timestamp_us for kf, _ in frames])
            s2e_b = frames[0][0].calib.sensor2ego.matrix()
            try:
                res = resolve_scale_gps(est, gps_ref_xy, s2e_b, gps_valid)
                print(f"\n[route B · GPS · no GT] scale = {res.scale:.4f}  "
                      f"(horizontal residual {res.residual:.3f} m, {res.num_iters} iters, "
                      f"{int(gps_valid.sum())}/{len(gps_valid)} frames)")
                if args.eval_gt:
                    rel = abs(res.scale - err3.scale) / err3.scale
                    print(f"  vs oracle scale {err3.scale:.4f}: relative difference {rel:.1%}  "
                          f"(agreement validates the GPS route without touching GT)")
            except NotImplementedError as exc:
                print(f"\n[route B] pending: {exc}")
                print("          implement resolve_scale_gps (nuslam.recon.scale) -- see the Umeyama handout.")

    # ---- Rerun: horizontal, upright metric-up-to-scale reconstruction ----
    # Everything is rotated by R_VIZ into a Z-up world before logging, so the road
    # sits horizontal and the scene stands upright in the viewer.
    if args.rerun or args.save is not None:
        rrlog.init(f"nuslam-metric-{scene_name}", spawn=args.rerun, save=args.save)
        t0 = frames[0][0].timestamp_us
        # Whole-scene paths -> static, so they show at every time cursor (logged before
        # the loop's first set_time, they'd otherwise be absent from the frame timeline).
        rrlog.log_trajectory("traj/est", est[:, :3, 3] @ R_VIZ.T, color=(80, 200, 120), static=True)
        # Map-frame overlay: place the reconstruction by the GPS Sim(3) (Route B) when it
        # ran, else by the GT Sim(3). Whichever is used, GT (cyan) and the GPS reference
        # track (green) are drawn alongside for comparison. This frame is already Z-up
        # (GT/GPS map frame), so no R_VIZ here.
        T_ref = res.T if res is not None else (T_sim if args.eval_gt else None)
        ref_scale = res.scale if res is not None else (err3.scale if args.eval_gt else None)
        map_origin = np.zeros(3)
        if T_ref is not None:
            est_ref = est[:, :3, 3] @ T_ref[:3, :3].T + T_ref[:3, 3]   # est cameras in the map
            # The map frame sits ~1 km from its origin; recenter the whole overlay on the
            # camera centroid so the view lands on the cameras, not empty space far away.
            # A pure display shift (subtracted from every map-frame entity) -- geometry unchanged.
            map_origin = est_ref.mean(0)
            est_ref = est_ref - map_origin
            # metric map frame is tens of m -- scale the line radius to the track extent
            # so a 0.1 m tube is not sub-pixel.
            diag = float(np.linalg.norm(est_ref.max(0) - est_ref.min(0)))
            traj_r = max(0.08, 0.002 * diag)
            rrlog.log_trajectory("traj/est_aligned", est_ref, color=(200, 160, 60), radius=traj_r, static=True)
            if args.eval_gt:  # GT camera track, for comparison
                rrlog.log_trajectory("traj/gt", gt[:, :3, 3] - map_origin, color=(60, 200, 255), radius=traj_r, static=True)
            if res is not None:  # the GPS reference track the fit was aligned to, on z=0
                gps3 = np.column_stack([gps_ref_xy[gps_valid], np.zeros(int(gps_valid.sum()))])
                rrlog.log_trajectory("traj/gps", gps3 - map_origin, color=(80, 220, 120), radius=traj_r, static=True)
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
            if T_ref is not None:  # same camera aligned into the map frame -> pyramids on the overlay
                Rs = T_ref[:3, :3] / ref_scale       # un-fold scale -> orthonormal rotation
                Ta = np.eye(4)
                Ta[:3, :3] = Rs @ T[:3, :3]
                Ta[:3, 3] = T_ref[:3, :3] @ T[:3, 3] + T_ref[:3, 3] - map_origin
                rrlog.log_estimated_camera(f"aligned/est/{i:03d}", Ta, kf.calib.intrinsic,
                                           kf.calib.width, kf.calib.height, channel=args.camera,
                                           image_plane_distance=(plane_dist * ref_scale if plane_dist else None),
                                           color=(230, 150, 40))
            if e:
                continue
            pts, cols = metric_point_cloud(dmetric, kf.image(), kf.calib.intrinsic, T,
                                           stride=args.stride, conf=dm.conf, sky=dm.sky)
            rrlog.log_points(f"depth/{i:03d}", pts @ R_VIZ.T, colors=cols, radii=args.point_size)
            if T_ref is not None:  # aligned cloud + lidar in the map frame -> metric reference to eyeball
                rrlog.log_points(f"aligned/depth/{i:03d}", pts @ T_ref[:3, :3].T + T_ref[:3, 3] - map_origin,
                                 colors=cols, radii=args.point_size)
                lidar_xyz = lidar_points_global(source, kf)
                if len(lidar_xyz):
                    rrlog.log_points(f"aligned/lidar/{i:03d}", lidar_xyz - map_origin,
                                     colors=(190, 190, 190), radii=args.lidar_size)
        _ref_name = "GPS(route B)" if res is not None else ("GT" if args.eval_gt else None)
        print("logged metric-up-to-scale reconstruction (poses + cloud) to Rerun"
              + (f"  [+ {_ref_name}-aligned overlay + lidar]" if _ref_name else "")
              + (f" -> {args.save}" if args.save else ""))
    print("\nmetric upgrade complete (reconstruction is metric up to a global scale).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
