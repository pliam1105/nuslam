"""Stage 3 launchpad -- prepares everything the Gaussian ladder consumes, then hands
off to the author's 3DGS at a single seam.

This is PLUMBING ONLY. It loads the scene, the cached metric upgrade (poses + true
intrinsics) and the cached DA3 depth, resolves the global scale from GPS (Route B,
GT-free) so the seed cloud is in metres, builds a voxel-downsampled metric init
cloud, splits held-out views, and sets up the Rerun overlays (trajectory + depth
cloud + camera frustums). Where it says SEAM, the author writes the 3DGS: the
Gaussian representation, the gsplat rasterizer calls, the training loop, the pose
retraction, and the losses (photometric + ground/wheel anchor). The eval + per-rung
metric hooks after the seam are ready for that code to call.

    python scripts/run_gs.py --scene scene-0061 --rerun            # prep + overlays
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import (  # noqa: E402
    NuScenesMonoSource, gps_positions_at, lidar_points_global, load_proprio_streams,
)
from nuslam.eval import evaluate_render, holdout_indices, record_rung  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.pointcloud import voxel_downsample  # noqa: E402
from nuslam.recon import metric_point_cloud, resolve_scale_gps  # noqa: E402
from nuslam.viz import rerun_logging as rrlog  # noqa: E402

# Reconstruction lives in camera-0's frame (RDF); rotate into a Z-up world for viz.
R_VIZ = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
T_VIZ = np.eye(4); T_VIZ[:3, :3] = R_VIZ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataroot", type=Path, default=Path("data/nuscenes"))
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene", default=None)
    p.add_argument("--camera", default="CAM_FRONT")
    p.add_argument("--cache-root", type=Path, default=Path("out/frontend_cache"))
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=4, help="depth back-projection pixel stride")
    p.add_argument("--voxel", type=float, default=0.1, help="init-cloud voxel size (metres)")
    p.add_argument("--holdout-every", type=int, default=8, help="hold out every k-th view for novel-view PSNR")
    p.add_argument("--no-scale", action="store_true", help="skip Route-B metric scaling (leave up-to-scale)")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--save", type=Path, default=None)
    p.add_argument("--point-size", type=float, default=0.03, help="point radius (init cloud + lidar)")
    p.add_argument("--eval-gt", action="store_true",
                   help="ORACLE overlay: also draw GT trajectory + lidar (and the GPS track), "
                        "mapped into the reconstruction frame via the Route-B Sim(3). Needs GPS.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name

    mu = cache.load_metric_upgrade(args.cache_root, scene_name)
    depth = cache.load_depth(args.cache_root, scene_name)
    if mu is None or depth is None:
        print(f"need cached metric upgrade + depth under {args.cache_root}/{scene_name} "
              "-- run run_recon.py then run_metric_upgrade.py --save-cache first")
        return 1
    frames = [(kf, depth[kf.token]) for kf in keyframes if kf.token in depth]
    world_from_cam = mu.world_from_cam            # (N,4,4) camera->world, metric up to scale
    K_true = mu.K_true

    # ---- metric scale from GPS (Route B, GT-free) so the seed cloud is in metres ----
    # Keep the full result: res.T (recon -> map Sim(3)) also places the GT/GPS/lidar
    # reference overlay back into the reconstruction frame below.
    scale, res, gps_xy, gps_valid = 1.0, None, None, None
    if not args.no_scale:
        streams = load_proprio_streams(args.dataroot, scene_name, version=args.version)
        if streams.gps:
            gps_xy, gps_valid = gps_positions_at(streams, [kf.timestamp_us for kf, _ in frames])
            s2e = frames[0][0].calib.sensor2ego.matrix()
            try:
                res = resolve_scale_gps(world_from_cam, gps_xy, s2e, gps_valid)
                scale = float(res.scale)
                print(f"Route-B metric scale = {scale:.4f} (recon units -> metres)")
            except NotImplementedError:
                print("[scale] resolve_scale_gps not implemented -- staying up-to-scale (scale=1)")
        else:
            print("[scale] no GPS stream -- staying up-to-scale (scale=1)")

    poses_m = world_from_cam.copy()
    poses_m[:, :3, 3] *= scale                     # metric camera->world

    # ---- init cloud: back-project depth (metres), voxel-downsample PER KEYFRAME, then
    #      concatenate and downsample the merged cloud once more. Thinning each frame
    #      before the merge keeps the concat + final pass cheap. ----
    raw_xyz, raw_rgb, ds_xyz, ds_rgb = [], [], [], []
    voxel_ok = True
    for i, (kf, dm) in enumerate(frames):
        pts, cols = metric_point_cloud(dm.depth, kf.image(), K_true, world_from_cam[i],
                                       stride=args.stride, conf=dm.conf, sky=dm.sky)
        pts = pts * scale
        raw_xyz.append(pts); raw_rgb.append(cols)
        if voxel_ok:
            try:
                p, c = voxel_downsample(pts, cols, voxel_size=args.voxel)  # per-keyframe
                ds_xyz.append(p); ds_rgb.append(c)
            except NotImplementedError:
                voxel_ok = False
    n_raw = sum(len(p) for p in raw_xyz)
    if voxel_ok:
        xyz, rgb = voxel_downsample(np.concatenate(ds_xyz), np.concatenate(ds_rgb),
                                    voxel_size=args.voxel)               # merged pass
        print(f"init cloud: {n_raw} raw -> {sum(len(p) for p in ds_xyz)} (per-frame) "
              f"-> {len(xyz)} merged points (voxel {args.voxel} m)")
    else:
        xyz = np.concatenate(raw_xyz); rgb = np.concatenate(raw_rgb)
        print(f"[init cloud] voxel_downsample not implemented -- using {n_raw} raw points "
              f"(implement nuslam.pointcloud.voxel_downsample to thin)")

    # ---- held-out split for novel-view PSNR ----
    train_idx, test_idx = holdout_indices(len(frames), every=args.holdout_every)
    print(f"{len(frames)} frames: {len(train_idx)} train / {len(test_idx)} held-out for novel-view PSNR")

    # ---- Rerun overlays: init cloud + trajectory + camera frustums (upright frame) ----
    if args.rerun or args.save is not None:
        rrlog.init(f"nuslam-gs-{scene_name}", spawn=args.rerun, save=args.save)
        rrlog.log_points("init/cloud", xyz @ R_VIZ.T, colors=rgb, radii=args.point_size, static=True)
        rrlog.log_trajectory("traj/est", poses_m[:, :3, 3] @ R_VIZ.T, color=(50, 120, 240), static=True)
        for i, (kf, _) in enumerate(frames):
            rrlog.log_estimated_camera(f"est/{i:03d}", T_VIZ @ poses_m[i], K_true,
                                       kf.calib.width, kf.calib.height, channel=args.camera, static=True)

        # ---- reference overlays: GPS track, and (oracle) GT trajectory + lidar ----
        # These live in the nuScenes map frame; map them into this reconstruction frame
        # with the Route-B Sim(3) inverse: map M -> recon P = (M - t) Rs, then R_VIZ.
        if res is not None:
            Rs = res.T[:3, :3] / res.scale        # orthonormal (scale un-folded)
            t_map = res.T[:3, 3]
            to_view = lambda M: ((np.asarray(M, float) - t_map) @ Rs) @ R_VIZ.T
            if gps_xy is not None and gps_valid.any():  # GPS reference track (the fit target)
                gps3 = np.column_stack([gps_xy[gps_valid], np.zeros(int(gps_valid.sum()))])
                rrlog.log_trajectory("ref/gps", to_view(gps3), color=(230, 150, 40), radius=0.3, static=True)
            if args.eval_gt:  # ORACLE: GT camera trajectory + lidar, for comparison only
                s2e_m = frames[0][0].calib.sensor2ego.matrix()
                gt_cam = np.stack([kf.ego2global_gt.matrix() @ s2e_m for kf, _ in frames])[:, :3, 3]
                rrlog.log_trajectory("ref/gt", to_view(gt_cam), color=(60, 200, 255), radius=0.3, static=True)
                for i, (kf, _) in enumerate(frames):
                    lx = lidar_points_global(source, kf)
                    if len(lx):
                        rrlog.log_points(f"ref/lidar/{i:03d}", to_view(lx),
                                         colors=(190, 190, 190), radii=args.point_size, static=True)
        elif args.eval_gt:
            print("[eval-gt] no Route-B alignment (needs GPS) -- skipping GT/GPS/lidar overlay")

        print("logged init cloud + trajectory + frustums"
              + ("  [+ GPS/GT/lidar reference]" if res is not None else "")
              + " to Rerun" + (f" -> {args.save}" if args.save else ""))

    # ===================================================================================
    # SEAM -- the author writes the 3DGS here (core substance, CLAUDE.md section 3).
    #
    # Prepared inputs above, ready to consume:
    #   xyz (M,3) float32, rgb (M,3) uint8  -- metric init cloud (seed Gaussian means/colors)
    #   poses_m (N,4,4)                      -- metric camera->world  (viewmats = inv(poses_m))
    #   K_true (3,3)                         -- true intrinsics (per-view Ks = tile over N)
    #   frames[i][0].image()                -- GT image for view i
    #   train_idx / test_idx                -- optimize on train, score novel-view PSNR on test
    #
    # Write: the Gaussian representation + reparametrization, the gsplat rasterization(...)
    # calls (render_mode="RGB+ED" for the ground anchor), the training loop, the se3 pose
    # retraction on viewmats, and the losses (photometric + ground/wheel anchor). Then, to
    # visualize alongside the overlays above and score each rung, call the wired hooks:
    #
    #   rrlog.log_gaussians("gs/means", means, scales, quats_wxyz, colors_rgba)  # 3D splats
    #   rrlog.log_image("render/est", rendered);  rrlog.log_image("render/gt", gt)  # 2D compare
    #   e = evaluate_render(rendered, gt);  # PSNR/SSIM/L1 on a held-out view
    #   record_rung(args.cache_root / scene_name / "rungs.json", "3.1",
    #               {"psnr": e.psnr, "ssim": e.ssim, ...})  # per-rung metric log
    # ===================================================================================
    print("\nStage-3 inputs prepared. Write the 3DGS at the SEAM in this script "
          "(scripts/run_gs.py) -- the viz + eval + rung-log hooks are ready to call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
