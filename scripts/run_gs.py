"""Stage 3 driver -- prepares the inputs the Gaussian reconstruction consumes.

Plumbing. Loads the scene, the cached metric upgrade (poses + intrinsics) and the
cached DA3 depth, resolves the global scale from GPS (Route B, GT-free) so the seed
cloud is in metres, builds a voxel-downsampled metric init cloud, splits held-out
views, and sets up the Rerun overlays (trajectory + depth cloud + camera frustums).
The Gaussian reconstruction is called at the marked handoff -- its representation,
gsplat rasterizer calls, optimization loop, pose retraction, and losses live in
``nuslam.recon.gaussians``, not here. The visualization and evaluation hooks after the
handoff are ready for that code to call.

    python scripts/run_gs.py --scene scene-0061 --rerun            # prep + overlays
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import (  # noqa: E402
    NuScenesMonoSource, gps_positions_at, lidar_points_global, load_proprio_streams,
)
from nuslam.eval import evaluate_render, holdout_indices, record_metrics  # noqa: E402
from nuslam.frontend import RoadSegmenter, Sam3Segmenter, SegConfig, cache  # noqa: E402
from nuslam.pointcloud import voxel_downsample  # noqa: E402
from nuslam.recon import metric_depth, metric_point_cloud, resolve_scale_gps, train_gaussians  # noqa: E402
from nuslam.viz import rerun_logging as rrlog  # noqa: E402

# Reconstruction lives in camera-0's frame (RDF); rotate into a Z-up world for viz.
R_VIZ = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
T_VIZ = np.eye(4); T_VIZ[:3, :3] = R_VIZ

# Per-parameter Adam learning rates, keyed to the ParameterDict; starting values to tune.
LR_FOR = {
    "means": 1.6e-4,
    "quats": 1.0e-3,
    "scales": 5.0e-3,
    "opacities": 5.0e-2,
    "colors": 2.5e-3,
    # camera-pose refinement (--optimize-poses). Read by a SEPARATE pose optimizer, not the
    # MCMC-managed params dict (densification indexes every param by Gaussian id). Tune.
    "pose_quats": 1.0e-3,
    "pose_trans": 1.0e-3,
}


def _sky_keep_mask(means, poses_m, K, frames, sky_masks):
    """Keep-mask (N,) for Gaussians: drop those that project onto sky in a majority of the
    views that see them. ``means`` (N,3) is in the metric world frame of ``poses_m``."""
    means = np.asarray(means, dtype=np.float64)
    n = len(means)
    sky_hits = np.zeros(n); seen = np.zeros(n)
    K = np.asarray(K, dtype=np.float64)
    for i, (kf, _) in enumerate(frames):
        sky = sky_masks.get(kf.token)
        if sky is None:
            continue
        h, w = sky.shape
        vm = np.linalg.inv(poses_m[i])                    # world -> camera
        Xc = means @ vm[:3, :3].T + vm[:3, 3]             # (N,3)
        uv = Xc @ K.T                                     # (N,3)
        u = uv[:, 0] / uv[:, 2]; v = uv[:, 1] / uv[:, 2]
        inb = (Xc[:, 2] > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = np.clip(u, 0, w - 1).astype(int); vi = np.clip(v, 0, h - 1).astype(int)
        sky_hits += sky[vi, ui] & inb
        seen += inb
    frac = np.divide(sky_hits, np.maximum(seen, 1))
    return ~((seen > 0) & (frac > 0.5))                   # keep unless seen and majority-sky


def _range_far_mask(depth_m, K, pose, centres, thresh):
    """(H, W) bool: pixels whose back-projected metric point lies farther than ``thresh`` metres from
    EVERY camera centre. The init cloud back-projects one point per pixel, so each pixel's range to
    the trajectory is known here directly. ``depth_m`` is the per-pixel metric depth (metres), ``pose``
    its camera->world (metric, the training frame), ``centres`` (N, 3) all camera centres. Pixels with
    non-positive depth are never marked far (their point is undefined)."""
    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H, 0:W]
    rays = np.stack([uu, vv, np.ones_like(uu)], -1).astype(np.float64) @ np.linalg.inv(K).T   # (H,W,3)
    world = (rays * depth_m[..., None]) @ pose[:3, :3].T + pose[:3, 3]                          # (H,W,3)
    mind = np.full((H, W), np.inf)
    for c in centres:
        mind = np.minimum(mind, np.linalg.norm(world - c, axis=-1))
    return (mind > thresh) & (depth_m > 0)


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
    p.add_argument("--holdout-offset", type=int, default=0,
                   help="phase of the held-out split: hold out frames where (i-offset) %% every == 0 "
                        "(e.g. --holdout-every 2 --holdout-offset 1 trains 0,2,4 and holds out 1,3)")
    p.add_argument("--no-scale", action="store_true", help="skip Route-B metric scaling (leave up-to-scale)")
    p.add_argument("--train", action="store_true", help="fit the Gaussians (else only prep + overlays)")
    p.add_argument("--num-iters", type=int, default=10000, help="optimization iterations")
    p.add_argument("--structural-lambda", type=float, default=0.2, help="weight of the structural (D-SSIM) term")
    p.add_argument("--log-every", type=int, default=100, help="log progress/renders every N iters (0 = off)")
    p.add_argument("--snapshot-every", type=int, default=500, help="save a GS .npz snapshot every N iters")
    p.add_argument("--from-checkpoint", type=Path, default=None,
                   help="warm-start the Gaussians from a saved gs_*.npz instead of the point cloud")
    p.add_argument("--clip-scales", action="store_true",
                   help="warm start only: clip oversized checkpoint scales to the local kNN spacing")
    p.add_argument("--optimize-poses", action="store_true",
                   help="refine camera poses jointly through the rasterizer (raw per-view quaternion "
                        "+ translation); off by default, poses frozen at the recovered solution")
    p.add_argument("--gt-poses", action="store_true",
                   help="ORACLE: use nuScenes GT camera poses (mapped into the metric-recon frame via "
                        "the Route-B Sim(3)) as the training cameras, to isolate DA3 pose error. Never "
                        "used for scale; diagnostic only")
    p.add_argument("--colmap-poses", action="store_true",
                   help="use cached COLMAP camera poses (aligned to the nuScenes global frame by the same "
                        "Route-B Umeyama, built by scratchpad/cache_colmap_poses.py -> "
                        "<cache>/<scene>/colmap_poses_global.npz) as the training cameras, subset by token. "
                        "Like --gt-poses but SfM poses instead of the GT oracle; the DA3 metric depth still "
                        "seeds the dense init cloud. Not used for scale")
    p.add_argument("--opacity-reg", type=float, default=0.0,
                   help="MCMC opacity-sparsity weight (L1 on opacities); 0=off, ~0.01 typical")
    p.add_argument("--scale-reg", type=float, default=0.0,
                   help="MCMC covariance weight (L1 on scales); 0=off, ~0.01 typical")
    p.add_argument("--depth-lambda", type=float, default=0.0,
                   help="weight of the expected-depth loss vs the DA3 metric depth; 0=off")
    p.add_argument("--sky-lambda", type=float, default=0.0,
                   help="weight of the sky->black loss (needs --mask-sky); 0=off")
    p.add_argument("--run-name", default=None,
                   help="subdirectory name for this run's logs/checkpoints/renders "
                        "(default: timestamp). Runs never overwrite each other.")
    p.add_argument("--mask-sky", action="store_true",
                   help="segment sky (CLIPSeg 'sky' prompt) and drop sky pixels from the init cloud")
    p.add_argument("--remove-sky-gaussians", action="store_true",
                   help="drop Gaussians that project onto sky in most views (needs sky masks; "
                        "post-processes a warm-started/loaded set)")
    p.add_argument("--segmenter", choices=["sam3", "clipseg"], default="sam3",
                   help="masking backend for --mask-sky/--mask-vehicles: sam3 (facebook/sam3, sharp "
                        "instance masks, ~3.5 GB, GPU) or clipseg (CIDAS/clipseg, small, softer)")
    p.add_argument("--seg-cpu", action="store_true", help="run the SAM3 segmenter on CPU (slow; default GPU)")
    p.add_argument("--no-seg-cache", action="store_true",
                   help="re-estimate sky/vehicle masks instead of loading the on-disk cache "
                        "(cache is keyed by segmenter+prompt+threshold under the scene cache dir)")
    p.add_argument("--sky-threshold", type=float, default=0.35,
                   help="sky mask threshold (CLIPSeg sigmoid prob, or SAM3 detection score)")
    p.add_argument("--sky-gpu", action="store_true", help="run the CLIPSeg segmenter on GPU (default CPU)")
    p.add_argument("--mask-vehicles", action="store_true",
                   help="segment vehicles (via --segmenter) and drop them from BOTH the init cloud and "
                        "the training loss. Appearance-based, so ALL vehicles are removed (parked "
                        "included) -- motion-aware removal needs the per-instance tracking step")
    p.add_argument("--vehicle-prompt", default="vehicle", help="CLIPSeg prompt for the vehicle class")
    p.add_argument("--vehicle-threshold", type=float, default=0.5,
                   help="vehicle mask threshold (CLIPSeg sigmoid prob, or SAM3 detection score)")
    p.add_argument("--range-mask", action="store_true",
                   help="bound the reconstruction by range: drop init-cloud points farther than "
                        "--range-thresh from EVERY camera, and (with --sky-lambda) supervise their "
                        "pixels toward black via the sky->black term, so distant unobserved regions "
                        "do not grow arbitrary floaters. Per-pixel, keyed off the init cloud's one "
                        "point per pixel")
    p.add_argument("--range-thresh", type=float, default=60.0,
                   help="range threshold in metres for --range-mask (min distance to all cameras)")
    p.add_argument("--rerun", action="store_true", help="spawn a native Rerun viewer (needs a display)")
    p.add_argument("--save", type=Path, default=None, help="write a .rrd (opened later, not live)")
    p.add_argument("--serve", action="store_true",
                   help="stream to a live Rerun gRPC server; connect a viewer for live updates "
                        "(headless/remote-friendly). Prints the connect URL.")
    p.add_argument("--serve-port", type=int, default=None, help="gRPC port for --serve (default 9876)")
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
    # Align the cached poses to the (possibly --max-frames-limited) keyframe subset by
    # token, so world_from_cam stays 1:1 with `frames`.
    pose_idx = {t: i for i, t in enumerate(mu.tokens)}
    frames = [(kf, depth[kf.token]) for kf in keyframes if kf.token in depth and kf.token in pose_idx]
    world_from_cam = mu.world_from_cam[[pose_idx[kf.token] for kf, _ in frames]]  # (N,4,4) camera->world
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

    gt_poses = None
    if args.gt_poses:
        # ORACLE: use nuScenes GT camera poses DIRECTLY -- they are already metric and live in the
        # map/GT frame, so there is NOTHING to align: no res, no Sim(3) on the poses. res.scale is
        # used only to put the homography-rectified depth into metres (below). The init cloud is
        # rebuilt from these GT poses so cloud and cameras share the GT frame. GT never sets scale.
        s2e_m = frames[0][0].calib.sensor2ego.matrix()
        gt_poses = np.stack([kf.ego2global_gt.matrix() @ s2e_m for kf, _ in frames])  # (N,4,4) metres
        poses_m = gt_poses
        print(f"[gt-poses] ORACLE: raw GT camera poses in the GT metric frame ({len(gt_poses)} frames)")

    colmap_poses = None
    if args.colmap_poses:
        # Cached COLMAP poses, already in the nuScenes global metric frame (aligned by the same
        # Route-B Umeyama over the full trajectory). Subset to `frames` by token so they stay 1:1,
        # exactly like the GT branch -- the seed cloud is rebuilt from them below.
        cp_path = args.cache_root / scene_name / "colmap_poses_global.npz"
        if not cp_path.exists():
            print(f"[colmap-poses] {cp_path} not found -- run scratchpad/cache_colmap_poses.py first")
            return 1
        d = np.load(cp_path, allow_pickle=True)
        cmap = {str(t): P for t, P in zip(d["tokens"], d["poses"])}
        missing = [kf.token for kf, _ in frames if kf.token not in cmap]
        if missing:
            print(f"[colmap-poses] {len(missing)}/{len(frames)} frames missing from {cp_path} "
                  "(unregistered by COLMAP?) -- cannot use as training cameras")
            return 1
        colmap_poses = np.stack([cmap[kf.token] for kf, _ in frames])
        poses_m = colmap_poses
        print(f"[colmap-poses] COLMAP poses in the global metric frame ({len(colmap_poses)} frames, "
              f"scale={float(d['scale']):.4f}, GPS residual={float(d['residual']):.3f} m)")

    # GT / COLMAP both supply metric poses already in the global frame; the seed cloud is then
    # back-projected from THESE poses (metric DA3 depth * scale) instead of the DA3 metric-upgrade poses.
    override_poses = gt_poses if gt_poses is not None else colmap_poses

    # ---- sky / vehicle masks: drop those pixels from the cloud + training loss. SAM3 (default)
    #      gives sharper instance masks than CLIPSeg; a single SAM3 model is reused across prompts
    #      and released before training. --segmenter clipseg falls back to the CLIPSeg backend. ----
    _seg_cache = {}

    def _segment(prompt, threshold):
        # on-disk cache keyed by backend+prompt+threshold, so masks are estimated once per config
        # (like depth / metric upgrade) and reused across runs. --no-seg-cache forces re-estimation.
        tag = f"{args.segmenter}-{prompt}-{threshold}"
        if not args.no_seg_cache:
            cached = cache.load_seg_masks(args.cache_root, scene_name, tag)
            if cached is not None:
                print(f"[seg cache] '{prompt}' ({args.segmenter}, thr {threshold}): loaded {len(cached)} masks")
                return cached
        kfs = [kf for kf, _ in frames]
        if args.segmenter == "sam3":
            seg = _seg_cache.get("sam3")
            if seg is None:
                seg = _seg_cache["sam3"] = Sam3Segmenter(
                    SegConfig(prompt=prompt, threshold=threshold,
                              device=("cpu" if args.seg_cpu else None)))  # GPU by default (pre-training)
            seg.config.prompt, seg.config.threshold = prompt, threshold
        else:
            seg = RoadSegmenter(SegConfig(prompt=prompt, threshold=threshold,
                                          device=(None if args.sky_gpu else "cpu")))
        gms = seg.segment_scene(kfs)
        if not args.no_seg_cache:
            cache.save_seg_masks(args.cache_root, scene_name, tag, gms)
        return {gm.token: gm.mask for gm in gms}

    sky_masks = {}
    if args.mask_sky or args.remove_sky_gaussians:
        sky_masks = _segment("sky", args.sky_threshold)
        frac = np.mean([m.mean() for m in sky_masks.values()]) if sky_masks else 0.0
        print(f"sky masks: segmented {len(sky_masks)} keyframes ({args.segmenter} 'sky'), ~{frac:.1%} sky")

    vehicle_masks = {}
    if args.mask_vehicles:
        vehicle_masks = _segment(args.vehicle_prompt, args.vehicle_threshold)
        frac = np.mean([m.mean() for m in vehicle_masks.values()]) if vehicle_masks else 0.0
        print(f"vehicle masks: segmented {len(vehicle_masks)} keyframes "
              f"({args.segmenter} '{args.vehicle_prompt}'), ~{frac:.1%} vehicle")

    if _seg_cache.get("sam3") is not None:
        _seg_cache["sam3"].release()   # free the ~3.5 GB SAM3 off the GPU before training

    far_masks = {}   # token -> (H,W) bool, populated below once metric_depths/poses exist (--range-mask)

    def drop_mask(kf, dm, *, da3_sky=True):
        """Pixels to exclude: union of the enabled sky, vehicle and far-range masks for this frame.
        The init cloud passes ``da3_sky=True`` so it also drops DA3's own sky estimate when
        ``--mask-sky`` is off (preserving prior behaviour); the loss passes ``da3_sky=False`` so it
        masks only the explicitly-requested classes. Returns an (H, W) bool mask, or None if nothing
        is masked for this frame."""
        if args.mask_sky:
            m = sky_masks.get(kf.token)
        else:
            m = dm.sky if da3_sky else None
        if args.mask_vehicles and vehicle_masks.get(kf.token) is not None:
            v = vehicle_masks[kf.token]
            m = v if m is None else (m | v)
        if args.range_mask and far_masks.get(kf.token) is not None:
            f = far_masks[kf.token]
            m = f if m is None else (m | f)
        return m

    # ---- init cloud: back-project depth (metres), voxel-downsample PER KEYFRAME, then
    #      concatenate and downsample the merged cloud once more. Thinning each frame
    #      before the merge keeps the concat + final pass cheap. ----
    # rectify each DA3 depth to metric via the DAQ homography H (H acts in the world frame, so per
    # pixel: unproject with the DA3 pose -> apply H -> reproject with the metric pose to read z).
    # Up-to-scale here; * scale -> metres. This is the depth the cloud AND the depth loss use.
    metric_depths = [metric_depth(dm.depth, dm.intrinsic, dm.extrinsic, mu.H, np.linalg.inv(world_from_cam[i]))
                     for i, (kf, dm) in enumerate(frames)]
    # range mask: per pixel, is its metric init point beyond --range-thresh from EVERY camera? Uses the
    # training poses (poses_m) so the point matches the cloud exactly. Feeds drop_mask (init + loss
    # exclusion) and the black-supervision mask below. Must precede the cloud loop, which calls drop_mask.
    if args.range_mask:
        centres = poses_m[:, :3, 3]
        for i, (kf, dm) in enumerate(frames):
            far_masks[kf.token] = _range_far_mask(metric_depths[i] * scale, K_true, poses_m[i],
                                                  centres, args.range_thresh)
        frac = np.mean([m.mean() for m in far_masks.values()]) if far_masks else 0.0
        print(f"range mask: pixels > {args.range_thresh:.0f} m from all cameras -> ~{frac:.1%} per frame "
              f"(dropped from init, supervised black with --sky-lambda)")
    raw_xyz, raw_rgb, ds_xyz, ds_rgb = [], [], [], []
    voxel_ok = True
    for i, (kf, dm) in enumerate(frames):
        if override_poses is not None:   # GT/COLMAP run: metric depth (metres) back-projected from those poses
            pts, cols = metric_point_cloud(metric_depths[i] * scale, kf.image(), K_true, override_poses[i],
                                           stride=args.stride, conf=dm.conf, sky=drop_mask(kf, dm))
        else:                      # standard: up-to-scale depth from the metric-upgrade poses, then * scale
            pts, cols = metric_point_cloud(metric_depths[i], kf.image(), K_true, world_from_cam[i],
                                           stride=args.stride, conf=dm.conf, sky=drop_mask(kf, dm))
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
    train_idx, test_idx = holdout_indices(len(frames), every=args.holdout_every, offset=args.holdout_offset)
    print(f"{len(frames)} frames: {len(train_idx)} train / {len(test_idx)} held-out for novel-view PSNR")

    # ---- Rerun overlays: init cloud + trajectory + camera frustums (upright frame) ----
    if args.rerun or args.save is not None or args.serve:
        # Distinct app id per mode so the viewer doesn't reuse the training blueprint
        # (with its empty render/loss panels) for a prep-only point-cloud recording.
        app_id = f"nuslam-gs-{scene_name}" if args.train else f"nuslam-pointcloud-{scene_name}"
        rrlog.init(app_id, spawn=args.rerun, save=args.save,
                   serve=args.serve, serve_port=args.serve_port)
        rrlog.log_points("init/cloud", xyz @ R_VIZ.T, colors=rgb, radii=args.point_size, static=True)
        rrlog.log_trajectory("traj/est", poses_m[:, :3, 3] @ R_VIZ.T, color=(50, 120, 240), static=True)
        for i, (kf, _) in enumerate(frames):
            rrlog.log_estimated_camera(f"est/{i:03d}", T_VIZ @ poses_m[i], K_true,
                                       kf.calib.width, kf.calib.height, channel=args.camera, static=True)

        # ---- reference overlays: GPS track, and (oracle) GT trajectory + lidar ----
        # These live in the nuScenes map frame; map them into this reconstruction frame
        # with the Route-B Sim(3) inverse: map M -> recon P = (M - t) Rs, then R_VIZ.
        if res is not None:
            if override_poses is not None:        # poses_m already in the nuScenes global frame: no remap
                to_view = lambda M: np.asarray(M, float) @ R_VIZ.T
            else:                                 # DA3 recon frame: bring the map reference in via Sim(3) inv
                Rs = res.T[:3, :3] / res.scale    # orthonormal (scale un-folded)
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
    # Handoff to the Gaussian reconstruction (nuslam.recon.gaussians.train_gaussians).
    #
    # Prepared inputs above, ready to consume:
    #   xyz (M,3) float32, rgb (M,3) uint8  -- metric init cloud (seed Gaussian means/colors)
    #   poses_m (N,4,4)                      -- metric camera->world  (viewmats = inv(poses_m))
    #   K_true (3,3)                         -- intrinsics (per-view Ks = tile over N)
    #   frames[i][0].image()                -- image for view i
    #   train_idx / test_idx                -- optimize on train, score novel views on test
    #
    # After it returns, visualize alongside the overlays above and score each stage with
    # the wired hooks:
    #   rrlog.log_gaussians("gs/means", means, scales, quats_wxyz, colors_rgba)  # 3D splats
    #   rrlog.log_image("render/est", rendered);  rrlog.log_image("render/gt", gt)  # 2D compare
    #   e = evaluate_render(rendered, gt);  # PSNR/SSIM/L1 on a held-out view
    #   record_metrics(args.cache_root / scene_name / "metrics.json", "photometric",
    #                  {"psnr": e.psnr, "ssim": e.ssim, ...})  # per-stage metric log
    # ===================================================================================
    if not args.train:
        print("\nStage-3 inputs prepared. Pass --train to fit the Gaussians "
              "(nuslam.recon.gaussians consumes the inputs above).")
        return 0

    images = [kf.image() for kf, _ in frames]
    to_viz = bool(args.rerun or args.save is not None or args.serve)
    # Per-run output dir so runs never overwrite each other (concatenate later for final visuals).
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = args.cache_root / scene_name / "runs" / run_name
    gs_dir = log_dir / "gs_snapshots"; gs_dir.mkdir(parents=True, exist_ok=True)
    print(f"run outputs -> {log_dir}")
    C0 = 0.28209479177387814  # SH DC basis; display colour = DC*C0 + 0.5

    def gs_numpy(g):
        means, scales, quats, opac, sh = g
        return {
            "means": means.detach().cpu().numpy(),
            "scales": scales.detach().cpu().numpy(),
            "quats": torch.nn.functional.normalize(quats, dim=-1).detach().cpu().numpy(),
            "opacities": opac.detach().cpu().numpy(),
            "sh": sh.detach().cpu().numpy(),
        }

    def gs_rgba(gd):
        rgb01 = np.clip(gd["sh"][:, 0, :] * C0 + 0.5, 0.0, 1.0)
        return (np.concatenate([rgb01, gd["opacities"].reshape(-1, 1)], axis=1) * 255).astype(np.uint8)

    def log_gs(name, gd, *, static):
        rrlog.log_transform(name, R_VIZ, static=static)   # upright, matching init/cloud
        rrlog.log_gaussians(f"{name}/splats", gd["means"], gd["scales"], gd["quats"],
                            gs_rgba(gd), static=static)

    # ---- persistent training log (CSV) + GS snapshots + render PNGs + live Rerun ----
    csv_file = open(log_dir / "train_log.csv", "w", newline="")
    cw = csv.writer(csv_file); cw.writerow(["step", "loss", "photometric", "dssim", "num_gaussians", "heldout_psnr"])
    render_dir = log_dir / "renders"; render_dir.mkdir(parents=True, exist_ok=True)
    ti, hi = int(train_idx[0]), int(test_idx[0])
    Image.fromarray(frames[hi][0].image()).save(render_dir / "heldout_gt.png")   # gt once
    Image.fromarray(frames[ti][0].image()).save(render_dir / "train_gt.png")

    def _png(img01):  # (H,W,3) float [0,1] -> uint8
        return (np.clip(img01, 0, 1) * 255).astype(np.uint8)

    def on_log(step, loss, photo, dssim, gaussians, render):
        gd = gs_numpy(gaussians)
        n = len(gd["means"])
        ho = render(poses_m[hi], K_true)                 # render each view ONCE, reuse
        tr = render(poses_m[ti], K_true)
        e = evaluate_render(ho, frames[hi][0].image())   # held-out score
        print(f"iter {step:6d}  loss {loss:.4f}  photo {photo:.4f}  dssim {dssim:.4f}  "
              f"N={n}  heldout_psnr={e.psnr:.2f}")
        cw.writerow([step, loss, photo, dssim, n, e.psnr]); csv_file.flush()
        if args.snapshot_every and step % args.snapshot_every == 0:        # intermediate GS (disk-heavy)
            np.savez_compressed(gs_dir / f"gs_{step:06d}.npz", **gd)
        Image.fromarray(_png(ho)).save(render_dir / f"heldout_est_{step:06d}.png")   # inspectable PNGs
        Image.fromarray(_png(tr)).save(render_dir / f"train_est_{step:06d}.png")
        if to_viz:
            rrlog.set_step(step)
            rrlog.log_scalar("train/loss", loss); rrlog.log_scalar("train/photometric", photo)
            rrlog.log_scalar("train/dssim", dssim); rrlog.log_scalar("heldout/psnr", e.psnr)
            rrlog.log_image("render/heldout_est", ho); rrlog.log_image("render/heldout_gt", frames[hi][0].image())
            rrlog.log_image("render/train_est", tr); rrlog.log_image("render/train_gt", frames[ti][0].image())
            log_gs("train/gs", gd, static=True)   # overwrite-in-place: live view updates but memory stays
                                                  # bounded (a full GS set per log step would balloon a long
                                                  # --serve run); the evolution is kept in the .npz snapshots

    ckpt = {k: np.load(args.from_checkpoint)[k] for k in np.load(args.from_checkpoint).files} \
        if args.from_checkpoint else None
    if ckpt is not None:
        print(f"warm-starting from checkpoint {args.from_checkpoint} ({len(ckpt['means'])} Gaussians)")
        if args.remove_sky_gaussians and sky_masks:
            keep = _sky_keep_mask(ckpt["means"], poses_m, K_true, frames, sky_masks)
            n0 = len(keep); ckpt = {k: v[keep] for k, v in ckpt.items()}
            print(f"removed sky Gaussians: {n0} -> {len(ckpt['means'])} kept")
    # per-frame keep-masks (complement of the sky/vehicle drop mask), so the training loss ignores
    # those pixels and no Gaussians are grown to reconstruct them
    masking = bool((args.mask_sky and sky_masks) or (args.mask_vehicles and vehicle_masks)
                   or (args.range_mask and far_masks))
    train_masks = [~drop_mask(kf, dm, da3_sky=False) for kf, dm in frames] if masking else None
    # per-view DA3 metric depth (recon depth * scale, same units as the Gaussians) for the
    # expected-depth loss; only built when the depth loss is on
    depth_maps = [metric_depths[i] * scale for i in range(len(frames))] if args.depth_lambda > 0 else None
    # pixels supervised toward black by the sky_lambda term: the segmented sky, plus (--range-mask) the
    # far-range pixels whose init point lies beyond the threshold from every camera -- left out of the
    # photometric loss above, so pushing them black keeps them from growing arbitrary floaters.
    black_src = {}
    if args.mask_sky and sky_masks:
        for kf, _ in frames:
            black_src[kf.token] = sky_masks[kf.token]
    if args.range_mask and far_masks:
        for kf, _ in frames:
            f = far_masks[kf.token]
            black_src[kf.token] = f if kf.token not in black_src else (black_src[kf.token] | f)
    sky_only_masks = ([black_src[kf.token] for kf, _ in frames]
                      if (args.sky_lambda > 0 and black_src) else None)
    # scale the means LR by the scene extent (3DGS spatial_lr_scale = camera-bounding radius): the
    # 1.6e-4 base is meant to be multiplied by this, else in a metric scene of tens of metres the
    # means barely move. Also scales the MCMC position-noise (which uses lr_for["means"]).
    cam_c = poses_m[:, :3, 3]
    spatial_lr_scale = float(np.linalg.norm(cam_c - cam_c.mean(0), axis=1).max() * 1.1)
    lr_for = {**LR_FOR, "means": LR_FOR["means"] * spatial_lr_scale}
    print(f"spatial_lr_scale = {spatial_lr_scale:.1f} m -> means LR {lr_for['means']:.2e} "
          f"(base {LR_FOR['means']:.1e})")
    gs_tuple, render = train_gaussians(
        xyz, rgb, poses_m, K_true, images, train_idx,
        lr_for=lr_for, num_iters=args.num_iters, structural_lambda=args.structural_lambda,
        log_every=args.log_every, on_log=on_log, init_gaussians=ckpt,
        clip_scales_to_knn=args.clip_scales, masks=train_masks, optimize_poses=args.optimize_poses,
        opacity_reg=args.opacity_reg, scale_reg=args.scale_reg,
        depths=depth_maps, depth_lambda=args.depth_lambda,
        sky_masks=sky_only_masks, sky_lambda=args.sky_lambda,
    )
    csv_file.close()
    print(f"\nfit complete ({args.num_iters} iters, structural_lambda={args.structural_lambda})")

    # ---- final GS: persist + score the held-out views + log the run's metrics ----
    final_gd = gs_numpy(gs_tuple)
    np.savez_compressed(log_dir / "gs_final.npz", **final_gd)
    errs = [evaluate_render(render(poses_m[i], K_true), frames[i][0].image()) for i in test_idx]
    metrics = {
        "psnr": float(np.mean([e.psnr for e in errs])),
        "ssim": float(np.mean([e.ssim for e in errs])),
        "l1": float(np.mean([e.l1 for e in errs])),
    }
    mpath = record_metrics(log_dir / "metrics.json", "photometric", metrics)
    print(f"held-out ({len(test_idx)} views): PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}  "
          f"L1={metrics['l1']:.4f}")
    print(f"logs: {log_dir/'train_log.csv'}  |  GS: {log_dir/'gs_final.npz'} (+ {gs_dir}/)  |  metrics: {mpath}")

    # ---- final Gaussians + a few novel-view renders in Rerun ----
    if to_viz:
        log_gs("gs", final_gd, static=True)
        for i in test_idx[:3]:
            rrlog.log_image(f"render/est/{i:03d}", render(poses_m[i], K_true))
            rrlog.log_image(f"render/gt/{i:03d}", frames[i][0].image())
        print("logged final Gaussians + novel-view renders to Rerun"
              + (f" -> {args.save}" if args.save else ""))

    if args.serve:  # keep the process (and the live server) alive for review
        try:
            input("\n[rerun] serving live -- press Enter to stop the server and exit...")
        except (EOFError, KeyboardInterrupt):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
