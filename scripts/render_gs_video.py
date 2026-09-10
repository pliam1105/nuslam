"""Render videos from a finished GS run (viz only). Two modes:

  panel       -- a grid of a few train + heldout views (GT over estimate),
                 the estimate evolving across the run's saved snapshots.
  flythrough  -- the final snapshot rendered from a smooth camera path along
                 the (metric) camera trajectory.

Rendering reuses the author's `render(pose, K)` closure unchanged: we call
`train_gaussians(num_iters=0, init_gaussians=<snapshot>)`, which loads a snapshot's
Gaussians and hands back that same closure -- no rendering code is written here.
Pose/split/scale setup mirrors run_gs (cached, so cheap).

    python scripts/render_gs_video.py --scene scene-0061 --run run4 --mode both
"""
import argparse
import glob
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nuslam.data import NuScenesMonoSource, gps_positions_at, load_proprio_streams  # noqa: E402
from nuslam.eval import holdout_indices  # noqa: E402
from nuslam.frontend import cache  # noqa: E402
from nuslam.recon import resolve_scale_gps, train_gaussians  # noqa: E402

LR_FOR = {"means": 1.6e-4, "quats": 1.0e-3, "scales": 5.0e-3, "opacities": 5.0e-2, "colors": 2.5e-3}


def load_geometry(args):
    """Rebuild frames, metric poses, K, images, and the train/heldout split (mirrors run_gs)."""
    source = NuScenesMonoSource(args.dataroot, args.version, camera=args.camera)
    scene = args.scene or source.list_scenes()[0][1]
    keyframes = source.load_scene(scene, max_frames=args.max_frames)
    scene_name = keyframes[0].scene_name

    mu = cache.load_metric_upgrade(args.cache_root, scene_name)
    depth = cache.load_depth(args.cache_root, scene_name)
    if mu is None or depth is None:
        sys.exit(f"need cached metric upgrade + depth under {args.cache_root}/{scene_name}")
    pose_idx = {t: i for i, t in enumerate(mu.tokens)}
    frames = [(kf, depth[kf.token]) for kf in keyframes if kf.token in depth and kf.token in pose_idx]
    world_from_cam = mu.world_from_cam[[pose_idx[kf.token] for kf, _ in frames]]
    K_true = mu.K_true

    scale, res = 1.0, None
    if not args.no_scale:
        streams = load_proprio_streams(args.dataroot, scene_name, version=args.version)
        if streams.gps:
            gps_xy, gps_valid = gps_positions_at(streams, [kf.timestamp_us for kf, _ in frames])
            s2e = frames[0][0].calib.sensor2ego.matrix()
            try:
                res = resolve_scale_gps(world_from_cam, gps_xy, s2e, gps_valid)
                scale = float(res.scale)
            except NotImplementedError:
                pass
    poses_m = world_from_cam.copy()
    poses_m[:, :3, 3] *= scale
    print(f"[geometry] {len(frames)} frames, metric scale {scale:.4f}")

    # actual nuScenes GT camera poses, mapped from the map frame into the metric-recon frame
    # the Gaussians live in, via the Route-B Sim(3) (map -> recon: P = Rs^T (M - t_map)).
    gt_poses_recon = None
    if res is not None:
        s2e_m = frames[0][0].calib.sensor2ego.matrix()
        Rs = res.T[:3, :3] / res.scale            # orthonormal recon->map rotation
        t_map = res.T[:3, 3]
        gt_poses_recon = np.stack([map_to_recon(kf.ego2global_gt.matrix() @ s2e_m, Rs, t_map)
                                   for kf, _ in frames])
        dc = np.linalg.norm(gt_poses_recon[:, :3, 3] - poses_m[:, :3, 3], axis=1)
        print(f"[gt] GT vs recon camera-centre offset: mean {dc.mean():.2f} m, max {dc.max():.2f} m")

    # stack once: train_gaussians does torch.tensor(images), which is ~200x slower on a
    # list of arrays than on a single ndarray (83s -> 0.4s per snapshot load).
    images = np.stack([kf.image() for kf, _ in frames])
    train_idx, test_idx = holdout_indices(len(frames), every=args.holdout_every)
    return frames, poses_m, K_true, images, np.asarray(train_idx), np.asarray(test_idx), gt_poses_recon


def map_to_recon(pose_map, Rs, t_map):
    """Map a cam->map pose into the metric-recon frame: R' = Rs^T R,  c' = Rs^T (c - t_map)."""
    M = np.eye(4)
    M[:3, :3] = Rs.T @ pose_map[:3, :3]
    M[:3, 3] = Rs.T @ (pose_map[:3, 3] - t_map)
    return M


def render_from_snapshot(snap_path, poses_m, K_true, images, train_idx):
    """Load a snapshot's Gaussians and return the author's render(pose, K) closure bound to them."""
    import torch
    d = np.load(snap_path)
    snap = {k: d[k] for k in ("means", "scales", "quats", "opacities", "sh")}
    dummy = np.zeros((len(snap["means"]), 3), np.float32)
    (_gs, render) = train_gaussians(
        dummy, dummy, poses_m, K_true, images, train_idx,
        lr_for=LR_FOR, num_iters=0, init_gaussians=snap,
    )
    return render, (_gs, torch)


def u8(img01):
    return (np.clip(np.asarray(img01), 0, 1) * 255).astype(np.uint8)


def fit_w(img, w):
    h = round(img.shape[0] * w / img.shape[1])
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def label(img, text, y=22):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def make_panel(views, gts, ests, header, cell_w):
    """views: list of labels; gts/ests: uint8 images. Two rows (GT / estimate) x N cols."""
    cols = []
    for lab, gt, est in zip(views, gts, ests):
        g = label(fit_w(gt, cell_w).copy(), f"{lab} GT")
        e = label(fit_w(est, cell_w).copy(), f"{lab} est")
        pad = np.full((6, cell_w, 3), 30, np.uint8)
        cols.append(np.vstack([g, pad, e]))
    body = np.hstack([np.hstack([c, np.full((c.shape[0], 6, 3), 30, np.uint8)]) for c in cols])
    bar = np.full((34, body.shape[1], 3), 20, np.uint8)
    label(bar, header, y=24)
    return np.vstack([bar, body])


def interp_poses(poses, k):
    if k <= 1:
        return poses
    t = np.arange(len(poses))
    slerp = Slerp(t, Rotation.from_matrix(poses[:, :3, :3]))
    fine = np.linspace(0, len(poses) - 1, (len(poses) - 1) * k + 1)
    Rf = slerp(fine).as_matrix()
    tf = np.stack([np.interp(fine, t, poses[:, j, 3]) for j in range(3)], axis=1)
    out = np.tile(np.eye(4), (len(fine), 1, 1))
    out[:, :3, :3] = Rf
    out[:, :3, 3] = tf
    return out


def video_panel(args, geom, snaps, out):
    frames_meta, poses_m, K_true, images, train_idx, test_idx, _gt = geom
    tr = np.linspace(0, len(train_idx) - 1, args.n_train).round().astype(int)
    he = np.linspace(0, len(test_idx) - 1, args.n_heldout).round().astype(int)
    view_ids = [int(train_idx[i]) for i in tr] + [int(test_idx[i]) for i in he]
    labels = [f"train{int(train_idx[i])}" for i in tr] + [f"held{int(test_idx[i])}" for i in he]
    gts = [frames_meta[v][0].image() for v in view_ids]

    kept = snaps[:: args.snapshot_stride]
    if snaps[-1] not in kept:
        kept.append(snaps[-1])
    with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
        for si, sp in enumerate(kept):
            step = int(Path(sp).stem.split("_")[-1])
            render, (gs, torch) = render_from_snapshot(sp, poses_m, K_true, images, train_idx)
            ests = [u8(render(poses_m[v], K_true)) for v in view_ids]
            panel = make_panel(labels, gts, ests, f"{args.run}  iter {step}", args.cell_w)
            w.append_data(panel)
            del render, gs
            torch.cuda.empty_cache()
            print(f"  panel frame {si+1}/{len(kept)} (iter {step})")
    print(f"wrote {out}")


def video_flythrough(args, geom, snaps):
    frames_meta, poses_m, K_true, images, train_idx, test_idx, gt_poses = geom
    jobs = []
    if args.trajectory in ("est", "both"):
        jobs.append((poses_m, "camera trajectory", f"out/{args.run}_flythrough.mp4"))
    if args.trajectory in ("gt", "both"):
        if gt_poses is None:
            sys.exit("--trajectory gt/both needs the Route-B Sim(3) (GPS scale); none available (--no-scale or no GPS)")
        jobs.append((gt_poses, "GT trajectory", f"out/{args.run}_flythrough_gt.mp4"))

    render, (gs, torch) = render_from_snapshot(snaps[-1], poses_m, K_true, images, train_idx)  # once, reused
    for traj, tag, out in jobs:
        path = interp_poses(traj, args.interp)
        with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
            for i, P in enumerate(path):
                frame = label(u8(render(P, K_true)).copy(), f"{args.run}  3DGS along {tag}  {i+1}/{len(path)}")
                w.append_data(frame)
                if (i + 1) % 40 == 0 or i + 1 == len(path):
                    print(f"  [{tag}] frame {i+1}/{len(path)}")
        print(f"wrote {out}")
    del render, gs
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene")
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", choices=["panel", "flythrough", "both"], default="both")
    ap.add_argument("--cache-root", default="out/frontend_cache")
    ap.add_argument("--dataroot", default="data/nuscenes")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-scale", action="store_true")
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=3, help="# train views in the panel")
    ap.add_argument("--n-heldout", type=int, default=3, help="# heldout views in the panel")
    ap.add_argument("--cell-w", type=int, default=360, help="panel cell width (px)")
    ap.add_argument("--snapshot-stride", type=int, default=2, help="panel: use every Nth snapshot")
    ap.add_argument("--interp", type=int, default=6, help="flythrough: substeps between keyframe poses")
    ap.add_argument("--trajectory", choices=["est", "gt", "both"], default="est",
                    help="flythrough path: est = DA3/metric-upgrade estimate, gt = actual nuScenes GT "
                         "(mapped into the recon frame), both = render each (reuses one snapshot load)")
    ap.add_argument("--fps", type=int, default=8)
    args = ap.parse_args()

    run_dir = Path(args.cache_root) / (args.scene or "") / "runs" / args.run
    snaps = sorted(glob.glob(str(run_dir / "gs_snapshots" / "gs_*.npz")))
    if not snaps:
        sys.exit(f"no snapshots under {run_dir}")
    geom = load_geometry(args)

    if args.mode in ("panel", "both"):
        video_panel(args, geom, snaps, f"out/{args.run}_panel.mp4")
    if args.mode in ("flythrough", "both"):
        video_flythrough(args, geom, snaps)


if __name__ == "__main__":
    main()
