"""Frozen-pose 3DGS on the tracked van: scaled COLMAP poses + points, masked-van images, van global
metric depth (SILog), supervise-black background + blank-area prune, cap 100k, batch 10. Invokes the
author's train_gaussians (unchanged) with these inputs."""
import numpy as np, pycolmap, torch
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache
from nuslam.recon import metric_depth, train_gaussians
from nuslam.eval import holdout_indices
SP = "out/vehicle_cache"
LR_FOR = {"means": 1.6e-4, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2, "colors": 2.5e-3, "pose_quats": 1e-3, "pose_trans": 1e-3}

src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
mu = cache.load_metric_upgrade("out/frontend_cache", "scene-0061"); dep = cache.load_depth("out/frontend_cache", "scene-0061")
pidx = {t: i for i, t in enumerate(mu.tokens)}; K = kfs[0].calib.intrinsic
bt = dict((int(f), m) for f, m in np.load(f"{SP}/best_track.npy", allow_pickle=True))
rec = pycolmap.Reconstruction(f"{SP}/colmap_van/sparse_tuned/0")

def cfw(im):
    rig = im.cam_from_world; rig = rig() if callable(rig) else rig
    try: M = np.asarray(rig.matrix())
    except Exception: M = np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
    T = np.eye(4); T[:3] = M; return T
info = {im.image_id: (int(im.name[1:3]), cfw(im)) for im in rec.images.values()}
mds = {}
for iid, (fi, _) in info.items():
    kf = kfs[fi]
    mds[fi] = metric_depth(dep[kf.token].depth, dep[kf.token].intrinsic, dep[kf.token].extrinsic, mu.H, np.linalg.inv(mu.world_from_cam[pidx[kf.token]]))

# scale s = median(z_global / z_colmap)
ratios = []
for p in rec.points3D.values():
    X = np.append(p.xyz, 1.0)
    for el in p.track.elements:
        fi, C = info[el.image_id]; zc = (C @ X)[2]; u, v = rec.images[el.image_id].points2D[el.point2D_idx].xy
        if 0 <= int(v) < mds[fi].shape[0] and 0 <= int(u) < mds[fi].shape[1] and zc > 1e-6 and mds[fi][int(v), int(u)] > 1e-6:
            ratios.append(mds[fi][int(v), int(u)] / zc)
s = float(np.median(ratios)); print(f"scale s = {s:.4f}")

order = sorted(info.items(), key=lambda kv: kv[1][0])
poses, images, depths, masks, blackmasks, fis = [], [], [], [], [], []
for iid, (fi, C) in order:
    Cs = C.copy(); Cs[:3, 3] *= s; poses.append(np.linalg.inv(Cs))             # camera->world, scaled metric van frame
    van = bt[fi]; img = np.asarray(kfs[fi].image()).copy(); img[~van] = 0        # masked-van image (black bg)
    images.append(img); masks.append(van); blackmasks.append(~van)
    depths.append((mds[fi] * van).astype(np.float32)); fis.append(fi)            # van-only metric depth (0 elsewhere) for SILog
poses = np.stack(poses).astype(np.float32)
Xall = np.array([p.xyz for p in rec.points3D.values()], np.float32) * s          # scaled COLMAP init means
rgb = np.array([p.color for p in rec.points3D.values()])[:, :3].astype(np.uint8)
tr, te = holdout_indices(len(poses), every=8, offset=0)
cam_c = poses[:, :3, 3]; spatial = float(np.linalg.norm(cam_c - cam_c.mean(0), axis=1).max() * 1.1)
lr_for = {**LR_FOR, "means": LR_FOR["means"] * spatial}
print(f"{len(poses)} views ({len(tr)} train / {len(te)} held-out), {len(Xall)} init pts, spatial_lr_scale {spatial:.2f}")

def on_log(step, loss, photo, dssim, g, render, ps, gr, cr):
    m = g["means"] if isinstance(g, dict) else g[0]
    print(f"  iter {step:5d}  loss {loss:.4f}  photo {photo:.4f}  N={int(m.shape[0])}", flush=True)

def tonp(v): return v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)

gs, render, _ = train_gaussians(
    Xall, rgb, poses, K, images, tr, lr_for=lr_for, num_iters=10000, batch_size=10,
    structural_lambda=0.2, log_every=200, on_log=on_log,
    masks=masks, depths=depths, depth_lambda=0.05, depth_silog=True, depth_silog_lambda=1.0,
    sky_masks=blackmasks, sky_lambda=0.05, opacity_reg=0.01, scale_reg=0.01,
    prune_offscene=True, prune_range=0.0, prune_every=100, cap_max=100_000)
means = gs["means"] if isinstance(gs, dict) else gs[0]
print(f"done: {int(means.shape[0])} Gaussians")
items = gs.items() if isinstance(gs, dict) else zip(["means", "scales", "quats", "opacities", "sh"], gs)
np.savez(f"{SP}/vehicle_gs.npz", **{k: tonp(v) for k, v in items})

# render a couple held-out + train views to eyeball
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = list(te[:3]) + list(tr[:3])
fig, ax = plt.subplots(2, len(show), figsize=(len(show) * 2.4, 4.6))
for k, i in enumerate(show):
    r = np.clip(tonp(render(poses[i], K)), 0, 1)
    ax[0, k].imshow(images[i]); ax[0, k].set_title(("held" if i in te else "train") + f" f{fis[i]}", fontsize=8); ax[0, k].axis("off")
    ax[1, k].imshow(r); ax[1, k].axis("off")
ax[0, 0].set_ylabel("target"); ax[1, 0].set_ylabel("render")
fig.suptitle("vehicle 3DGS (SILog, cap 100k): target (top) vs render (bottom)")
fig.tight_layout(); fig.savefig(f"{SP}/vehicle_gs_render.png", dpi=90, bbox_inches="tight"); print("wrote vehicle_gs_render.png")
