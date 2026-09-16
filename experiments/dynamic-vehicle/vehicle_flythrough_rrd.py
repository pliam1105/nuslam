"""Flythrough mp4 + rrd for the trained vehicle 3DGS (same as the scene GS runs)."""
import numpy as np, pycolmap, imageio.v2 as imageio, torch
from scipy.spatial.transform import Rotation, Slerp
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache
from nuslam.recon import metric_depth, train_gaussians
from nuslam.viz import rerun_logging as rrlog
SP = "out/vehicle_cache"
C0 = 0.28209479177387814
LR = {"means": 1.6e-4, "quats": 1e-3, "scales": 5e-3, "opacities": 5e-2, "colors": 2.5e-3, "pose_quats": 1e-3, "pose_trans": 1e-3}

src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
mu = cache.load_metric_upgrade("out/frontend_cache", "scene-0061"); dep = cache.load_depth("out/frontend_cache", "scene-0061")
pidx = {t: i for i, t in enumerate(mu.tokens)}; K = kfs[0].calib.intrinsic
rec = pycolmap.Reconstruction(f"{SP}/colmap_van/sparse_tuned/0")
def cfw(im):
    rig = im.cam_from_world; rig = rig() if callable(rig) else rig
    try: M = np.asarray(rig.matrix())
    except Exception: M = np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
    T = np.eye(4); T[:3] = M; return T
info = {im.image_id: (int(im.name[1:3]), cfw(im)) for im in rec.images.values()}
mds = {fi: metric_depth(dep[kfs[fi].token].depth, dep[kfs[fi].token].intrinsic, dep[kfs[fi].token].extrinsic, mu.H, np.linalg.inv(mu.world_from_cam[pidx[kfs[fi].token]])) for iid, (fi, _) in info.items()}
ratios = []
for p in rec.points3D.values():
    X = np.append(p.xyz, 1.0)
    for el in p.track.elements:
        fi, C = info[el.image_id]; zc = (C @ X)[2]; u, v = rec.images[el.image_id].points2D[el.point2D_idx].xy
        if 0 <= int(v) < mds[fi].shape[0] and 0 <= int(u) < mds[fi].shape[1] and zc > 1e-6 and mds[fi][int(v), int(u)] > 1e-6:
            ratios.append(mds[fi][int(v), int(u)] / zc)
s = float(np.median(ratios))
order = sorted(info.items(), key=lambda kv: kv[1][0])
poses = np.stack([np.linalg.inv(np.block([[C[:3, :3], (C[:3, 3] * s)[:, None]], [np.zeros(3), 1]])) for _, (_, C) in order]).astype(np.float32)

g = np.load(f"{SP}/vehicle_gs.npz")
init_g = {k: g[k] for k in ("means", "scales", "quats", "opacities", "sh")}
images = [np.zeros((900, 1600, 3), np.uint8)] * len(poses)
_gs, render, _ = train_gaussians(np.zeros((10, 3), np.float32), np.zeros((10, 3), np.uint8),
                                 poses, K, images, np.arange(len(poses)), lr_for=LR, num_iters=0, init_gaussians=init_g)

def interp(P, k=6):
    t = np.arange(len(P)); fine = np.linspace(0, len(P) - 1, (len(P) - 1) * k + 1)
    Rf = Slerp(t, Rotation.from_matrix(P[:, :3, :3]))(fine).as_matrix()
    tf = np.stack([np.interp(fine, t, P[:, j, 3]) for j in range(3)], axis=1)
    out = np.tile(np.eye(4), (len(fine), 1, 1)); out[:, :3, :3] = Rf; out[:, :3, 3] = tf; return out
path = interp(poses, 6)
with imageio.get_writer("out/vehicle_gs_flythrough.mp4", fps=15, macro_block_size=None) as w:
    for P in path:
        r = render(P, K); r = r.detach().cpu().numpy() if torch.is_tensor(r) else np.asarray(r)
        w.append_data((np.clip(r, 0, 1) * 255).astype(np.uint8))
print(f"wrote out/vehicle_gs_flythrough.mp4 ({len(path)} frames)")

# rrd: splats + trajectory + frustums
sh = g["sh"]; dc = sh[:, 0, :] if sh.ndim == 3 else sh
rgba = (np.concatenate([np.clip(dc * C0 + 0.5, 0, 1), g["opacities"].reshape(-1, 1)], 1) * 255).astype(np.uint8)
q = g["quats"] / (np.linalg.norm(g["quats"], axis=1, keepdims=True) + 1e-9)
diag = float(np.linalg.norm(poses[:, :3, 3].max(0) - poses[:, :3, 3].min(0)))
rrlog.init("nuslam-vehicle-gs", save="out/vehicle_gs.rrd")
rrlog.log_gaussians("gs/van", g["means"], g["scales"], q, rgba)
rrlog.log_trajectory("gs/trajectory", poses[:, :3, 3], color=(230, 90, 90), radius=diag / 400, static=True)
for i, P in enumerate(poses):
    T = P.astype(np.float64); U, _, Vt = np.linalg.svd(T[:3, :3])          # float64 for pyquaternion's tolerance
    T[:3, :3] = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt   # nearest proper rotation (det=+1)
    rrlog.log_estimated_camera(f"gs/cam{i:02d}", T, K, 1600, 900, image_plane_distance=diag / 9, color=(230, 90, 90), static=True)
print(f"wrote out/vehicle_gs.rrd ({g['means'].shape[0]} splats, {len(poses)} frustums)")
