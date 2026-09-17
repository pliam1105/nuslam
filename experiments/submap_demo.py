"""§14 submap demonstration (DA3-metric route, NO ground anchor). Split scene-0061 into 5 OVERLAPPING
submaps. Each submap is made metric ON ITS OWN by the COLMAP->DA3 median depth ratio -- s_m = median(z_da3 /
z_colmap) over that window's COLMAP point-observations -- applied to the raw (up-to-scale) COLMAP poses; its
map is the DA3 metric point cloud (per-frame DA3 depth back-projected). Adjacent submaps are then aligned
through their SHARED cameras and assembled INCREMENTALLY. The overlap-implied inter-submap scale is the §14.2
cross-check (should agree; disagreement localizes a bad window). The DA3+DAQ metric upgrade and the factor
graph/iSAM2 that would replace this greedy assembly are author §3."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pycolmap
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache
from nuslam.transforms import umeyama

CACHE, SPARSE = "out/frontend_cache", "out/colmap/sparse/0"
src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
mu = cache.load_metric_upgrade(CACHE, "scene-0061"); dep = cache.load_depth(CACHE, "scene-0061")
pidx = {t: i for i, t in enumerate(mu.tokens)}; frames = [kf for kf in kfs if kf.token in dep and kf.token in pidx]
N = len(frames); toks = [kf.token for kf in frames]; tset = set(toks)
K = np.asarray(mu.K_true, np.float64); Kinv = np.linalg.inv(K)
s2e = frames[0].calib.sensor2ego.matrix()
GT = np.stack([kf.ego2global_gt.matrix() @ s2e for kf in frames])

# --- raw COLMAP: per-frame up-to-scale cam->world poses, and per-point (token, z_colmap, z_da3) obs ---
rec = pycolmap.Reconstruction(SPARSE)
def img_tok(im):
    n = im.name.rsplit(".", 1)[0]
    return n if n in pidx else (toks[int(n[1:])] if n[1:].isdigit() and int(n[1:]) < N else None)
raw_pose = {}                                            # token -> (4,4) cam->world, RAW colmap scale
for im in rec.images.values():
    tok = img_tok(im)
    if tok is None or tok not in tset: continue
    rig = im.cam_from_world; rig = rig() if callable(rig) else rig
    M = np.asarray(rig.matrix()) if hasattr(rig, "matrix") else np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
    R, t = M[:3, :3], M[:3, 3]
    P = np.eye(4); P[:3, :3] = R.T; P[:3, 3] = -R.T @ t   # world->cam (R,t) => cam->world
    raw_pose[tok] = P
obs = {t: [] for t in toks}                              # token -> list of (z_colmap_raw, z_da3_metric)
for p in rec.points3D.values():
    X = np.append(p.xyz, 1.0)
    for el in p.track.elements:
        im = rec.images[el.image_id]; tok = img_tok(im)
        if tok is None or tok not in tset: continue
        rig = im.cam_from_world; rig = rig() if callable(rig) else rig
        M = np.asarray(rig.matrix()) if hasattr(rig, "matrix") else np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
        zc = (M @ X)[2]; u, v = im.points2D[el.point2D_idx].xy; dd = dep[tok].depth
        if zc > 1e-6 and 0 <= int(v) < dd.shape[0] and 0 <= int(u) < dd.shape[1]:
            zg = dd[int(v), int(u)]
            if zg > 1e-6: obs[tok].append((zc, zg))
raw_pose = {t: raw_pose[t] for t in toks if t in raw_pose}
assert len(raw_pose) == N, f"missing raw COLMAP poses: {N - len(raw_pose)}"

def da3_cloud(sub_tokens, poses, stride=12):
    """DA3 metric point cloud for a submap: back-project per-frame DA3 depth through the metric poses."""
    pts = []
    for tok in sub_tokens:
        d = dep[tok].depth; H, W = d.shape
        vv, uu = np.mgrid[0:H:stride, 0:W:stride].reshape(2, -1); z = d[vv, uu]
        keep = (z > 0) & (z < np.percentile(z[z > 0], 80)); vv, uu, z = vv[keep], uu[keep], z[keep]  # drop sky/far
        Xc = (Kinv @ np.stack([uu, vv, np.ones_like(uu)], 0)) * z                 # (3,Nk) camera-frame metric
        P = poses[tok]; pts.append((P[:3, :3] @ Xc + P[:3, 3:4]).T)
    return np.concatenate(pts, 0) if pts else np.zeros((0, 3))

def metric_submap(sub):
    """Make a submap metric via the COLMAP->DA3 median depth ratio; return metric cam->world poses + s_m + cloud."""
    tk = [toks[i] for i in sub]
    ratios = np.array([zg / zc for t in tk for (zc, zg) in obs[t]])               # z_da3 / z_colmap
    s_m = float(np.median(ratios))                                                # meters per colmap-unit
    poses = {}
    for t in tk:
        P = raw_pose[t].copy(); P[:3, 3] = s_m * P[:3, 3]                          # scale world (centres) to metric
        poses[t] = P
    cloud = da3_cloud(tk, poses)
    return poses, s_m, ratios.size, cloud

def apply_sim3(s, R, t, P):
    Q = P.copy(); Q[:3, :3] = R @ P[:3, :3]; Q[:3, 3] = s * (R @ P[:3, 3]) + t; return Q

# 5 overlapping windows (size 11, step 7) covering 0..38 -> 4-camera overlaps
subs = [list(range(a, min(a + 11, N))) for a in [0, 7, 14, 21, 28]]
print("submaps (frame spans):", [(s[0], s[-1]) for s in subs])
local, scales, clouds = [], [], []
for m, sub in enumerate(subs):
    poses, s_m, n, cloud = metric_submap(sub)
    local.append(poses); scales.append(s_m); clouds.append(cloud)
    print(f"  submap {m}: frames {sub[0]}..{sub[-1]} ({len(sub)}), COLMAP->DA3 median s_m={s_m:.4f} m/unit "
          f"(n={n} obs), DA3 cloud {cloud.shape[0]} pts")

# incremental assembly through shared cameras. Each submap is already metric -> compose RIGIDLY (SE3);
# the overlap-implied Sim3 scale is kept as the §14.2 cross-check diagnostic only.
def assemble(use_scale):
    world = {toks[i]: local[0][toks[i]] for i in subs[0]}; a_scales = []
    for m in range(1, len(subs)):
        sh = [toks[i] for i in sorted(set(subs[m]) & set(subs[m - 1]))]
        A = np.stack([world[t][:3, 3] for t in sh]); B = np.stack([local[m][t][:3, 3] for t in sh])
        s, R, t = umeyama(B, A, with_scale=True); a_scales.append(s)              # new -> world
        s_use = s if use_scale else 1.0
        for i in subs[m]: world[toks[i]] = apply_sim3(s_use, R, t, local[m][toks[i]])
    return world, a_scales

def ate(world):
    # no ground anchor => the whole assembly is Euclidean up-to-scale (shared DA3/DAQ frame);
    # score shape against GT with a Sim3 (scale-free) fit. Absolute metric scale is what the
    # dropped ground anchor / GPS / a §14 depth-ratio prior would fix.
    asm = np.stack([world[t][:3, 3] for t in toks])
    sg, Rg, tg = umeyama(asm, GT[:, :3, 3], with_scale=True)
    e = np.linalg.norm((sg * (asm @ Rg.T) + tg) - GT[:, :3, 3], axis=1); return e, sg, Rg, tg

W_rigid, align_scales = assemble(False); W_sim3, _ = assemble(True)
e_rigid, sg, Rg, tg = ate(W_rigid); e_sim3, _, _, _ = ate(W_sim3)
print("\n§14.2 cross-check -- overlap-implied inter-submap scale (should be ~1 if the medians agree):")
print("  ", [round(x, 3) for x in align_scales], " <- flags submap 0 (its median is the outlier)")
print(f"per-submap DA3 scale s_m: {[round(x,3) for x in scales]}  (spread {max(scales)/min(scales):.3f}x)")
print(f"incremental assembly vs GT:  RIGID(SE3) mean {e_rigid.mean():.2f} med {np.median(e_rigid):.2f} max {e_rigid.max():.2f} m"
      f"   |   SIM3 mean {e_sim3.mean():.2f} med {np.median(e_sim3):.2f} max {e_sim3.max():.2f} m")

# plot: rigid-assembled submap trajectories + DA3 metric clouds, aligned to GT, local frame at GT[0]
cols = plt.cm.tab10(np.arange(len(subs))); o = GT[0, :3, 3]
def to_gt(X): return (sg * (X @ Rg.T) + tg) - o    # Sim3 into GT frame (scale-free scoring), local at GT[0]
gt = GT[:, :3, 3] - o
fig, ax = plt.subplots(figsize=(7.5, 9))
ax.plot(gt[:, 0], gt[:, 1], "-", color="k", lw=4, alpha=0.3, label="GT ego")
for m in range(len(subs)):
    cl = to_gt(clouds[m]); ax.scatter(cl[:, 0], cl[:, 1], s=1, color=cols[m], alpha=0.06, linewidths=0)
for m in range(len(subs)):    # SIM3-assembled trajectory (the sound composition here: overlaps carry the scale)
    c = to_gt(np.stack([W_sim3[toks[i]][:3, 3] for i in subs[m]]))
    ax.plot(c[:, 0], c[:, 1], "-o", color=cols[m], ms=4, lw=1.6, label=f"submap {m}: fr {subs[m][0]}-{subs[m][-1]}, s={scales[m]:.2f}")
for m in range(1, len(subs)):
    sh = [toks[i] for i in sorted(set(subs[m]) & set(subs[m - 1]))]
    c = to_gt(np.stack([W_sim3[t][:3, 3] for t in sh]))
    ax.plot(c[:, 0], c[:, 1], "s", mfc="none", mec="k", ms=11, mew=1.3, label="shared cams" if m == 1 else None)
pad = 12; ax.set_xlim(gt[:, 0].min() - pad, gt[:, 0].max() + pad); ax.set_ylim(gt[:, 1].min() - pad, gt[:, 1].max() + pad)
ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"5 submaps, each scaled by its COLMAP->DA3 median depth ratio (no ground anchor);\n"
             f"DA3 clouds + poses assembled via shared cameras (Sim3). ATE vs GT: mean {e_sim3.mean():.2f} m")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
fig.tight_layout(); fig.savefig("out/submap_demo.png", dpi=110, bbox_inches="tight"); print("wrote out/submap_demo.png")
