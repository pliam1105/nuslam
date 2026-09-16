"""Scale the COLMAP van to the global scene via the depth anchor (median z_global/z_colmap over the
COLMAP points' own observations), then compose: global scene metric cloud + the scaled van placed at
each frame's world position + ego frustums. All in the global (mu) metric frame."""
import numpy as np, pycolmap
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache
from nuslam.recon import metric_depth
from nuslam.viz import rerun_logging as rrlog
SP = "out/vehicle_cache"
ANCHOR = 19.962  # global ground-anchor scale (mu-units -> real metres), for reporting only

src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
mu = cache.load_metric_upgrade("out/frontend_cache", "scene-0061"); dep = cache.load_depth("out/frontend_cache", "scene-0061")
pidx = {t: i for i, t in enumerate(mu.tokens)}
K = kfs[0].calib.intrinsic
rec = pycolmap.Reconstruction(f"{SP}/colmap_van/sparse_tuned/0")

# per COLMAP image: frame index, cam_from_world (4x4, van frame), world_from_cam (global metric, mu)
def cfw(im):
    rig = im.cam_from_world; rig = rig() if callable(rig) else rig
    try: M = np.asarray(rig.matrix())
    except Exception: M = np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
    T = np.eye(4); T[:3] = M; return T
info = {}  # image_id -> (fi, cfw, world_from_cam_global, metric_depth)
for im in rec.images.values():
    fi = int(im.name[1:3]); kf = kfs[fi]; wi = pidx[kf.token]
    md = metric_depth(dep[kf.token].depth, dep[kf.token].intrinsic, dep[kf.token].extrinsic, mu.H, np.linalg.inv(mu.world_from_cam[wi]))
    info[im.image_id] = (fi, cfw(im), mu.world_from_cam[wi], md)

# ---- scale via the anchor: median(z_global / z_colmap) over point observations ----
ratios = []
for p in rec.points3D.values():
    X = np.append(p.xyz, 1.0)
    for el in p.track.elements:
        fi, C, _, md = info[el.image_id]
        zc = (C @ X)[2]
        u, v = rec.images[el.image_id].points2D[el.point2D_idx].xy
        if 0 <= int(v) < md.shape[0] and 0 <= int(u) < md.shape[1]:
            zg = md[int(v), int(u)]
            if zc > 1e-6 and zg > 1e-6: ratios.append(zg / zc)
s = float(np.median(ratios))
Xall0 = np.array([p.xyz for p in rec.points3D.values()])
lo, hi = np.percentile(Xall0, [2, 98], axis=0)                 # drop spurious background points
Xall = Xall0[np.all((Xall0 >= lo) & (Xall0 <= hi), axis=1)]
dims = (Xall.max(0) - Xall.min(0)) * s * ANCHOR  # real metres
print(f"scale s = {s:.4f}  ({len(ratios)} obs)   van bbox ~ {dims.round(1)} m (real), kept {len(Xall)}/{len(Xall0)} pts")

# ---- global scene metric cloud (mu frame): back-project global depth, RGB ----
gp, gc = [], []
for iid, (fi, C, W, md) in sorted(info.items(), key=lambda kv: kv[1][0]):
    img = np.asarray(kfs[fi].image()); H, Wd = md.shape
    vv, uu = np.mgrid[0:H:12, 0:Wd:12]; z = md[vv, uu]; m = z > 1e-6
    uu, vv, z = uu[m], vv[m], z[m]
    Xc = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
    Xw = (W[:3, :3] @ Xc + W[:3, 3:4]).T
    gp.append(Xw); gc.append(img[vv, uu])
GP = np.vstack(gp); GC = np.vstack(gc).astype(np.uint8)

# ---- place the scaled van at each frame's world position, color by frame ----
vp, vc = [], []
import matplotlib.cm as cm
for iid, (fi, C, W, md) in sorted(info.items(), key=lambda kv: kv[1][0]):
    Xc = (C[:3, :3] @ Xall.T + C[:3, 3:4]) * s          # metric cam coords (mu frame)
    Xw = (W[:3, :3] @ Xc + W[:3, 3:4]).T
    vp.append(Xw); vc.append(np.tile((np.array(cm.viridis(fi / 38)[:3]) * 255).astype(np.uint8), (len(Xw), 1)))
VP = np.vstack(vp); VC = np.vstack(vc)

rrlog.init("nuslam-vehicle-composed", save=f"{SP}/vehicle_composed.rrd")
rrlog.log_points("scene/global", GP, colors=GC, radii=0.01, static=True)
rrlog.log_points("van/placed", VP, colors=VC, radii=0.02, static=True)
for iid, (fi, C, W, md) in sorted(info.items(), key=lambda kv: kv[1][0]):
    rrlog.log_estimated_camera(f"ego/f{fi:02d}", W, K, 1600, 900, image_plane_distance=0.15, color=(230, 200, 80), static=True)
print(f"wrote vehicle_composed.rrd: global {len(GP)} pts, van {len(VP)} pts ({len(info)} placements)")

# ---- 2D top-down sanity (X-Z, camera looks along +Z): vans should sit ahead of the ego along the road ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
egoc = np.array([W[:3, 3] for _, (_, _, W, _) in sorted(info.items(), key=lambda kv: kv[1][0])])
sub = np.random.default_rng(0).choice(len(GP), min(40000, len(GP)), replace=False)
fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(GP[sub, 0], GP[sub, 2], s=1, c="#bbb", alpha=0.4, label="global scene")
ax.scatter(VP[:, 0], VP[:, 2], s=3, c=VC / 255, label="van (per frame)")
ax.plot(egoc[:, 0], egoc[:, 2], "-^", c="red", ms=4, label="ego")
ax.set_xlabel("X (mu units)"); ax.set_ylabel("Z (mu units)"); ax.legend(); ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title("composed: scaled COLMAP van (color=frame) in the global scene, top-down (X-Z)")
fig.savefig(f"{SP}/composed_topdown.png", dpi=90, bbox_inches="tight"); print("wrote composed_topdown.png")
