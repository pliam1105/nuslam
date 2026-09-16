"""Van-frame rrd: the COLMAP camera poses (SCALED) + the van's GLOBAL metric depth (mask only) brought
into the COLMAP van frame via those scaled poses. No scene, no ego poses, no COLMAP points. If the
scale + poses are right, the per-frame van-depth surfaces align into one coherent van."""
import numpy as np, pycolmap, rerun as rr
from nuslam.data import NuScenesMonoSource
from nuslam.frontend import cache
from nuslam.recon import metric_depth
from nuslam.viz import rerun_logging as rrlog
import matplotlib.cm as cm
SP = "out/vehicle_cache"
DIAG, PR, IPD = 50.0, 0.03, 4.0

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
info = {}
for im in rec.images.values():
    fi = int(im.name[1:3]); kf = kfs[fi]
    md = metric_depth(dep[kf.token].depth, dep[kf.token].intrinsic, dep[kf.token].extrinsic, mu.H, np.linalg.inv(mu.world_from_cam[pidx[kf.token]]))
    info[im.image_id] = (fi, cfw(im), md)

# scale s = median(z_global/z_colmap) over COLMAP observations
ratios = []
for p in rec.points3D.values():
    X = np.append(p.xyz, 1.0)
    for el in p.track.elements:
        fi, C, md = info[el.image_id]; zc = (C @ X)[2]
        u, v = rec.images[el.image_id].points2D[el.point2D_idx].xy
        if 0 <= int(v) < md.shape[0] and 0 <= int(u) < md.shape[1] and zc > 1e-6 and md[int(v), int(u)] > 1e-6:
            ratios.append(md[int(v), int(u)] / zc)
s = float(np.median(ratios)); print(f"scale s = {s:.4f} ({len(ratios)} obs)")

# build in the scaled van frame
vp, vc, fposes = [], [], []
for iid, (fi, C, md) in sorted(info.items(), key=lambda kv: kv[1][0]):
    Cs = C.copy(); Cs[:3, 3] *= s                       # scale COLMAP translation -> metric van frame
    Wv = np.linalg.inv(Cs)                              # cam->world (van frame), scaled
    fposes.append((fi, Wv))
    m = bt[fi] & (md > 1e-6); vv, uu = np.where(m); sub = slice(None, None, max(1, len(vv) // 1200)); vv, uu = vv[sub], uu[sub]
    z = md[vv, uu]; pcam = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
    pv = (Wv[:3, :3] @ pcam + Wv[:3, 3:4]).T
    vp.append(pv); vc.append(np.tile((np.array(cm.viridis(fi / 38)[:3]) * 255).astype(np.uint8), (len(pv), 1)))
VP = np.vstack(vp); VC = np.vstack(vc)
lo, hi = np.percentile(VP, [3, 97], axis=0)             # drop DA3-depth outliers (noisy on the van surface)
keep = np.all((VP >= lo) & (VP <= hi), axis=1); VP, VC = VP[keep], VC[keep]
allc = np.array([w[:3, 3] for _, w in fposes])          # camera centres, for normalization extent
ctr = np.vstack([VP, allc]).mean(0); sc = float(np.linalg.norm(np.vstack([VP, allc]).max(0) - np.vstack([VP, allc]).min(0))) / DIAG

rrlog.init("nuslam-vehicle-scaledposes", save=f"{SP}/vehicle_scaledposes.rrd")
rr.log("world", rr.ViewCoordinates.RDF, static=True)
rrlog.log_points("van/global_depth", (VP - ctr) / sc, colors=VC, radii=PR, static=True)
rrlog.log_trajectory("cam/trajectory", (allc - ctr) / sc, color=(230, 90, 90), radius=0.05, static=True)
for fi, Wv in fposes:
    T = Wv.copy(); T[:3, 3] = (Wv[:3, 3] - ctr) / sc
    rrlog.log_estimated_camera(f"cam/f{fi:02d}", T, K, 1600, 900, image_plane_distance=IPD, color=(230, 90, 90), static=True)
print(f"wrote vehicle_scaledposes.rrd: van {len(VP)} pts (mask-only global depth), {len(fposes)} COLMAP frustums")

# alignment check: per-frame van-depth centroid spread (small = frames align into one van)
cents = np.array([p.mean(0) for p in vp]); spread = np.linalg.norm(cents - cents.mean(0), axis=1)
print(f"per-frame van centroid spread: median {np.median(spread)*s*0 + np.median(spread):.3f}, max {spread.max():.3f} (mu units); van size ~{np.linalg.norm(VP.max(0)-VP.min(0)):.3f}")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
for a, (i, j, t) in zip(ax, [(0, 2, "top-ish (X,Z)"), (0, 1, "front (X,Y)")]):
    a.scatter(VP[:, i], VP[:, j], c=VC / 255, s=3, alpha=0.4); a.set_title(t); a.set_aspect("equal"); a.grid(alpha=0.3)
    if j == 1: a.invert_yaxis()
fig.suptitle("van-only global depth in the scaled COLMAP frame, colored by frame (overlap=aligned)")
fig.savefig(f"{SP}/scaledposes_check.png", dpi=90, bbox_inches="tight"); print("wrote scaledposes_check.png")
