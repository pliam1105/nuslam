"""Verify the tuned-COLMAP van reconstruction (does it get far->near right, unlike DA3-crop?) and write
rrds for both the COLMAP and DA3-crop van reconstructions."""
import numpy as np, pycolmap
from pathlib import Path
from nuslam.viz import rerun_logging as rrlog
SP = "out/vehicle_cache"

# ---- COLMAP: extract cameras + points, check per-frame distance ----
rec = pycolmap.Reconstruction(f"{SP}/colmap_van/sparse_tuned/0")
imgs = sorted(rec.images.values(), key=lambda im: im.name)
cams, fnums = [], []
for im in imgs:
    try: c = np.asarray(im.projection_center())
    except Exception:
        rig = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        Rm = rig.rotation.matrix(); c = -Rm.T @ np.asarray(rig.translation)
    cams.append(c); fnums.append(int(im.name[1:3]))
cams = np.array(cams)
pts = np.array([p.xyz for p in rec.points3D.values()]); ctr = pts.mean(0)
dist = np.linalg.norm(cams - ctr, axis=1); dn = dist / dist[0]
print(f"COLMAP van: {len(cams)} cams, {len(pts)} pts")
print(f"  far/near ratio: {dn.max()/dn.min():.1f}x   (truth/global 5.9x; DA3-crop 2.3x, DA3-masked 2.0x)")
order = np.argsort(fnums)
print("  per-frame dist(norm):", [round(float(x),2) for x in dn[order]])

# ---- rrd: COLMAP ----
rrlog.init("nuslam-vehicle-colmap", save=f"{SP}/vehicle_colmap.rrd")
rrlog.log_trajectory("colmap/cameras", cams[order], color=(230, 80, 80), static=True)
try:
    rgb = np.array([p.color for p in rec.points3D.values()])[:, :3].astype(np.uint8)
except Exception:
    rgb = np.full((len(pts), 3), 200, np.uint8)
rrlog.log_points("colmap/points", pts, colors=rgb, static=True)
print(f"wrote vehicle_colmap.rrd")

# ---- rrd: DA3 crop (for contrast) ----
d = np.load(f"{SP}/da3_cloud.npz"); P, C, dcams = d["P"], d["C"], d["cams"]
rrlog.init("nuslam-vehicle-da3crop", save=f"{SP}/vehicle_da3crop.rrd")
rrlog.log_trajectory("da3/cameras", dcams, color=(80, 160, 230), static=True)
col = (np.stack([(C - C.min()) / (C.ptp() + 1e-9), np.zeros_like(C), 1 - (C - C.min()) / (C.ptp() + 1e-9)], 1) * 255).astype(np.uint8)
rrlog.log_points("da3/points", P, colors=col, static=True)
print(f"wrote vehicle_da3crop.rrd")
