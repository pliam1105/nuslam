"""Vehicle rrds: COLMAP (usable) + DA3 masked-full (the OOD-but-consistent-intrinsics one). Each
up-to-scale recon normalized to diag 50 (physical-ish), Y-down (RDF), 0.003 points, frustums + trajectory."""
import numpy as np, pycolmap, rerun as rr
from nuslam.data import NuScenesMonoSource
from nuslam.viz import rerun_logging as rrlog
SP = "out/vehicle_cache"
DIAG, PR, TRAJ_R, IPD = 50.0, 0.03, 0.05, 5.0

def ortho(R): U, _, Vt = np.linalg.svd(np.asarray(R, float)); return U @ Vt

def build(name, save, pts, colors, poses, Ks, sizes, color):
    """poses: list of (center, cam->world R). Normalize to diag DIAG, log Y-down."""
    ctr = pts.mean(0); sc = float(np.linalg.norm(pts.max(0) - pts.min(0))) / DIAG
    rrlog.init(name, save=save)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)   # Right-Down-Forward -> Y axis down
    rrlog.log_points(f"{name}/points", (pts - ctr) / sc, colors=colors, radii=PR, static=True)
    rrlog.log_trajectory(f"{name}/trajectory", np.array([(c - ctr) / sc for c, _ in poses]),
                         color=color, radius=TRAJ_R, static=True)
    for i, ((c, R), K, (h, w)) in enumerate(zip(poses, Ks, sizes)):
        T = np.eye(4); T[:3, :3] = ortho(R); T[:3, 3] = (c - ctr) / sc
        rrlog.log_estimated_camera(f"{name}/f{i:02d}", T, K, int(w), int(h), image_plane_distance=IPD,
                                   color=color, static=True)
    print(f"{name}: {len(pts)} pts, {len(poses)} frustums (diag->{DIAG}, was {sc*DIAG:.2f})")

# ---- COLMAP ----
K = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT").load_scene("scene-0061")[0].calib.intrinsic
rec = pycolmap.Reconstruction(f"{SP}/colmap_van/sparse_tuned/0")
pts = np.array([p.xyz for p in rec.points3D.values()])
try: rgb = np.array([p.color for p in rec.points3D.values()])[:, :3].astype(np.uint8)
except Exception: rgb = np.full((len(pts), 3), 210, np.uint8)
ims = sorted(rec.images.values(), key=lambda i: i.name)
def cpose(im):
    c = np.asarray(im.projection_center()); rig = im.cam_from_world; rig = rig() if callable(rig) else rig
    try: M = np.asarray(rig.matrix())
    except Exception: M = np.hstack([np.asarray(rig.rotation.matrix()), np.asarray(rig.translation)[:, None]])
    return c, M[:3, :3].T
build("colmap", f"{SP}/vehicle_colmap.rrd", pts, rgb, [cpose(im) for im in ims],
      [K] * len(ims), [(900, 1600)] * len(ims), (230, 90, 90))

# ---- DA3 masked-full ----
d = np.load(f"{SP}/da3_maskedfull_cloud.npz"); P, C, Es, Ks, hws = d["P"], d["C"], d["Es"], d["Ks"], d["hws"]
u = (C - C.min()) / (C.ptp() + 1e-9); col = (np.stack([u, 0.3 + 0 * u, 1 - u], 1) * 255).astype(np.uint8)
build("da3masked", f"{SP}/vehicle_da3masked.rrd", P, col, [(E[:3, 3], E[:3, :3]) for E in Es],
      Ks, hws, (90, 160, 230))
