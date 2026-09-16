"""Experiment B: DA3-Base multi-view reconstruction on the tracked van's per-frame crops.
Feeds bbox+margin crops (DA3 is intrinsic-agnostic -> estimates its own K per crop), back-projects the
masked depth into DA3's shared frame, and checks whether it forms a coherent object + sane camera orbit."""
import types as pytypes, numpy as np
from nuslam.data import NuScenesMonoSource
from nuslam.frontend.depth import DA3ReconEstimator

SP = "out/vehicle_cache"
MARGIN = 0.20; MIN_AREA = 6000
src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
bt = dict((int(f), m) for f, m in np.load(f"{SP}/best_track.npy", allow_pickle=True))

class CropKF:  # minimal shim: estimate_scene only touches token, calib.h/w, image()
    def __init__(self, token, img):
        self._img = img; h, w = img.shape[:2]
        self.token = token; self.calib = pytypes.SimpleNamespace(height=h, width=w)
    def image(self): return self._img

crops, meta = [], []
for fi in sorted(bt):
    m = bt[fi]
    if m.sum() < MIN_AREA: continue
    ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    dy, dx = int((y1 - y0) * MARGIN), int((x1 - x0) * MARGIN)
    H, W = m.shape; y0, y1 = max(0, y0 - dy), min(H, y1 + dy); x0, x1 = max(0, x0 - dx), min(W, x1 + dx)
    img = np.asarray(kfs[fi].image())[y0:y1, x0:x1]
    crops.append(CropKF(f"crop{fi}", img)); meta.append((fi, m[y0:y1, x0:x1], (y0, x0)))
print(f"{len(crops)} crops (area>={MIN_AREA}), sizes {[c._img.shape[:2] for c in crops[:4]]}...")

est = DA3ReconEstimator()
dmaps = est.estimate_scene(crops); est._model = None
import torch; torch.cuda.empty_cache()

# back-project masked depth into DA3's shared frame (cam0 = origin)
allpts, allcol, cams, Es, Ks, hws = [], [], [], [], [], []
for (fi, mask_c, _), dm in zip(meta, dmaps):
    d = dm.depth; K = dm.intrinsic; E = dm.extrinsic  # world->cam, cam0=I
    Rt = np.linalg.inv(E); cams.append(Rt[:3, 3]); Es.append(Rt); Ks.append(K); hws.append(d.shape[:2])
    hh, ww = d.shape; ys, xs = np.where(mask_c & (d > 0))
    sub = slice(None, None, max(1, len(ys) // 1500)); ys, xs = ys[sub], xs[sub]
    z = d[ys, xs]; rays = (np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)])).T * z[:, None]
    pw = (Rt[:3, :3] @ rays.T + Rt[:3, 3:4]).T
    allpts.append(pw); allcol.append(np.full(len(pw), fi))
    print(f"  f{fi:2d}: crop {d.shape}  K_da3 fx={K[0,0]:6.1f}  depth[{z.min():.2f},{z.max():.2f}]  cam_t={Rt[:3,3].round(2)}")
P = np.vstack(allpts); C = np.concatenate(allcol); cams = np.array(cams)
print(f"van cloud: {len(P)} pts  extent {P.max(0)-P.min(0)}")

np.savez(f"{SP}/da3_cloud.npz", P=P, C=C, cams=cams,
         Es=np.array(Es), Ks=np.array(Ks), hws=np.array(hws), fis=np.array([m[0] for m in meta]))
import matplotlib; matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 5))
for k, (el, az) in enumerate([(20, -60), (89, -90), (10, 0)]):
    ax = fig.add_subplot(1, 3, k + 1, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, cmap="viridis", s=1, alpha=0.3)
    ax.scatter(cams[:, 0], cams[:, 1], cams[:, 2], c="red", s=25, marker="^")
    ax.view_init(elev=el, azim=az); ax.set_title(["oblique", "top-down", "front"][k]); ax.set_box_aspect((1,1,1))
fig.suptitle(f"DA3-on-crops: van cloud ({len(P)} pts, {len(crops)} views) + cameras (red)")
fig.tight_layout(); fig.savefig(f"{SP}/da3_vehicle.png", dpi=85, bbox_inches="tight")
print("wrote da3_vehicle.png")
