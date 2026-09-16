"""DA3 on MASKED FULL FRAMES: van kept at its true full-frame size/position, background blacked.
True apparent-size cue (correct far->near distance) + object-centric (static van, moving camera).
Contrast with the crop version (collapsed distance) and the global recon (correct distance but smeared van)."""
import types as pytypes, numpy as np
from nuslam.data import NuScenesMonoSource
from nuslam.frontend.depth import DA3ReconEstimator

SP = "out/vehicle_cache"
MIN_AREA = 6000
src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
bt = dict((int(f), m) for f, m in np.load(f"{SP}/best_track.npy", allow_pickle=True))

class KF:  # shim: estimate_scene uses token, calib.h/w, image()
    def __init__(self, tok, img): self._img = img; h, w = img.shape[:2]; self.token = tok; self.calib = pytypes.SimpleNamespace(height=h, width=w)
    def image(self): return self._img

shims, meta = [], []
for fi in sorted(bt):
    m = bt[fi]
    if m.sum() < MIN_AREA: continue
    img = np.asarray(kfs[fi].image()).copy(); img[~m] = 0          # full frame, background blacked
    shims.append(KF(f"mf{fi}", img)); meta.append((fi, m))
print(f"{len(shims)} masked full frames (900x1600), van at true size")

est = DA3ReconEstimator(); dmaps = est.estimate_scene(shims); est._model = None
import torch; torch.cuda.empty_cache()

allpts, allcol, cams, dist, Es, Ks, hws = [], [], [], [], [], [], []
for (fi, mask), dm in zip(meta, dmaps):
    d, K, E = dm.depth, dm.intrinsic, dm.extrinsic; Rt = np.linalg.inv(E); cams.append(Rt[:3, 3])
    Es.append(Rt); Ks.append(K); hws.append(d.shape[:2])
    ys, xs = np.where(mask & (d > 0)); sub = slice(None, None, max(1, len(ys) // 1500)); ys, xs = ys[sub], xs[sub]
    z = d[ys, xs]; rays = (np.linalg.inv(K) @ np.stack([xs, ys, np.ones_like(xs)])).T * z[:, None]
    pw = (Rt[:3, :3] @ rays.T + Rt[:3, 3:4]).T
    allpts.append(pw); allcol.append(np.full(len(pw), fi)); dist.append(np.linalg.norm(pw - Rt[:3, 3], axis=1).mean())
P = np.vstack(allpts); C = np.concatenate(allcol); cams = np.array(cams); fis = [m[0] for m in meta]
np.savez(f"{SP}/da3_maskedfull_cloud.npz", P=P, C=C, cams=cams,
         Es=np.array(Es), Ks=np.array(Ks), hws=np.array(hws), fis=np.array(fis))
dn = np.array(dist) / dist[0]
print(f"masked-full far/near ratio: {dn.max()/dn.min():.1f}x  (full-frame global 5.9x, crop 2.3x)")
print("per-frame dist(norm to f{}):".format(fis[0]), [round(float(x),2) for x in dn])
print(f"van cloud: {len(P)} pts  extent {(P.max(0)-P.min(0)).round(3)}  cam-path extent {(cams.max(0)-cams.min(0)).round(3)}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
for a, ((i, j), t) in zip(ax, [((0, 1), "front (X,Y)"), ((0, 2), "top-down (X,Z)"), ((2, 1), "side (Z,Y)")]):
    a.scatter(P[:, i], P[:, j], c=C, cmap="viridis", s=1, alpha=0.25)
    a.scatter(cams[:, i], cams[:, j], c="red", s=30, marker="^")
    a.set_title(t); a.set_aspect("equal"); a.grid(alpha=0.2)
    if j == 1: a.invert_yaxis()
fig.suptitle(f"DA3 masked-full-frame: van cloud ({len(P)} pts, {len(shims)} views) + cameras")
fig.tight_layout(); fig.savefig(f"{SP}/da3_maskedfull.png", dpi=88, bbox_inches="tight"); print("wrote da3_maskedfull.png")
