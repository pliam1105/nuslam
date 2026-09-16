"""Greedy nearest-centroid frame-to-frame association of SAM3 vehicle instances into tracks.
Spec: order current-frame instances by centroid distance to each active track, assign with
exclusivity, RECOMPUTE after every match (global greedy min-pair), reject beyond a pixel threshold.
(Future: optical-flow-predicted centroid, OpenCV trackers, Hungarian.)"""
import numpy as np
from nuslam.data import NuScenesMonoSource

SP = "out/vehicle_cache"
MIN_AREA = 800        # drop spurious tiny detections
PIX_THRESH = 400.0    # max centroid jump between keyframes (2 Hz -> large motion)
MAX_GAP = 2           # frames a track may go unmatched before it can't be extended

inst = np.load(f"{SP}/inst.npy", allow_pickle=True)
def centroid(m): ys, xs = np.where(m); return np.array([xs.mean(), ys.mean()])
# per frame: list of (mask, centroid, score, area) filtered by area
frames = []
for tok, masks, scores in inst:
    dets = [(m, centroid(m), float(s), int(m.sum())) for m, s in zip(masks, scores) if m.sum() >= MIN_AREA]
    frames.append(dets)

tracks = []  # each: dict(id, dets={fi:det}, last_fi, last_c)
for fi, dets in enumerate(frames):
    active = [t for t in tracks if fi - t["last_fi"] <= MAX_GAP]
    avail = list(range(len(dets)))
    if active and avail:
        D = np.array([[np.linalg.norm(t["last_c"] - dets[j][1]) for j in avail] for t in active])
        used_t, used_j = set(), set()
        while True:
            D2 = D.copy()
            for ti in used_t: D2[ti, :] = np.inf
            for jj in used_j: D2[:, jj] = np.inf
            ti, jc = np.unravel_index(np.argmin(D2), D2.shape)
            if not np.isfinite(D2[ti, jc]) or D2[ti, jc] > PIX_THRESH: break
            t = active[ti]; j = avail[jc]
            t["dets"][fi] = dets[j]; t["last_fi"] = fi; t["last_c"] = dets[j][1]
            used_t.add(ti); used_j.add(jc)
        matched_j = {avail[jc] for jc in used_j}
    else:
        matched_j = set()
    for j, det in enumerate(dets):        # unmatched dets -> new tracks
        if j not in matched_j:
            tracks.append({"id": len(tracks), "dets": {fi: det}, "last_fi": fi, "last_c": det[1]})

tracks.sort(key=lambda t: -len(t["dets"]))
print(f"{len(tracks)} tracks (min_area={MIN_AREA}, pix_thresh={PIX_THRESH}, max_gap={MAX_GAP})")
for t in tracks[:12]:
    fis = sorted(t["dets"]); areas = [t["dets"][f][3] for f in fis]
    print(f"  track {t['id']:2d}: {len(fis):2d} frames  span [{fis[0]}..{fis[-1]}]  "
          f"mean_area {int(np.mean(areas)):6d}  frames={fis}")

best = tracks[0]
np.save(f"{SP}/best_track.npy", np.array([(f, best['dets'][f][0]) for f in sorted(best['dets'])], dtype=object), allow_pickle=True)

# montage of the best track's mask across its frames
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT"); kfs = src.load_scene("scene-0061")
fis = sorted(best["dets"]); n = len(fis); cols = 6; rows = (n + cols - 1) // cols
fig, ax = plt.subplots(rows, cols, figsize=(cols * 3, rows * 1.9))
for a in np.ravel(ax): a.axis("off")
for k, fi in enumerate(fis):
    a = np.ravel(ax)[k]; img = np.asarray(kfs[fi].image()).astype(float) / 255
    m = best["dets"][fi][0]; ov = img.copy(); ov[m] = 0.4 * ov[m] + 0.6 * np.array([0.1, 0.9, 0.4])
    a.imshow(ov); a.set_title(f"f{fi}", fontsize=8)
fig.suptitle(f"best track (id {best['id']}): {n} frames", fontsize=11)
fig.tight_layout(); fig.savefig(f"{SP}/best_track.png", dpi=80, bbox_inches="tight")
print(f"wrote best_track.png ({n} frames)")
