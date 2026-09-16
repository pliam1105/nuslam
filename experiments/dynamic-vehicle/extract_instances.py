"""Extract per-frame SAM3 vehicle instances and report counts/scores; overlay a few frames to eyeball."""
import numpy as np
from nuslam.data import NuScenesMonoSource
from nuslam.frontend.segmentation import Sam3Segmenter, SegConfig

src = NuScenesMonoSource("data/nuscenes", "v1.0-mini", camera="CAM_FRONT")
kfs = src.load_scene("scene-0061")
seg = Sam3Segmenter(SegConfig(prompt="moving vehicle", threshold=0.5))
inst = seg.segment_instances(kfs)
seg.release()

print(f"{len(inst)} frames")
for i, (tok, masks, scores) in enumerate(inst):
    areas = [int(m.sum()) for m in masks]
    print(f"frame {i:2d}: {len(masks)} instances  scores={[round(float(s),2) for s in scores]}  areas={areas}")

np.save("out/vehicle_cache/inst.npy",
        np.array([(t, m, s) for t, m, s in inst], dtype=object), allow_pickle=True)

# overlay frames 0, 12, 24 with per-instance colors + centroids
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rng = np.random.default_rng(0); cols = rng.uniform(0.3, 1.0, (40, 3))
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
for a, fi in zip(ax, (0, 12, 24)):
    tok, masks, scores = inst[fi]
    img = np.asarray(kfs[fi].image()).astype(float) / 255
    ov = img.copy()
    for j, m in enumerate(masks):
        ov[m] = 0.5 * ov[m] + 0.5 * cols[j]
        ys, xs = np.where(m); a.plot(xs.mean(), ys.mean(), "o", color="k", ms=5)
    a.imshow(ov); a.set_title(f"frame {fi}: {len(masks)} instances"); a.axis("off")
fig.tight_layout(); fig.savefig("out/vehicle_cache/instances.png", dpi=90, bbox_inches="tight")
print("wrote instances.png")
