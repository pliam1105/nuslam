# Dynamic-vehicle experiments

Exploratory / visualization scripts for the per-instance dynamic-vehicle proof of concept
(see the main README's "Dynamic vehicles" section). These are one-off experiment scripts, not part
of the polished pipeline in `scripts/` — they have hardcoded params and target `scene-0061`'s leading van.

Run from the repo root with `PYTHONPATH=src`. Intermediates are read/written under
`out/vehicle_cache/` (gitignored — `mkdir -p out/vehicle_cache` first); final deliverables (rrds, mp4s)
land in `out/`.

## Pipeline order

1. **`extract_instances.py`** — SAM3 *per-instance* vehicle masks per frame (`segment_instances`) →
   `inst.npy` (+ `instances.png` overlay).
2. **`associate.py`** — greedy nearest-centroid frame-to-frame association → the single-vehicle track
   `best_track.npy` (+ `best_track.png` montage).

Per-instance geometry (two experiments + variants):
3. **`da3_vehicle.py`** — DA3 on tight crops → `da3_cloud.npz` (scale collapses — negative result).
4. **`da3_maskedfull.py`** — DA3 on masked full frames → `da3_maskedfull_cloud.npz` (also collapses).
5. **`colmap_vehicle.py`** — COLMAP untuned on the van (3/35 — negative result).
6. **`colmap_tuned.py`** — COLMAP tuned (35/35) → `colmap_van/sparse_tuned/` (the usable per-instance poses).

Scale + composition:
7. **`compose_vehicle.py`** — depth-anchor scale (`s = median(z_global/z_colmap)`) + place the scaled van
   into the whole global scene.
8. **`compose2.py`** — van-only global depth brought into the scaled COLMAP frame + scale-alignment check.
9. **`verify_rrd.py`** — verify COLMAP far→near distances; early rrds.
10. **`rrd_frustums.py`** — the COLMAP + DA3 vehicle rrds (frustums, normalized) → `out/vehicle_*.rrd`.

3DGS + outputs:
11. **`vehicle_gs.py`** — frozen-pose SILog 3DGS on the van (scaled COLMAP poses + points, masked images,
    van depth, blank-area prune, cap 100k, batch 10) → `out/vehicle_gs.npz` + render montage.
12. **`vehicle_flythrough_rrd.py`** — flythrough mp4 + rrd of the trained van → `out/vehicle_gs_flythrough.mp4`,
    `out/vehicle_gs.rrd`.
