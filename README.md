# nuslam — monocular metric reconstruction (SLAM) on nuScenes

Recover metric structure and motion from a **single** nuScenes camera by building
a **GTSAM factor graph** that anchors road-segmented landmarks to a locally
estimated ground plane and constrains the vehicle's wheel-contact points to that
plane — resolving monocular scale from semantic geometry, then fusing IMU, wheel
odometry, and GPS in the same graph.

> **Authorship note.** The estimator backend — the factor-graph design, the
> custom ground-plane / wheel-contact factors and their Jacobians, the
> scale-resolution logic, and the sensor fusion — is written by the author and is
> intentionally left unimplemented here (see `CLAUDE.md` §3). This repository
> provides everything *around* that seam: data, delegated frontend, visualization,
> evaluation, and the orchestration that feeds the graph and renders its output.

## What's built vs. what you write

| Layer | Module | Status |
|---|---|---|
| nuScenes monocular keyframe stream + calibration/GT | `nuslam.data` | ✅ built |
| CAN-bus IMU / wheel / GPS streams (rungs 3–5) | `nuslam.data.can_streams` | ✅ built (needs CAN download) |
| Offline tracking (CoTracker / KLT) + CLIPSeg road masks | `nuslam.frontend` | ✅ built |
| **Factor graph: variables, factors, Jacobians, fusion** | `nuslam.backend` | ✍️ **you** (seam only) |
| Live Foxglove bridge + trajectory/reconstruction figures | `nuslam.viz` | ✅ built |
| ATE / RPE vs. GT, Sim(3) scale read-out | `nuslam.eval` | ✅ built |
| data → frontend → graph → eval + viz orchestration | `nuslam.pipeline` | ✅ built |

## Setup

```bash
./scripts/setup_env.sh          # venv (reuses system torch) + pip install + import check
```

Data: the nuScenes **v1.0-mini** split. If you already have it, symlink it:

```bash
ln -s /path/to/nuscenes data/nuscenes
```

Otherwise fetch it (~4.2 GB, no login):

```bash
./scripts/download_data.sh data/nuscenes
```

## Verify the plumbing (do this first — CLAUDE.md §7)

```bash
# 1. calibration check: GT 3D boxes projected into CAM_FRONT via our calib
.venv/bin/python scripts/inspect_sample.py --scene scene-0061 --index 10 --out out/calib_check.png

# 2. live raw-data view (open Foxglove -> ws://localhost:8765)
.venv/bin/python scripts/visualize_scene.py --scene scene-0061
```

If box wireframes sit on the objects, intrinsics + the sensor→ego→global chain
are correct.

## Run the delegated frontend (offline, cached)

```bash
.venv/bin/python scripts/run_frontend.py --scene scene-0061 --preview out/frontend_preview
```

Writes `out/frontend_cache/scene-0061/{tracks,masks}.npz` and preview overlays
(road mask + tracks, cyan = ground-flagged). **Eyeball the previews** before
building the graph on them. First run downloads CLIPSeg (~150 MB) and CoTracker
weights.

Frontend options:

- `--tracker {cotracker,klt}` — learned CoTracker (dense, robust, GPU) or classic
  OpenCV Lucas-Kanade (sparse, corner-precise, ~40× faster on CPU). Each backend's
  output is also cached tagged (`tracks_cotracker.npz`, `tracks_klt.npz`).
- `--seed-method {shitomasi,grid}` — seed each grid cell on its strongest
  Shi-Tomasi corner (default) or its centre. Cells are **replenished**: any cell
  left empty as tracks drift off is re-seeded, so density holds across the scene.
- Subpixel refinement (`cv2.cornerSubPix`, on by default; `--no-refine`) re-localizes
  each visible point on the full-res image, accepting only small stable moves.
- `--cell-size`, `--no-replenish`, `--min-visible`, `--max-tracks` tune density.

Render a mask+tracks video from the cache: `scripts/render_frontend_video.py
--scene scene-0061 --tracker {cotracker,klt}`.

## Run the pipeline (step 0)

```bash
.venv/bin/python scripts/run_slam.py --scene scene-0061
```

This assembles the scene's `SlamInputs` (keyframes + tracks + masks + IMU/wheel/
GPS), calls `MonocularSLAM.run`, and — once you've written it — evaluates ATE/RPE
and writes `out/trajectory.png` (add `--live` to stream the reconstruction to
Foxglove). Until then it prints exactly what reaches the seam and exits cleanly,
so the whole pipeline is verifiable before a single factor exists.

## Where you write the backend

`src/nuslam/backend/`:

- `graph.py` — `MonocularSLAM.run(inputs) -> SlamEstimate`. Design the state and
  the graph here. `SlamInputs` (what you consume) and `SlamEstimate` (what viz/
  eval consume) are the fixed contract; everything else is yours.
- `factors.py` — `GroundPlaneFactor`, `WheelContactToPlaneFactor` stubs. Their
  residuals and Jacobians are the deep-dive centerpiece.

Follow the build ladder (`CLAUDE.md` §1): scale-free graph → **+ ground plane +
wheel contact (scale should resolve)** → IMU → wheel odometry → GPS. The Sim(3)
scale reported by `nuslam.eval` on rung 2 is the direct read-out of whether metric
scale locked.

### IMU / wheel / GPS (rungs 3–5)

These come from the **nuScenes-CAN expansion** (separate download; a free account
is required). Unzip it to `data/nuscenes/can_bus/`. Without it those streams are
empty and rungs 1–2 (the scale hypothesis) still run.

## Layout

```
src/nuslam/
  transforms.py          SE(3) helpers (numpy)
  types.py               data contract: CameraCalib, Keyframe, TrackSet, GroundMask, streams
  data/                  nuScenes monocular source + CAN streams
  frontend/              tracking (CoTracker + KLT), Shi-Tomasi seeding, subpixel
                         refine, CLIPSeg segmentation, on-disk cache
  backend/               THE FACTOR GRAPH — seam only; author writes the bodies
  viz/                   Foxglove bridge, trajectory/reconstruction figures
  eval/                  ATE/RPE, Umeyama Sim(3) alignment
  pipeline.py            end-to-end orchestration
scripts/                 setup, download, inspect, run_frontend, visualize, run_slam
tests/                   unit (transforms/metrics/cache/seam) + mini-data integration
```

Run the tests: `.venv/bin/python -m pytest tests/ -q` (data-dependent tests skip
if the mini split is absent).
