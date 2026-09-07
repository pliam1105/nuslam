# nuslam — monocular metric Gaussian-splat reconstruction on nuScenes

Reconstruct a scene from a **single** nuScenes camera with **3D (or 2D) Gaussian
splatting**, refine the camera poses **through the differentiable rasterizer**,
and resolve the **monocular scale** metrically with **semantic ground
constraints** — road/ground-segmented regions anchored to a locally estimated
ground plane, plus the vehicle's wheel-contact points constrained to that plane.
The metric anchor acts directly on the reconstruction, so the result is natively
metric. Batch/windowed optimization (Adam over Gaussians + pose deltas + the
scale/ground residuals).

**The novel core.** Monocular splat-SLAM (MonoGS and similar) inherits scale
ambiguity; resolving metric scale *inside* a Gaussian-splat reconstruction from
semantic ground geometry is the contribution. Monocular, metric via semantic
ground, batch-first.

> **Scope of this repository.** The Gaussian-splat reconstruction itself — the
> representation and initialization, the gsplat rasterizer calls, the training
> loop, the pose refinement, every loss (photometric + metric-anchor) and their
> weighting, and the scale-resolution geometry — is the core substance and is
> written by hand; it is **not** generated and is not part of this repository's
> code. What lives here is the surrounding infrastructure: data, segmentation,
> visualization, evaluation, and the gsplat build toolchain. The only delegated
> dependency is gsplat's internal CUDA kernels.

## Two-rung structure

The scale idea is de-risked in isolation before the joint optimization:

1. **Rung 1 — explicit scale variable.** Up-to-scale splat reconstruction + pose
   refinement + a single scalar metric-scale variable `s`, resolved by the
   ground/wheel residual. Validates that the semantic constraint recovers metric
   scale on its own (optimize `s`, compare to nuScenes GT). The safety net.
2. **Rung 2 — constraints directly on the Gaussians.** No separate scalar; the
   ground/wheel-contact residuals act on the Gaussians in joint optimization, so
   the reconstruction is natively metric.

The monocular ambiguity is a single scalar (metric scale); the recovered scalar
from the Sim(3) evaluation alignment (`nuslam.eval`) against GT is the direct
read-out of whether it locked (≈ 1.0 = locked).

## What's built vs. what's core

| Layer | Module | Status |
|---|---|---|
| nuScenes monocular keyframe stream + calibration/GT + windowing | `nuslam.data` | infrastructure (built) |
| Road/ground segmentation (CLIPSeg) + optional tracking (CoTracker/KLT) | `nuslam.frontend` | infrastructure (built) |
| **Gaussian-splat reconstruction: representation, gsplat calls, training loop, pose refinement, losses, scale** | — | **core substance (written by hand)** |
| Rerun logging (images, frusta, point clouds, splats) + figures | `nuslam.viz` | infrastructure (built) |
| Scale-vs-GT + ATE/RPE on refined poses | `nuslam.eval` | infrastructure (built) |
| Factor-graph SLAM integration (render-as-a-factor + fusion) | `nuslam.backend` | staged extension (parked) |

## Setup

```bash
./scripts/setup_env.sh      # venv (reuses system torch 2.9.1+cu129) + pip deps
./scripts/setup_gsplat.sh   # matched CUDA 12.9 toolchain + gsplat, kernels built
```

`setup_gsplat.sh` exists because gsplat JIT-compiles its CUDA kernels against the
torch build, which needs a **12.x** nvcc while the system nvcc is **13.1**. It
installs a CUDA 12.9 conda toolchain (`cuda129`) and a `.pth` hook so every
`.venv/bin/python` builds/loads gsplat transparently. Verify:

```bash
.venv/bin/python -c "import gsplat; from gsplat.cuda._backend import _C; print('gsplat', gsplat.__version__, _C is not None)"
```

Data — nuScenes **v1.0-mini**. Symlink an existing copy, or fetch (~4.2 GB, no
login):

```bash
ln -s /path/to/nuscenes data/nuscenes      # or:
./scripts/download_data.sh data/nuscenes
```

## Verify the infrastructure first

Order matters: clean masks and correct geometry before any modelling.

```bash
# 1. calibration: GT 3D boxes projected into CAM_FRONT via the extracted calib
.venv/bin/python scripts/inspect_sample.py --scene scene-0061 --index 10 --out out/calib_check.png

# 2. road/ground masks (+ optional tracks), cached and previewed
.venv/bin/python scripts/run_frontend.py --scene scene-0061 --preview out/frontend_preview

# 3. raw-data view in Rerun (poses, camera frustum, image; --masks overlays ground)
.venv/bin/python scripts/visualize_scene.py --scene scene-0061 --masks
#    or write a shareable recording:  --save out/scene-0061.rrd  (open: .venv/bin/rerun <file>.rrd)
```

Masks land in `out/frontend_cache/<scene>/masks.npz`. Eyeball the previews —
garbage masks give a garbage ground plane. The segmentation prompt, threshold,
and (optional) tracker are configurable; see `scripts/run_frontend.py --help`.

## Evaluation read-out

`nuslam.eval` scores a reconstruction's refined poses against nuScenes GT:
ATE/RPE, and the Umeyama Sim(3) alignment whose recovered **scalar scale** is the
metric-scale diagnostic (≈ 1.0 once resolved; an SE(3) alignment should then fit
without needing to solve scale). This is the number that says whether Rung 1/2
worked.

## Guardrails worth keeping in view

- Reach **Rung 1** (does the ground/wheel anchor recover scale via explicit `s`?)
  before **Rung 2**, so a failure is the scale idea, not the joint coupling.
- **RANSAC**, not least-squares, for the ground plane — segmentation bleeds onto
  curbs/low objects; gate the wheel-contact anchor on inlier support.
- Watch the **photometric-vs-metric weighting** — start the anchor loose so the
  render converges, then increase; the anchor overpowering the render (or vice
  versa) is the main failure mode.
- **2DGS** gives cleaner depth + a surface normal, directly useful for a ground
  plane — worth considering over 3DGS as the base.
- nuScenes is Boston/Singapore; the local-plane assumption holds where the road
  is locally near-planar — check the slope profile.

## Layout

```
src/nuslam/
  transforms.py          SE(3) helpers (numpy)
  types.py               data contract: CameraCalib, Keyframe, TrackSet, GroundMask, streams
  data/                  nuScenes monocular source + CAN streams
  frontend/              CLIPSeg road/ground segmentation, tracking (CoTracker + KLT), cache
  viz/                   Rerun logging (images/frusta/points/GaussianSplats3D) + figures
  eval/                  scale-vs-GT (Umeyama Sim(3)) + ATE/RPE
  backend/               factor-graph SLAM — staged extension, parked (not the current core)
scripts/                 setup_env, setup_gsplat, download_data, inspect_sample,
                         run_frontend, render_frontend_video, visualize_scene
tests/                   unit (transforms/metrics/cache/seeding/refine) + mini-data integration
```

The Gaussian-splat reconstruction (representation, gsplat calls, optimization,
losses, scale) is written by hand and is not part of this tree.

Run the tests: `.venv/bin/python -m pytest tests/ -q` (data-dependent tests skip
if the mini split is absent).
