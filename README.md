# nuslam — monocular metric Gaussian-splat reconstruction on nuScenes

Reconstruct a scene from a **single** nuScenes camera as **3D (or 2D) Gaussians**,
refine the camera poses **through the differentiable rasterizer**, and resolve the
**monocular scale** metrically — never by reading it from ground truth.

A monocular reconstruction model (Depth Anything 3, **DA3-Base**) supplies a full
joint reconstruction — per-frame depth, pose, and intrinsics — but under a wrong,
intrinsic-agnostic calibration. A **metric upgrade** recovers the true geometry from
it: knowing the true intrinsics collapses the projective ambiguity to a
**metric-up-to-scale** reconstruction via a dual-absolute-quadric (DAQ) solve. The one
remaining scalar — global metric scale — is then resolved by **semantic ground
constraints** (road/ground plane + wheel-contact), and independently cross-checked
against **GPS/IMU**. The anchor acts directly on the reconstruction, so the result is
natively metric.

**The novel core.** Monocular splat-SLAM (MonoGS) inherits scale ambiguity;
uncalibrated approaches (VGGT-SLAM) inherit the full projective ambiguity. Resolving
metric scale *inside* a splat reconstruction from semantic geometry — with known
intrinsics collapsing projective → metric-up-to-scale first — is the contribution.
The stratification the pipeline walks:

```
Projective  --known K-->  Metric-up-to-scale  --scale: ground / GPS-->  Metric
DAQ Ω*, 15 DoF            Sim(3), 7 DoF                                 SE(3), 6 DoF
```

The full staged plan is in `Metric_Anchored_GS_SLAM_Engineering_Plan_v3.pdf` (with
the metric-upgrade derivation in `Metric_Upgrade_Known_Intrinsics_Handbook.pdf`).

> **Scope of this repository.** The reconstruction core — the Gaussian representation
> and initialization, the gsplat rasterizer calls, the training loop, the pose
> refinement, every loss (photometric + metric-anchor) and their weighting, the
> DAQ metric-upgrade solve, and the scale-resolution geometry — is the core substance
> and is **written by hand** (not generated). What lives here as infrastructure: data,
> segmentation, running the delegated frontend models (depth/tracking), visualization,
> evaluation, and the gsplat build toolchain. The only delegated dependency is gsplat's
> internal CUDA kernels.
>
> **Ground truth is an oracle, never an input.** nuScenes GT poses and lidar score a
> finished reconstruction; scale is never fitted from GT (that would be cheating).

## Pipeline (staged)

Five stages; each is falsifiable on its own before the next is built. Stages 0–1 are
built; 2–4 are the work ahead.

0. **Infrastructure** *(built)* — data, frontend (depth/segmentation/tracking), lidar
   eval, Rerun viz, gsplat toolchain.
1. **DA3 metric upgrade** *(built + tested)* — DA3-Base reconstruction → normalized
   projective cameras → DAQ solve for the dual absolute quadric → plane at infinity →
   rectifying homography → metric cameras, corrected depth, and the **metric-up-to-scale**
   point cloud. `nuslam.recon.metric_upgrade` (hand-written core).
2. **Scale resolution** *(next)* — the one global scalar, three ways, **none using GT**:
   **(A) road/ground + wheel-contact** on the unprojected cloud (the novel core);
   **(B) GPS(+IMU tilt, +compass heading)** → a reference trajectory that the DA3 poses
   are Sim(3)-fit to; **(C) joint**. The three should agree; GT scores them afterward.
3. **Gaussian refinement** *(common tail)* — splatting with pose refinement + ground
   constraints (photometric + metric-anchor losses, weighting schedule).
4. **Factor-graph SLAM** *(future)* — render-as-a-factor + IMU/GPS/wheel + loop closure;
   multi-submap; DA3-SLAM follow-on.

**Route A is de-risked in two rungs:** *Rung 1* adds an explicit scalar `s` and checks it
recovers the metric value in isolation (the core hypothesis); *Rung 2* drops the scalar and
lets the ground/wheel residual act directly on the Gaussians, so the reconstruction is
natively metric.

## What's built vs. what's core

| Layer | Module | Status |
|---|---|---|
| nuScenes monocular keyframe stream + calibration/GT + windowing | `nuslam.data` | infrastructure (built) |
| Monocular depth + DA3-Base full reconstruction (depth+pose+K) | `nuslam.frontend` | infrastructure (built) |
| Road/ground segmentation (CLIPSeg) + optional tracking (CoTracker/KLT) | `nuslam.frontend` | infrastructure (built) |
| Lidar-into-camera projection: calib check + depth-vs-lidar eval | `nuslam.data` / `nuslam.eval` | infrastructure (built) |
| **DA3 metric upgrade: DAQ solve, rectifying homography, metric cameras/depth/cloud** | `nuslam.recon.metric_upgrade` | **core (written by hand) — built + tested** |
| **Scale resolution (road/ground + wheel-contact; GPS/IMU Sim(3) fit)** | `nuslam.recon` | **core (written by hand) — next** |
| **Gaussian-splat reconstruction: representation, gsplat calls, training loop, pose refinement, losses** | `nuslam.recon` | **core (written by hand) — next** |
| Rerun logging (images, frusta, point clouds, splats) + figures | `nuslam.viz` | infrastructure (built) |
| Trajectory eval (Umeyama Sim(3)/SE(3), ATE/RPE) — GT as oracle | `nuslam.eval` | infrastructure (built) |
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

# 3. DA3 monocular depth for init, cached; preview also scores recovered scale vs lidar
.venv/bin/python scripts/run_depth.py --scene scene-0061 --preview out/depth_preview

# 4. calibration via lidar: projected lidar (depth-colored) should sit on structure
.venv/bin/python scripts/inspect_sample.py --scene scene-0061 --index 10 --lidar --out out/calib_lidar.png

# 5. raw-data view in Rerun (poses, camera frustum, image; --masks overlays ground)
.venv/bin/python scripts/visualize_scene.py --scene scene-0061 --masks
#    or write a shareable recording:  --save out/scene-0061.rrd  (open: .venv/bin/rerun <file>.rrd)
```

Masks land in `out/frontend_cache/<scene>/masks.npz`. Eyeball the previews —
garbage masks give a garbage ground plane. The segmentation prompt, threshold,
and (optional) tracker are configurable; see `scripts/run_frontend.py --help`.

## Metric upgrade → metric-up-to-scale reconstruction

```bash
# 1. DA3-Base: cache per-frame depth + pose (frame 0 rebased to identity) + estimated K
.venv/bin/python scripts/run_recon.py --scene scene-0061

# 2. metric upgrade → metric-up-to-scale reconstruction, visualized with corrected intrinsics
.venv/bin/python scripts/run_metric_upgrade.py --scene scene-0061 --rerun
#    …or a shareable recording:  --save out/metric-0061.rrd   (open: .venv/bin/rerun <file>.rrd)
```

`run_metric_upgrade.py` solves the DAQ, recovers the metric cameras, and logs the
reconstruction to Rerun (upright/horizontal view; frustum size tracks cloud depth;
`--point-size` / `--lidar-size` / `--frustum-frac` tune the display). It prints the
per-camera recovered `K ≈ [c, c, 1]` — the free correctness check on the upgrade — and
notes that the global scale is **unresolved and never taken from GT**. `--eval-gt` (oracle
only) Sim(3)-aligns to GT to print ATE and the scale the road/GPS methods should reproduce,
and overlays lidar as a metric reference.

## Evaluation read-out — ground truth as an oracle

`nuslam.eval` scores a finished reconstruction against nuScenes GT (ATE/RPE, Umeyama
alignment); **GT is never an input to scale.** The three scale-resolution routes
(road/ground, GPS/IMU, joint) each produce the global scalar independently — and the
experiment is that they **agree with each other**, with GT scoring all three afterward.
The Umeyama Sim(3) scalar is a diagnostic: once scale is resolved legitimately it reads
≈ 1.0, and an SE(3) alignment (scale fixed) should already fit.

## Guardrails worth keeping in view

- **GT is an oracle, never an input to scale** — resolve scale from the ground anchor or
  GPS/IMU; a Sim(3) fit to GT would resolve the scalar by reading the answer.
- **Metric upgrade:** per-frame `K_da3` drift is expected and harmless (each is known and
  folded into its own camera); the live feasibility test is the DAQ residual / null-space
  eigenvalue gap / recovered-`K` anisotropy — **not** the focal spread. Rebase so the first
  camera is the origin (the block-form rectifier assumes that gauge).
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
  types.py               data contract: CameraCalib, Keyframe, TrackSet, GroundMask, DepthMap, streams
  data/                  nuScenes monocular source, CAN streams, lidar->camera projection
  frontend/              DA3 monocular depth + DA3-Base reconstruction, CLIPSeg, tracking (CoTracker+KLT), cache
  viz/                   Rerun logging (images/frusta/points/GaussianSplats3D) + figures
  eval/                  Umeyama Sim(3)/SE(3) + ATE/RPE + depth-vs-lidar (GT as oracle)
  backend/               factor-graph SLAM — staged extension, parked (not the current core)
  recon/                 reconstruction core — written by hand
    depth_init.py          depth back-projection (shared by viz + metric path) — built
    metric_upgrade.py      DAQ metric upgrade → metric-up-to-scale — built + tested
    (scale resolution, Gaussian representation/optimization/losses — to come)
scripts/                 setup_env, setup_gsplat, download_data, inspect_sample, run_frontend,
                         run_depth, run_recon (DA3-Base pose+K+depth), run_metric_upgrade,
                         render_frontend_video, visualize_scene, visualize_depth
tests/                   unit (transforms/metrics/cache/seeding/refine/metric_upgrade) + mini-data
```

`recon/` is the hand-written core. The metric upgrade is in place; the scale-resolution
routes and the Gaussian-splat reconstruction (representation, gsplat calls, optimization,
losses) are written there next.

Run the tests: `.venv/bin/python -m pytest tests/ -q` (data-dependent tests skip
if the mini split is absent).
