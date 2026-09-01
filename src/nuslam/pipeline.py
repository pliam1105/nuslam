"""End-to-end orchestration: data -> frontend -> [factor graph] -> eval + viz.

This is the seam wired for a run from step 0. It assembles a scene's
:class:`SlamInputs` -- keyframes, cached CoTracker tracks, road masks, IMU/wheel/
GPS streams -- hands them to :meth:`MonocularSLAM.run`, and routes the returned
:class:`SlamEstimate` into ATE/RPE eval and the trajectory figure.

Until the backend is written, :func:`run_scene` catches ``NotImplementedError`` and
reports exactly what reached the seam, so the whole pipeline is verifiable before
a single factor exists. Everything except ``MonocularSLAM.run`` is live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .backend import MonocularSLAM, SlamEstimate, SlamInputs
from .data import NuScenesMonoSource, load_proprio_streams
from .eval import TrajErrors, evaluate
from .frontend import cache
from .types import DEFAULT_CAMERA

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    dataroot: Path
    scene: str
    version: str = "v1.0-mini"
    camera: str = DEFAULT_CAMERA
    cache_root: Path = Path("out/frontend_cache")
    max_frames: int | None = None
    align: str = "sim3"


@dataclass
class PipelineResult:
    inputs: SlamInputs
    estimate: SlamEstimate | None
    errors: TrajErrors | None
    backend_ready: bool


def assemble_inputs(config: PipelineConfig) -> SlamInputs:
    """Build the graph's input contract for one scene from data + frontend cache.

    Requires the frontend cache to exist (run ``scripts/run_frontend.py`` first).
    Raises if tracks are missing -- reprojection needs correspondences.
    """
    source = NuScenesMonoSource(config.dataroot, config.version, camera=config.camera)
    keyframes = source.load_scene(config.scene, max_frames=config.max_frames)
    if len(keyframes) < 2:
        raise RuntimeError(f"scene {config.scene!r} yielded < 2 keyframes")
    scene_name = keyframes[0].scene_name

    tracks = cache.load_tracks(config.cache_root, scene_name)
    if tracks is None:
        raise FileNotFoundError(
            f"no cached tracks for {scene_name!r} under {config.cache_root} -- "
            f"run scripts/run_frontend.py --scene {scene_name} first"
        )
    masks = cache.load_masks(config.cache_root, scene_name) or {}
    if not masks:
        log.warning("no cached road masks for %s -- ground factors will have no gate", scene_name)

    proprio = load_proprio_streams(config.dataroot, scene_name, version=config.version)

    return SlamInputs(
        calib=keyframes[0].calib,
        keyframes=keyframes,
        tracks=tracks,
        masks=masks,
        proprio=proprio,
    )


def run_scene(config: PipelineConfig, slam_config: dict | None = None) -> PipelineResult:
    """Assemble inputs, run the estimator, evaluate. Backend may be unwritten."""
    inputs = assemble_inputs(config)
    _log_inputs(inputs)

    slam = MonocularSLAM(inputs.calib, slam_config)
    try:
        estimate = slam.run(inputs)
    except NotImplementedError as exc:
        log.warning("backend not implemented yet: %s", exc)
        return PipelineResult(inputs=inputs, estimate=None, errors=None, backend_ready=False)

    import numpy as np

    gt = np.stack([kf.ego2global_gt.matrix() for kf in inputs.keyframes])
    errors = evaluate(estimate.poses_ego2global, gt, align=config.align)
    log.info("eval: %s", errors)
    return PipelineResult(inputs=inputs, estimate=estimate, errors=errors, backend_ready=True)


def _log_inputs(inputs: SlamInputs) -> None:
    t = inputs.tracks
    ground = "n/a" if t.is_ground is None else int(t.is_ground.sum())
    log.info(
        "SlamInputs ready: %d keyframes | tracks %dx%d (query@%d, ground=%s) | "
        "masks %d | proprio imu=%d wheel=%d gps=%d",
        len(inputs.keyframes), t.num_frames, t.num_tracks, t.query_frame, ground,
        len(inputs.masks), len(inputs.proprio.imu), len(inputs.proprio.wheel),
        len(inputs.proprio.gps),
    )
