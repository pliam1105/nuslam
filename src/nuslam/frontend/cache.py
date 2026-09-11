"""On-disk caching for offline frontend outputs (tracks, masks).

The frontend (CoTracker, road segmentation) is delegated and run *offline*: once
per scene, its output is written under a cache root and the estimator loads it
back instantly. Keeps heavy model inference off the SLAM loop's critical path.

Layout::

    <cache_root>/<scene_name>/tracks.npz
    <cache_root>/<scene_name>/masks.npz
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..types import DepthMap, GroundMask, MetricUpgrade, TrackSet


def scene_dir(cache_root: Path | str, scene_name: str) -> Path:
    d = Path(cache_root) / scene_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- tracks --------------------------------------------------------------

def _tracks_name(variant: str) -> str:
    return "tracks.npz" if not variant else f"tracks_{variant}.npz"


def save_tracks(cache_root: Path | str, scene_name: str, tracks: TrackSet, *, variant: str = "") -> Path:
    """Save a TrackSet. ``variant`` (e.g. a backend name) namespaces the file so
    outputs from different trackers can coexist; empty => the canonical file that
    the pipeline loads by default."""
    path = scene_dir(cache_root, scene_name) / _tracks_name(variant)
    np.savez_compressed(
        path,
        frame_tokens=np.asarray(tracks.frame_tokens),
        points=tracks.points.astype(np.float32),
        visible=tracks.visible.astype(bool),
        query_frame=np.int64(tracks.query_frame),
        is_ground=(np.asarray([]) if tracks.is_ground is None else tracks.is_ground.astype(bool)),
        seed_frame=(np.asarray([], dtype=np.int64) if tracks.seed_frame is None
                    else tracks.seed_frame.astype(np.int64)),
    )
    return path


def load_tracks(cache_root: Path | str, scene_name: str, *, variant: str = "") -> TrackSet | None:
    path = Path(cache_root) / scene_name / _tracks_name(variant)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    is_ground = z["is_ground"]
    seed_frame = z["seed_frame"] if "seed_frame" in z.files else np.asarray([])
    return TrackSet(
        frame_tokens=[str(t) for t in z["frame_tokens"]],
        points=z["points"],
        visible=z["visible"],
        query_frame=int(z["query_frame"]),
        is_ground=(None if is_ground.size == 0 else is_ground),
        seed_frame=(None if seed_frame.size == 0 else seed_frame),
    )


# ---- masks ---------------------------------------------------------------

def save_masks(cache_root: Path | str, scene_name: str, masks: list[GroundMask]) -> Path:
    path = scene_dir(cache_root, scene_name) / "masks.npz"
    # Probabilities are quantized to uint8 (0..255 <-> 0..1): ample for threshold
    # sweeps and ~4x smaller on disk than float32.
    have_prob = bool(masks) and masks[0].prob is not None
    probs_u8 = (
        np.stack([np.clip(m.prob, 0.0, 1.0) * 255.0 for m in masks]).round().astype(np.uint8)
        if have_prob
        else np.asarray([], dtype=np.uint8)
    )
    np.savez_compressed(
        path,
        tokens=np.asarray([m.token for m in masks]),
        masks=np.stack([m.mask for m in masks]).astype(bool),
        probs_u8=probs_u8,
    )
    return path


def load_masks(cache_root: Path | str, scene_name: str) -> dict[str, GroundMask] | None:
    """Return a ``token -> GroundMask`` dict, or None if not cached."""
    path = Path(cache_root) / scene_name / "masks.npz"
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    tokens = [str(t) for t in z["tokens"]]
    masks = z["masks"]
    probs_u8 = z["probs_u8"]
    out: dict[str, GroundMask] = {}
    for i, tok in enumerate(tokens):
        out[tok] = GroundMask(
            token=tok,
            mask=masks[i],
            prob=(None if probs_u8.size == 0 else probs_u8[i].astype(np.float32) / 255.0),
        )
    return out


# ---- tagged segmentation masks (sky / vehicle / ... per backend+prompt+threshold) --------

def _seg_name(tag: str) -> str:
    slug = re.sub(r"[^a-z0-9.]+", "-", str(tag).lower()).strip("-")
    return f"seg_{slug}.npz"


def save_seg_masks(cache_root: Path | str, scene_name: str, tag: str,
                   masks: list[GroundMask]) -> Path:
    """Cache binary segmentation masks under ``tag`` (e.g. 'sam3-vehicle-0.5'), so re-running with
    the same segmenter/prompt/threshold reuses them instead of re-estimating. Binary only (the run
    consumes binary keep/drop masks); the tag namespaces prompts and settings so they never collide."""
    path = scene_dir(cache_root, scene_name) / _seg_name(tag)
    np.savez_compressed(
        path,
        tokens=np.asarray([m.token for m in masks]),
        masks=np.stack([m.mask for m in masks]).astype(bool) if masks else np.asarray([], dtype=bool),
    )
    return path


def load_seg_masks(cache_root: Path | str, scene_name: str, tag: str) -> dict[str, np.ndarray] | None:
    """Return a ``token -> bool mask`` dict for ``tag``, or None if not cached."""
    path = Path(cache_root) / scene_name / _seg_name(tag)
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    tokens = [str(t) for t in z["tokens"]]
    masks = z["masks"]
    return {tok: masks[i] for i, tok in enumerate(tokens)}


# ---- depth ---------------------------------------------------------------

def save_depth(cache_root: Path | str, scene_name: str, depths: list[DepthMap]) -> Path:
    path = scene_dir(cache_root, scene_name) / "depth.npz"
    # depth as float16 (relative, coarse init -- half precision is ample and halves
    # the file); conf quantized to uint8; sky packed as bits. Camera pose/intrinsic
    # (DA3-Base only) kept at float32 -- tiny (N*4*4 + N*3*3) and precision matters
    # for the metric upgrade.
    have_conf = bool(depths) and depths[0].conf is not None
    have_sky = bool(depths) and depths[0].sky is not None
    have_ext = bool(depths) and depths[0].extrinsic is not None
    have_int = bool(depths) and depths[0].intrinsic is not None
    np.savez_compressed(
        path,
        tokens=np.asarray([d.token for d in depths]),
        depth=np.stack([d.depth for d in depths]).astype(np.float16),
        is_metric=np.asarray([d.is_metric for d in depths], dtype=bool),
        conf=(np.stack([np.clip(d.conf, 0, 1) * 255 for d in depths]).round().astype(np.uint8)
              if have_conf else np.asarray([], dtype=np.uint8)),
        sky=(np.stack([d.sky for d in depths]).astype(bool) if have_sky
             else np.asarray([], dtype=bool)),
        extrinsic=(np.stack([d.extrinsic for d in depths]).astype(np.float32) if have_ext
                   else np.asarray([], dtype=np.float32)),
        intrinsic=(np.stack([d.intrinsic for d in depths]).astype(np.float32) if have_int
                   else np.asarray([], dtype=np.float32)),
    )
    return path


def load_depth(cache_root: Path | str, scene_name: str) -> dict[str, DepthMap] | None:
    """Return a ``token -> DepthMap`` dict, or None if not cached."""
    path = Path(cache_root) / scene_name / "depth.npz"
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    tokens = [str(t) for t in z["tokens"]]
    depth, is_metric, conf, sky = z["depth"], z["is_metric"], z["conf"], z["sky"]
    ext = z["extrinsic"] if "extrinsic" in z.files else np.asarray([])
    ins = z["intrinsic"] if "intrinsic" in z.files else np.asarray([])
    out: dict[str, DepthMap] = {}
    for i, tok in enumerate(tokens):
        sky_i = sky[i].astype(bool) if sky.size else None
        out[tok] = DepthMap(
            token=tok,
            depth=depth[i].astype(np.float32),
            is_metric=bool(is_metric[i]),
            conf=(None if conf.size == 0 else conf[i].astype(np.float32) / 255.0),
            sky=sky_i,
            extrinsic=(None if ext.size == 0 else ext[i].astype(np.float32)),
            intrinsic=(None if ins.size == 0 else ins[i].astype(np.float32)),
        )
    return out


# ---- metric upgrade (Stage 1 result) ------------------------------------

def save_metric_upgrade(cache_root: Path | str, scene_name: str, mu: MetricUpgrade) -> Path:
    """Cache the metric-upgrade result so the scale/Gaussian stages skip the resolve."""
    path = scene_dir(cache_root, scene_name) / "metric_upgrade.npz"
    diag_keys = np.asarray(list(mu.diagnostics.keys()))
    diag_vals = np.asarray([float(v) for v in mu.diagnostics.values()], dtype=np.float64)
    np.savez_compressed(
        path,
        tokens=np.asarray(mu.tokens),
        H=mu.H.astype(np.float64),
        omega_star=mu.omega_star.astype(np.float64),
        plane_at_infinity=mu.plane_at_infinity.astype(np.float64),
        world_from_cam=mu.world_from_cam.astype(np.float64),
        metric_cameras=mu.metric_cameras.astype(np.float64),
        K_recovered=mu.K_recovered.astype(np.float64),
        K_true=mu.K_true.astype(np.float64),
        diag_keys=diag_keys,
        diag_vals=diag_vals,
    )
    return path


def load_metric_upgrade(cache_root: Path | str, scene_name: str) -> MetricUpgrade | None:
    """Load the cached metric upgrade, or None if not present."""
    path = Path(cache_root) / scene_name / "metric_upgrade.npz"
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=False)
    diag = {str(k): float(v) for k, v in zip(z["diag_keys"], z["diag_vals"])}
    return MetricUpgrade(
        tokens=[str(t) for t in z["tokens"]],
        H=z["H"], omega_star=z["omega_star"], plane_at_infinity=z["plane_at_infinity"],
        world_from_cam=z["world_from_cam"], metric_cameras=z["metric_cameras"],
        K_recovered=z["K_recovered"], K_true=z["K_true"], diagnostics=diag,
    )
