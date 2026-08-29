"""On-disk caching for offline frontend outputs (tracks, masks).

The frontend (CoTracker, road segmentation) is delegated and run *offline*: once
per scene, its output is written under a cache root and the estimator loads it
back instantly. Keeps heavy model inference off the SLAM loop's critical path.

Layout::

    <cache_root>/<scene_name>/tracks.npz
    <cache_root>/<scene_name>/masks.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..types import GroundMask, TrackSet


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
