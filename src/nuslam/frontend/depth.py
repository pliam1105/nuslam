"""Monocular depth via Depth Anything 3 (delegated frontend, for initialization).

Runs DA3Mono-Large per keyframe and returns a per-frame :class:`~nuslam.types.DepthMap`
(relative depth + confidence + sky mask) at full image resolution. This is the
dense point source the reconstruction is seeded from: a depth map back-projected
on a strided pixel grid gives ~1e5 points immediately -- roughly where 3DGS lands
after densification -- which is what makes running without densification viable
early on.

Delegated on purpose: DA3 is used ONLY as an initializer (the role COLMAP plays in
standard 3DGS). Only its depth is extracted; its predicted poses and gaussians are
not used, so the reconstruction, poses, and scale are all resolved by the project's
own pipeline. DA3Mono-Large is Apache-2.0 (the Giant/Nested flagships are
CC-BY-NC; stay on the Mono/Metric-Large tier for a permissive repo).

The back-projection of this depth into Gaussian means/scales/colors is core
representation/initialization work and is written by hand -- it is not here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import DepthMap, Keyframe

log = logging.getLogger(__name__)

_MODEL_ID = "depth-anything/DA3Mono-Large"  # Apache-2.0, relative monocular depth


@dataclass
class DepthConfig:
    model_id: str = _MODEL_ID
    process_res: int = 504     # DA3 resizes the long side to this before inference
    device: str | None = None  # None -> cuda if available else cpu
    keep_conf: bool = True
    keep_sky: bool = True


class DepthEstimator:
    """Lazy-loaded DA3 wrapper. Call :meth:`estimate_scene` on keyframes."""

    def __init__(self, config: DepthConfig | None = None) -> None:
        self.config = config or DepthConfig()
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from depth_anything_3.api import DepthAnything3

        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading DA3 %s on %s (first run downloads weights)", self.config.model_id, self._device)
        self._model = DepthAnything3.from_pretrained(self.config.model_id).to(self._device).eval()

    def estimate_scene(self, keyframes: Sequence[Keyframe]) -> list[DepthMap]:
        """Estimate a DepthMap per keyframe, aligned to ``keyframes``."""
        self._ensure_loaded()
        import torch

        out: list[DepthMap] = []
        for i, kf in enumerate(keyframes):
            img = kf.image()  # (H, W, 3) uint8
            K = kf.calib.intrinsic.astype(np.float32)[None]
            with torch.no_grad():
                pred = self._model.inference([img], intrinsics=K, process_res=self.config.process_res)
            h, w = kf.calib.height, kf.calib.width
            depth = self._to_full(pred.depth, h, w, nearest=False)
            conf = (self._to_full(pred.conf, h, w, nearest=False)
                    if (self.config.keep_conf and getattr(pred, "conf", None) is not None) else None)
            sky = None
            if self.config.keep_sky and getattr(pred, "sky", None) is not None:
                sky = self._to_full(pred.sky, h, w, nearest=True) > 0.5
            out.append(DepthMap(
                token=kf.token,
                depth=depth.astype(np.float32),
                is_metric=bool(getattr(pred, "is_metric", False)),
                conf=(None if conf is None else conf.astype(np.float32)),
                sky=sky,
            ))
            log.info("depth %d/%d (range %.2f..%.2f, metric=%s)",
                     i + 1, len(keyframes), float(depth.min()), float(depth.max()),
                     out[-1].is_metric)
        return out

    @staticmethod
    def _to_full(arr, h: int, w: int, *, nearest: bool) -> np.ndarray:
        """Squeeze a (1, h', w') / (h', w') array and resize to full (h, w)."""
        import cv2

        a = np.asarray(arr)
        a = a.squeeze()
        if a.ndim != 2:
            a = a.reshape(a.shape[-2], a.shape[-1])
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.resize(a.astype(np.float32), (w, h), interpolation=interp)
