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
_RECON_MODEL_ID = "depth-anything/DA3-Base"  # Apache-2.0, full reconstruction (depth+extrinsics+intrinsics)


@dataclass
class DepthConfig:
    model_id: str = _MODEL_ID
    process_res: int = 504     # DA3 resizes the long side to this before inference
    device: str | None = None  # None -> cuda if available else cpu
    keep_conf: bool = True
    keep_sky: bool = True


@dataclass
class ReconConfig:
    model_id: str = _RECON_MODEL_ID
    process_res: int = 504
    device: str | None = None
    keep_conf: bool = True


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


class DA3ReconEstimator:
    """DA3-Base full reconstruction: per-frame depth + DA3 pose + DA3 estimated K.

    Unlike :class:`DepthEstimator` (per-frame relative depth), DA3-Base is a
    *multi-view* model: all keyframes are reconstructed jointly in one forward
    pass, yielding a consistent set of extrinsics/intrinsics. It is
    *intrinsic-agnostic* -- the true calibration is NOT passed in; DA3 estimates
    its own (typically wrong) ``K_da3``. Those estimates plus the true ``calib``
    intrinsics are what the metric upgrade consumes to rectify the reconstruction
    to metric.

    Returns a :class:`~nuslam.types.DepthMap` per keyframe with ``extrinsic`` and
    ``intrinsic`` populated. Intrinsics are rescaled from DA3's processing
    resolution to the keyframe's full image resolution so they pair with ``depth``
    and ``calib.intrinsic``. Everything is loaded into a single forward pass, so
    ``process_res`` / the number of frames govern peak memory -- window a long
    scene via ``max_frames`` upstream if it does not fit.
    """

    def __init__(self, config: ReconConfig | None = None) -> None:
        self.config = config or ReconConfig()
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from depth_anything_3.api import DepthAnything3

        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading DA3-Base %s on %s (first run downloads weights)", self.config.model_id, self._device)
        self._model = DepthAnything3.from_pretrained(self.config.model_id).to(self._device).eval()

    def estimate_scene(self, keyframes: Sequence[Keyframe]) -> list[DepthMap]:
        """Jointly reconstruct all keyframes; one DepthMap each (extrinsic+intrinsic set)."""
        self._ensure_loaded()
        import torch

        images = [kf.image() for kf in keyframes]  # each (H, W, 3) uint8
        with torch.no_grad():
            # No intrinsics passed: DA3-Base is intrinsic-agnostic and estimates its own.
            pred = self._model.inference(images, process_res=self.config.process_res)

        if pred.extrinsics is None or pred.intrinsics is None:
            raise RuntimeError(
                f"{self.config.model_id} returned no extrinsics/intrinsics; a full-reconstruction "
                "model (da3-base/-small/-large/-giant) is required, not a mono model.")

        # Rebase all poses so the FIRST camera is the origin ([R_0|t_0] = [I|0]).
        # Extrinsics are world->camera, so E_i @ inv(E_0) re-expresses every pose in
        # camera 0's frame (E_0 -> I). The metric upgrade's block-form rectifier
        # H = [[M_0, 0], [v^T, s]] is only valid in this gauge; DA3's raw output
        # frame is arbitrary, so canonicalize it once here.
        E0_inv = np.linalg.inv(self._homogenize(np.asarray(pred.extrinsics[0], np.float32)))

        out: list[DepthMap] = []
        for i, kf in enumerate(keyframes):
            h, w = kf.calib.height, kf.calib.width
            depth_proc = np.asarray(pred.depth[i]).squeeze()
            depth = DepthEstimator._to_full(depth_proc, h, w, nearest=False)
            conf = None
            if self.config.keep_conf and getattr(pred, "conf", None) is not None:
                conf = DepthEstimator._to_full(pred.conf[i], h, w, nearest=False).astype(np.float32)
            K = self._rescale_K(np.asarray(pred.intrinsics[i], np.float32),
                                 proc_hw=depth_proc.shape[:2], full_hw=(h, w))
            out.append(DepthMap(
                token=kf.token,
                depth=depth.astype(np.float32),
                is_metric=bool(getattr(pred, "is_metric", False)),
                conf=conf,
                sky=None,  # DA3-Base does not emit a sky mask
                extrinsic=self._homogenize(np.asarray(pred.extrinsics[i], np.float32)) @ E0_inv,
                intrinsic=K,
            ))
            log.info("recon %d/%d (depth %.2f..%.2f, f=%.1f)",
                     i + 1, len(keyframes), float(depth.min()), float(depth.max()), float(K[0, 0]))
        return out

    @staticmethod
    def _rescale_K(K: np.ndarray, *, proc_hw: tuple[int, int], full_hw: tuple[int, int]) -> np.ndarray:
        """Scale intrinsics from DA3 processing resolution to full image resolution."""
        sy = full_hw[0] / proc_hw[0]
        sx = full_hw[1] / proc_hw[1]
        S = np.diag([sx, sy, 1.0]).astype(np.float32)
        return (S @ K).astype(np.float32)

    @staticmethod
    def _homogenize(E: np.ndarray) -> np.ndarray:
        """Ensure a (4, 4) homogeneous pose (DA3 may hand back (3, 4))."""
        if E.shape == (4, 4):
            return E
        H = np.eye(4, dtype=np.float32)
        H[:3, :4] = E[:3, :4]
        return H
