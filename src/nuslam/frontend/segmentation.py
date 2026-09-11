"""Prompt-based road/ground segmentation (delegated frontend).

Uses CLIPSeg (``CIDAS/clipseg-rd64-refined``) with a text prompt ("road") to
produce a per-keyframe ground mask. CLIPSeg is small (~150 MB) and prompt-driven,
which matches the project's "prompt-based road segmentation" plan and keeps the
ground definition swappable without retraining.

Delegated frontend: the mask quality gates the ground-plane / wheel-contact
factors, so the segmentation is verified visually before the graph is built
around it (``scripts/run_frontend.py --preview``). The segmentation model itself
is not estimator substance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import GroundMask, Keyframe

log = logging.getLogger(__name__)

_MODEL_ID = "CIDAS/clipseg-rd64-refined"


@dataclass
class SegConfig:
    prompt: str = "road"
    threshold: float = 0.35  # sigmoid prob above which a pixel is "ground"
    device: str | None = None  # None -> cuda if available else cpu
    batch_size: int = 8


class RoadSegmenter:
    """Lazy-loaded CLIPSeg wrapper. Call :meth:`segment_scene` on keyframes."""

    def __init__(self, config: SegConfig | None = None) -> None:
        self.config = config or SegConfig()
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading CLIPSeg %s on %s", _MODEL_ID, self._device)
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID)
        self._model = CLIPSegForImageSegmentation.from_pretrained(_MODEL_ID).to(self._device).eval()

    def segment_scene(
        self, keyframes: Sequence[Keyframe], *, keep_prob: bool = True
    ) -> list[GroundMask]:
        """Segment every keyframe; returns masks aligned to ``keyframes``."""
        self._ensure_loaded()
        import torch

        out: list[GroundMask] = []
        bs = self.config.batch_size
        for start in range(0, len(keyframes), bs):
            batch = keyframes[start : start + bs]
            images = [kf.image() for kf in batch]  # (H, W, 3) uint8 each
            prompts = [self.config.prompt] * len(batch)
            inputs = self._processor(
                text=prompts, images=images, padding=True, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits  # (B, h, w) at model resolution
                if logits.ndim == 2:  # transformers squeezes a batch of 1
                    logits = logits[None]
                probs = torch.sigmoid(logits)
            for kf, prob in zip(batch, probs):
                p = prob.float().cpu().numpy()
                mask = self._resize_to(p, kf.calib.height, kf.calib.width)
                out.append(
                    GroundMask(
                        token=kf.token,
                        mask=mask >= self.config.threshold,
                        prob=(mask.astype(np.float32) if keep_prob else None),
                    )
                )
            log.info("segmented %d/%d keyframes", min(start + bs, len(keyframes)), len(keyframes))
        return out

    @staticmethod
    def _resize_to(prob: np.ndarray, h: int, w: int) -> np.ndarray:
        import cv2

        return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


_SAM3_MODEL_ID = "facebook/sam3"


class Sam3Segmenter:
    """Lazy-loaded SAM3 (``facebook/sam3``) concept segmenter, drop-in for RoadSegmenter.

    SAM3 does open-vocabulary INSTANCE segmentation: given a text concept it returns
    per-instance masks + detection scores. The per-keyframe binary mask is the union of
    instances scoring above ``config.threshold`` (a detection score here, not a per-pixel
    sigmoid) with each mask binarized at ``mask_threshold``. Sharper, better-localized masks
    than CLIPSeg, at ~3.5 GB and GPU-preferred; call :meth:`release` to free it before training.
    """

    def __init__(self, config: SegConfig | None = None, *, mask_threshold: float = 0.5) -> None:
        self.config = config or SegConfig()
        self.mask_threshold = mask_threshold
        self._model = None
        self._processor = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading SAM3 %s on %s", _SAM3_MODEL_ID, self._device)
        self._processor = Sam3Processor.from_pretrained(_SAM3_MODEL_ID)
        self._model = Sam3Model.from_pretrained(_SAM3_MODEL_ID).to(self._device).eval()

    def segment_scene(
        self, keyframes: Sequence[Keyframe], *, keep_prob: bool = True
    ) -> list[GroundMask]:
        """Segment every keyframe with the current prompt; masks aligned to ``keyframes``."""
        self._ensure_loaded()
        import torch
        from PIL import Image

        out: list[GroundMask] = []
        for kf in keyframes:
            rgb = np.asarray(kf.image())
            h, w = rgb.shape[:2]
            inputs = self._processor(
                images=Image.fromarray(rgb), text=self.config.prompt, return_tensors="pt"
            ).to(self._device)
            with torch.no_grad():
                res = self._processor.post_process_instance_segmentation(
                    self._model(**inputs),
                    threshold=self.config.threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=[(h, w)],
                )[0]
            m = res["masks"]
            mask = (
                (m.sum(0) > 0).cpu().numpy().astype(bool)
                if (m is not None and len(m))
                else np.zeros((h, w), dtype=bool)
            )
            out.append(
                GroundMask(
                    token=kf.token,
                    mask=mask,
                    prob=(mask.astype(np.float32) if keep_prob else None),
                )
            )
        log.info("SAM3 segmented %d keyframes ('%s')", len(keyframes), self.config.prompt)
        return out

    def release(self) -> None:
        """Drop the model/processor and free GPU memory (call before a training run)."""
        import torch

        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
