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
