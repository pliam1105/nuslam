"""Score a rendered view against the ground-truth image (photometric quality).

Novel-view quality is read on held-out keyframes the optimization never saw. This
module is pure scoring -- it takes a rendered image and the ground-truth image and
returns PSNR / SSIM / L1; producing the render (the gsplat rasterizer call) is
estimator substance and lives in the reconstruction code, not here.

``holdout_indices`` gives the train/test keyframe split so novel-view scoring is well
defined and reproducible.

Plumbing: this scores appearance; it makes no modelling decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.metrics import structural_similarity


@dataclass
class RenderErrors:
    psnr: float           # dB, higher is better
    ssim: float           # [0, 1], higher is better
    l1: float             # mean absolute error in [0, 1] intensity
    num_pixels: int       # pixels scored (mask-aware for psnr/l1)

    def __str__(self) -> str:  # pragma: no cover - reporting
        return (f"render: PSNR={self.psnr:.2f} dB  SSIM={self.ssim:.4f}  "
                f"L1={self.l1:.4f}  n={self.num_pixels}")


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Normalize an (H, W, 3) or (H, W) image to float in [0, 1].

    Accepts uint8 (0..255) or float (already 0..1, or 0..255 -- detected by range).
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    if a.size and a.max() > 1.5:  # looks like 0..255 floats
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def evaluate_render(
    rendered: np.ndarray, gt: np.ndarray, *, mask: np.ndarray | None = None
) -> RenderErrors:
    """Score a rendered image against the GT image.

    ``rendered``, ``gt`` are (H, W, 3) or (H, W), uint8 or float. PSNR and L1 are
    computed over ``mask`` (H, W bool) if given -- useful to score only road pixels
    or exclude sky; SSIM is windowed so it is always computed over the full frame
    (its local windows do not admit an arbitrary mask cleanly). Returns
    :class:`RenderErrors`.
    """
    r = _to_float01(rendered)
    g = _to_float01(gt)
    if r.shape != g.shape:
        raise ValueError(f"render/gt shape mismatch: {r.shape} vs {g.shape}")

    diff = np.abs(r - g)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        sel = m[..., None] if r.ndim == 3 else m
        vals = diff[np.broadcast_to(sel, diff.shape)]
        n = int(vals.size)
        mse = float(np.mean(vals**2)) if n else 0.0
        l1 = float(np.mean(vals)) if n else 0.0
    else:
        n = int(diff.size)
        mse = float(np.mean(diff**2))
        l1 = float(np.mean(diff))

    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    ssim = float(structural_similarity(
        g, r, data_range=1.0, channel_axis=(-1 if r.ndim == 3 else None)))
    return RenderErrors(psnr=psnr, ssim=ssim, l1=l1, num_pixels=n)


def holdout_indices(n_frames: int, *, every: int = 8, offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split ``n_frames`` keyframes into (train, test) for novel-view scoring.

    Every ``every``-th frame (starting at ``offset``) is held out of the Gaussian
    optimization and used only to read novel-view PSNR/SSIM; the rest train. Frame
    0 is kept in train by default (``offset`` shifts which frames are held out).
    Returns ``(train_idx, test_idx)`` as int arrays.
    """
    idx = np.arange(int(n_frames))
    is_test = ((idx - offset) % every == 0) & (idx >= offset)
    return idx[~is_test], idx[is_test]
