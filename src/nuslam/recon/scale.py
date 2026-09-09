"""Metric scale resolution (Stage 2).

The DA3 metric upgrade (Stage 1, ``metric_upgrade.py``) leaves the reconstruction
metric UP TO ONE global scale. This module resolves that single scalar without ground
truth -- ground truth is only ever an oracle used to *score* the resolved scale (see
``nuslam.eval``).

Route B (GPS): fit the recovered camera trajectory to the GPS (nuScenes-CAN ``pose``)
ground track by an iterated Umeyama alignment. The fit accounts for the camera->ground
lever arm using each frame's own recovered orientation (so no compass/heading is
needed) and pins the vertical to the measured ground plane (z = 0). The Umeyama
primitive itself, :func:`nuslam.transforms.umeyama`, is shared infrastructure; the
lever-arm reduction and the fixed-point iteration resolve the metric scale.

Derivation and the reason a closed form does not exist (the leftover objective in the
rotation is quadratic, not linear): see ``Umeyama_Alignment_Handout.pdf``, sections 6-7.

Other routes attach here as they are built: Route A (road/ground + wheel-contact on the
unprojected cloud) and Route C (joint). The three resolve the same scalar independently
and should agree; ground truth scores all three afterward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..transforms import umeyama  # Sim(3)/SE(3) primitive: (src, dst, with_scale) -> (s, R, t)


@dataclass(frozen=True)
class ScaleResult:
    """Output of a scale-resolution route.

    ``scale`` is the resolved metric scale (reconstruction units -> metres) that maps
    the up-to-scale reconstruction onto the reference (GPS) ground track. ``T`` is the
    (4, 4) Sim(3) that takes est camera centres into the reference/map frame;
    ``aligned_xy`` is the reduced est ground track after alignment (for plotting
    against GPS); ``residual`` is the RMS horizontal fit error in metres; ``num_iters``
    is how many fixed-point steps ran. Extend the fields if the resolver needs to
    report more.
    """

    scale: float
    T: np.ndarray                 # (4, 4) est-camera -> reference-frame similarity
    aligned_xy: np.ndarray        # (M, 2) reduced ground track in reference xy
    residual: float               # RMS horizontal residual (m)
    num_iters: int


def resolve_scale_gps(
    world_from_cam: np.ndarray,   # (N, 4, 4) est camera->world, reconstruction units
    gps_xy: np.ndarray,           # (N, 2) interpolated GPS map-frame xy
    sensor2ego: np.ndarray,       # (4, 4) camera->ego extrinsic (metric)
    valid: np.ndarray,            # (N,) bool, GPS-coverage mask
    *,
    iters: int = 3,
) -> ScaleResult:
    """Route B: resolve the global metric scale by an iterated lever-arm Umeyama fit of
    the recovered camera trajectory to the GPS ground track.

    Method (handout sections 6-7), over the ``valid`` frames only:

      * lever arm ``a = -R_c2e^T t_c2e`` -- the ground/ego point in the camera frame
        (metres), from ``sensor2ego``;
      * per-frame ``b_i = R^w_i a`` -- known, since ``R^w_i`` (the recovered camera
        orientation) comes from ``world_from_cam``;
      * target ``r_i = (gps_x_i, gps_y_i, 0)`` -- GPS horizontal with z pinned to the
        measured ground plane;
      * iterate: modified targets ``r_i - R b_i``, then a plain Sim(3) Umeyama of the
        camera centres onto them, until the scale settles.

    The lever-arm term is scale-free once written with the metric ``a`` (the similarity
    scale and the unit-conversion 1/s cancel), leaving a rotation-only coupling with no
    finite closed form -- hence the fixed point. Ground truth is never used.
    :func:`nuslam.transforms.umeyama` (imported above) is the Sim(3) primitive to call
    for both the pre-iteration fit and each in-loop fit.

    Returns:
        :class:`ScaleResult` with the resolved ``scale`` and the alignment.
    """
    a = -sensor2ego[:3,:3].T @ sensor2ego[:3,3] # (3,)
    b = (world_from_cam[valid,:3,:3] @ a.reshape(1,3,1)).reshape(-1,3) # (N,3)
    r = np.concatenate([gps_xy[valid], np.zeros_like(gps_xy[valid])[:,:1]], axis=1) # (N,3)
    C = world_from_cam[valid,:3,3] # (N,3)
    s, R, t = umeyama(C, r, with_scale=True)
    for i in range(iters):
      s, R, t = umeyama(C, r - b @ R.T, with_scale=True)
    T = np.eye(4)
    T[:3, :3] = s*R
    T[:3, 3] = t
    ground = s*C @ R.T + t + b @ R.T
    aligned_xy = ground[:,:2]
    residual = np.linalg.norm(aligned_xy - gps_xy[valid], axis=1)
    return ScaleResult(s, T, aligned_xy, np.sqrt((residual**2).mean()), iters)