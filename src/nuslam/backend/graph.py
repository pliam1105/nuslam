"""Factor-graph estimator -- the body is left unimplemented by design.

    ############################################################################
    #  The factor-graph DESIGN is core estimator substance: which variables,   #
    #  which factors, how they connect, batch vs. incremental (ISAM2), the      #
    #  state definition, and the scale-resolution logic. This file provides the #
    #  seam the surrounding pipeline plugs into -- the inputs the graph receives #
    #  and the estimate it returns -- and nothing more.                         #
    ############################################################################

Everything upstream (data streaming, tracks, road masks, IMU/wheel/GPS streams)
and everything downstream (trajectory viz, ATE/RPE eval) is built and runnable.
``scripts/run_slam.py`` calls :meth:`MonocularSLAM.run`, hands it the
fully-populated :class:`SlamInputs`, and routes the returned :class:`SlamEstimate`
into the viz + eval path. The body is built incrementally along the build ladder:

    rung 1  poses + landmarks + reprojection (Huber)     -> observe scale ambiguity
    rung 2  + RANSAC ground plane + wheel-contact         -> scale should resolve
    rung 3  + IMU preintegration
    rung 4  + wheel-odometry velocity
    rung 5  + GPS (loose, then robustified)

Only :class:`SlamInputs` (what the graph consumes) and :class:`SlamEstimate`
(what viz/eval consume) are fixed here, so the surrounding pipeline has a stable
contract. Extend :class:`SlamEstimate` as the state grows (velocities, biases,
the plane).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import CameraCalib, GroundMask, Keyframe, ProprioStreams, TrackSet


@dataclass
class SlamInputs:
    """Everything the estimator is given for one scene.

    This is the graph's input contract, assembled by the pipeline. How these map
    onto variables and factors -- and which to trust where -- is a design choice.
    """

    calib: CameraCalib
    keyframes: list[Keyframe]                     # temporal order
    tracks: TrackSet                              # tracked correspondences
    masks: dict[str, GroundMask] = field(default_factory=dict)  # token -> road mask
    proprio: ProprioStreams = field(default_factory=ProprioStreams)  # IMU/wheel/GPS


@dataclass
class SlamEstimate:
    """Estimator output consumed by viz (``nuslam.viz``) and eval (``nuslam.eval``).

    Minimal by design: the estimated ego->global trajectory aligned to
    ``tokens``, and optional landmark points for the reconstruction overlay.
    Extend with velocities/biases/plane as the state grows -- viz/eval read what
    is present.
    """

    tokens: list[str]                             # keyframe tokens, in order
    poses_ego2global: np.ndarray                  # (T, 4, 4) estimated poses
    landmarks: np.ndarray | None = None           # (K, 3) points in the global frame
    landmark_is_ground: np.ndarray | None = None  # (K,) bool, optional

    def __post_init__(self) -> None:
        poses = np.asarray(self.poses_ego2global, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses_ego2global must be (T, 4, 4), got {poses.shape}")
        if len(self.tokens) != poses.shape[0]:
            raise ValueError("tokens and poses length mismatch")
        self.poses_ego2global = poses


class MonocularSLAM:
    """Monocular metric-reconstruction estimator. :meth:`run` is unimplemented by design.

    The pipeline constructs this with the camera calibration and a free-form
    config, then calls :meth:`run` once per scene. Whether ``run`` builds a batch
    graph or drives ISAM2 keyframe-by-keyframe internally is a design choice --
    the seam only fixes inputs and outputs.
    """

    def __init__(self, calib: CameraCalib, config: dict | None = None) -> None:
        self.calib = calib
        self.config = config or {}

    def run(self, inputs: SlamInputs) -> SlamEstimate:
        """Estimate the trajectory (and landmarks) for one scene. Built here.

        Build the factor graph in this method: define the state, add the
        reprojection / ground-plane / wheel-contact / IMU / wheel-odometry / GPS
        factors per the build ladder, optimize (Levenberg-Marquardt or ISAM2),
        and return a :class:`SlamEstimate`.
        """
        raise NotImplementedError(
            "MonocularSLAM.run is the factor graph itself -- core estimator substance, "
            "left unimplemented by design and built here. Everything feeding it (inputs) "
            "and consuming it (SlamEstimate -> viz/eval) is ready; run scripts/run_slam.py "
            "to see the fully-populated SlamInputs reach this seam."
        )
