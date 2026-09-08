"""Shared data types for the monocular SLAM pipeline.

These are the *data contract* between the plumbing (data loading, frontend
tracking/segmentation, viz, eval) and the estimator backend. They describe what
is available to the graph -- calibration, keyframe images, tracked
correspondences, ground masks, and the IMU/wheel/GPS streams -- without
prescribing how the factor graph consumes them. Deciding which of these become
variables and factors, and how, is core backend design (see ``nuslam.backend``).

Everything here is numpy / plain Python so the contract stays framework-neutral;
the frontend converts to torch internally only where a model needs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .transforms import SE3

# nuScenes monocular SLAM uses the forward camera by default; the pipeline is
# single-camera but the channel is configurable.
DEFAULT_CAMERA: str = "CAM_FRONT"


@dataclass(frozen=True)
class CameraCalib:
    """Intrinsics and the (static) sensor->ego extrinsic for one camera channel.

    ``intrinsic`` is the pinhole K (3x3). ``sensor2ego`` is fixed per channel
    from ``calibrated_sensor``. Distortion is not modelled -- nuScenes camera
    images are already rectified/undistorted for the provided intrinsics.
    """

    channel: str
    intrinsic: np.ndarray  # (3, 3)
    sensor2ego: SE3
    width: int
    height: int

    @property
    def fx(self) -> float:
        return float(self.intrinsic[0, 0])

    @property
    def fy(self) -> float:
        return float(self.intrinsic[1, 1])

    @property
    def cx(self) -> float:
        return float(self.intrinsic[0, 2])

    @property
    def cy(self) -> float:
        return float(self.intrinsic[1, 2])


@dataclass(frozen=True)
class Keyframe:
    """One monocular keyframe (a nuScenes ``sample``), in temporal order.

    Carries the on-disk image path (decoded lazily via :meth:`image`), the camera
    calibration, and the ground-truth ego pose for *evaluation only*. The
    estimator must not consume ``ego2global_gt`` as an input -- it is nuScenes
    ground truth used to score the trajectory (see ``nuslam.eval``).
    """

    token: str  # nuScenes sample token
    scene_name: str
    frame_index: int  # 0-based position within the scene
    timestamp_us: int
    image_path: Path
    calib: CameraCalib
    ego2global_gt: SE3  # GROUND TRUTH ego pose (from ego_pose); eval/anchor only

    def image(self) -> np.ndarray:
        """Decode the RGB image as an (H, W, 3) uint8 array."""
        with Image.open(self.image_path) as img:
            return np.asarray(img.convert("RGB"))

    def image_bytes(self) -> bytes:
        """Raw encoded JPEG bytes, for zero-decode publishing to viz."""
        return self.image_path.read_bytes()


@dataclass(frozen=True)
class TrackSet:
    """CoTracker output over a contiguous window of keyframes.

    ``points`` are pixel coordinates ``(T, N, 2)`` for ``N`` tracks across ``T``
    keyframes; ``visible`` is the matching ``(T, N)`` boolean mask. ``frame_tokens``
    lines the ``T`` axis up with the keyframes that produced it. ``is_ground`` (if
    set) flags, per track, whether it landed on the road mask at its own seed
    frame -- the candidate set for ground-anchored landmarks. Whether/how to trust
    that flag is the backend's call.

    With grid replenishment tracks are seeded at different frames, so ``seed_frame``
    (if set) gives each track's first observed frame index; a track is invisible
    before it. ``query_frame`` records the initial full-grid seeding frame (0).
    """

    frame_tokens: list[str]
    points: np.ndarray  # (T, N, 2) float32, pixel (x, y)
    visible: np.ndarray  # (T, N) bool
    query_frame: int  # index into frame_tokens where the initial grid was seeded
    is_ground: np.ndarray | None = None  # (N,) bool, optional
    seed_frame: np.ndarray | None = None  # (N,) int, per-track first frame; None => all at query_frame

    @property
    def num_tracks(self) -> int:
        return int(self.points.shape[1])

    @property
    def num_frames(self) -> int:
        return int(self.points.shape[0])


@dataclass(frozen=True)
class GroundMask:
    """Per-keyframe road/ground segmentation.

    ``mask`` is (H, W) bool at image resolution: True where a pixel is road/ground.
    ``prob`` keeps the raw score for thresholding experiments. Produced offline by
    the segmentation frontend; the backend decides how (and whether) to gate the
    wheel-contact / ground-plane factors on it.
    """

    token: str
    mask: np.ndarray  # (H, W) bool
    prob: np.ndarray | None = None  # (H, W) float32 in [0, 1]

    def sample_at(self, points_xy: np.ndarray) -> np.ndarray:
        """Nearest-pixel mask lookup for an (N, 2) array of pixel coords."""
        h, w = self.mask.shape
        xy = np.round(np.asarray(points_xy)).astype(int)
        x = np.clip(xy[:, 0], 0, w - 1)
        y = np.clip(xy[:, 1], 0, h - 1)
        return self.mask[y, x]


@dataclass(frozen=True)
class DepthMap:
    """Per-keyframe monocular depth (from Depth Anything 3), for initialization.

    ``depth`` is (H, W) float32 at the keyframe's full image resolution, paired
    with ``calib.intrinsic``. From DA3Mono it is RELATIVE (scale-ambiguous):
    ``is_metric`` is False and the metric scale is exactly what the ground/wheel
    anchor must recover -- so the reconstruction is seeded at the wrong scale on
    purpose. ``conf`` (higher = more reliable) and ``sky`` gate which pixels to
    trust or exclude when back-projecting.

    ``extrinsic`` and ``intrinsic`` are populated only by the DA3-Base *full*
    reconstruction (``DA3ReconEstimator``), not by DA3Mono. They carry DA3's own
    per-frame pose and estimated calibration:

      * ``extrinsic`` (4, 4): DA3's world->camera pose ``[R_i | t_i]`` (homogeneous),
        in DA3's own reconstruction frame. (DA3 stores extrinsics world->camera;
        verify the convention with a reprojection before trusting it.)
      * ``intrinsic`` (3, 3): DA3's estimated ``K_da3``, rescaled to full image
        resolution to pair with ``depth`` and ``calib.intrinsic``.

    DA3-Base is intrinsic-agnostic: ``intrinsic`` here is DA3's *own* (typically
    wrong) estimate, NOT the true calibration. Feeding ``(extrinsic, intrinsic)``
    plus the true ``calib.intrinsic`` into the metric upgrade
    (``nuslam.recon.metric_upgrade``) is how the reconstruction is rectified to
    metric. DA3's predicted gaussians remain unused.
    """

    token: str
    depth: np.ndarray  # (H, W) float32, relative unless is_metric
    is_metric: bool = False
    conf: np.ndarray | None = None  # (H, W) float32 confidence
    sky: np.ndarray | None = None   # (H, W) bool, sky pixels to exclude
    extrinsic: np.ndarray | None = None  # (4, 4) DA3 world->camera pose (DA3-Base only)
    intrinsic: np.ndarray | None = None  # (3, 3) DA3 estimated K, full-res (DA3-Base only)


@dataclass(frozen=True)
class MetricUpgrade:
    """Result of the DA3 metric upgrade (Stage 1), cached for the scale/Gaussian stages.

    Everything needed downstream so the DA3 forward pass and the DAQ solve do not
    rerun: the rectifier ``H``, the recovered metric cameras/poses (metric up to one
    global scale), the true intrinsics they pair with, and the solve diagnostics.
    ``world_from_cam`` is camera->world in the metric frame (camera 0 at the origin);
    the corrected depth for any frame is recovered from ``H`` + the cached DA3 depth.
    """

    tokens: list[str]                    # frame order (aligns the N axis)
    H: np.ndarray                        # (4, 4) rectifier, projective -> metric
    omega_star: np.ndarray               # (4, 4) dual absolute quadric (projective frame)
    plane_at_infinity: np.ndarray        # (4,) (v, s)
    world_from_cam: np.ndarray           # (N, 4, 4) recovered metric camera->world (up to global scale)
    metric_cameras: np.ndarray           # (N, 3, 4) P_metric = P~ H^{-1}
    K_recovered: np.ndarray              # (N, 3, 3) decomposed K (should be ~ [c, c, 1])
    K_true: np.ndarray                   # (3, 3) true intrinsics the metric cameras use
    diagnostics: dict = field(default_factory=dict)  # DAQ diagnostics scalars


# --- Proprioceptive / global streams (nuScenes-CAN + GPS) -----------------
# These back build ladder rungs 3-5 (IMU, wheel odometry, GPS). They are present
# only if the nuScenes-CAN expansion is downloaded; the loader returns empty lists
# otherwise (see nuslam.data.can_streams). Raw measurements only -- preintegration,
# velocity factors and the GPS robust kernel are part of the estimator design.


@dataclass(frozen=True)
class ImuSample:
    timestamp_us: int
    linear_accel: np.ndarray  # (3,) m/s^2, sensor frame
    angular_rate: np.ndarray  # (3,) rad/s, sensor frame


@dataclass(frozen=True)
class WheelOdometry:
    timestamp_us: int
    speed: float  # m/s (vehicle_monitor / zoe wheel speeds, vehicle frame +x)


@dataclass(frozen=True)
class GpsFix:
    timestamp_us: int
    latitude: float
    longitude: float
    # nuScenes CAN pose is already in the local map/global frame; when a fix is
    # derived from it we also carry that directly for convenience.
    global_xy: np.ndarray | None = None  # (2,) meters in the scene's global frame


@dataclass(frozen=True)
class ProprioStreams:
    """All non-visual measurements for a scene, in timestamp order."""

    imu: list[ImuSample] = field(default_factory=list)
    wheel: list[WheelOdometry] = field(default_factory=list)
    gps: list[GpsFix] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return bool(self.imu or self.wheel or self.gps)
