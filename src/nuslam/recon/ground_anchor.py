"""Ground-plane pre-alignment: GPS-free metric scale from the semantic ground.

Fit a Sim(3) that levels the reconstruction to the ground plane and fixes its
metric scale, using two residual families that both read only the *transformed
vertical*:

  * ground residual  -- road/ground-segmented points should lie on the plane z = 0;
  * pose height residual -- the camera centres sit at a known metric height ``h``
    above that plane (from calibration, not assumed).

This is the same height-residual scale disambiguation as the Theseus DEM tutorial,
with the DEM degenerate to a single plane. Because both residuals depend on the
transform only through the vertical coordinate ``z'(x) = m . x + b`` -- where
``m`` is the (unconstrained) third row of ``sR`` (up-direction times scale) and
``b = t_z`` -- only 4 of the 7 Sim(3) DoF are observable (scale, the two tilt
angles, and ``t_z``); yaw and the horizontal translation are a gauge the anchor
cannot see. The ground points carry the tilt; the *camera height* carries the
scale (a flat plane alone is scale-blind). See README "Ground-plane pre-alignment".

Authorship (CLAUDE.md): assembling the inputs (segment/back-project the ground,
gather camera centres, read the metric height from calibration), the result
container, and the caller/apply plumbing are infrastructure and live here. The
**fit itself** -- solving the two residuals for the up-direction and the metric
scale -- is the scale-resolution geometry (the novel core, section 3e) and is the
author's; :func:`fit_ground_anchor` returns just those two quantities, and
:func:`apply_ground_anchor` builds the leveling Sim(3) from them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_init import backproject_depth_to_world


@dataclass(frozen=True)
class GroundAnchorInputs:
    """Data the ground-anchor fit consumes, all in the *current* reconstruction frame.

    ``ground_points`` (N, 3) are the road/ground-segmented points, back-projected
    from the metric depth of every keyframe (already scale-corrected to whatever
    frame the caller passes -- for the GPS-free route this is the DA3 or scaled-COLMAP
    frame that still needs levelling). ``camera_centres`` (M, 3) are the training
    camera positions. ``camera_height`` is the known metric height of the camera
    above the road (metres), read from the ``sensor2ego`` extrinsic -- it enters
    the fit as the scale-bearing constant and is never assumed away.
    """

    ground_points: np.ndarray     # (N, 3) ground-segmented points, recon frame
    camera_centres: np.ndarray    # (M, 3) camera centres, recon frame
    camera_height: float          # metres, camera above road (from calibration)


@dataclass(frozen=True)
class GroundAnchor:
    """Result of the ground-plane pre-alignment fit (the author fills this in).

    ``up`` (3,) is the fitted unit up-direction in the reconstruction frame,
    ``scale`` its recovered metric scale (recon units -> metres), and ``offset`` the
    fitted vertical intercept ``b`` (the ``t_z`` of the leveled frame): with
    ``m = scale * up``, a point's leveled height is ``m . x + offset``, so ground
    points land at ~0 and cameras at ~``h``. :func:`apply_ground_anchor` builds the
    leveling + scaling Sim(3) from all three (rotating ``up`` -> +z, scaling by
    ``scale``, setting the vertical translation to ``offset`` and gauging the
    unobservable horizontal translation to minimal displacement).
    """

    up: np.ndarray                # (3,) unit up-direction in recon frame
    scale: float                  # recovered metric scale (recon -> metres)
    offset: float                 # fitted vertical intercept b (t_z of the leveled frame)


def ground_anchor_inputs(
    metric_depths: list[np.ndarray],   # per keyframe, metric depth (metres) in the recon frame
    images: list[np.ndarray],          # per keyframe (H, W, 3) uint8, for shapes only
    ground_masks: dict[str, np.ndarray],  # token -> (H, W) bool, True where road/ground
    tokens: list[str],                 # per keyframe token, aligned with the lists above
    K_true: np.ndarray,                # (3, 3) true intrinsics
    poses: np.ndarray,                 # (M, 4, 4) camera->world in the recon frame
    camera_height: float,              # metres, camera above road (from calibration)
    *,
    stride: int = 8,
    extra_drop: dict[str, np.ndarray] | None = None,  # token -> (H, W) bool, also excluded
) -> GroundAnchorInputs:
    """Assemble the ground-anchor inputs (infrastructure).

    For each keyframe, keep only the road/ground pixels (minus any ``extra_drop``,
    e.g. vehicle / far-range masks) and back-project its metric depth with the
    recon-frame pose to a world-frame point set; concatenate across keyframes to
    form ``ground_points``. The camera centres are taken straight from ``poses``.
    No fitting happens here -- this only gathers what :func:`fit_ground_anchor`
    needs.
    """
    chunks: list[np.ndarray] = []
    for depth, image, tok, pose in zip(metric_depths, images, tokens, poses):
        road = ground_masks.get(tok)
        if road is None:
            continue
        keep = np.asarray(road, bool)
        if extra_drop is not None and extra_drop.get(tok) is not None:
            keep = keep & ~np.asarray(extra_drop[tok], bool)
        # backproject uses ``sky`` as a DROP mask, so pass the complement of "keep".
        pts, _ = backproject_depth_to_world(
            depth, image, K_true, pose, stride=stride, sky=~keep)
        if len(pts):
            chunks.append(pts)
    ground_points = (np.concatenate(chunks, axis=0) if chunks
                     else np.empty((0, 3), dtype=float))
    return GroundAnchorInputs(
        ground_points=ground_points,
        camera_centres=np.asarray(poses)[:, :3, 3].copy(),
        camera_height=float(camera_height),
    )


def fit_ground_anchor(inputs: GroundAnchorInputs) -> GroundAnchor:
    """Fit the ground-plane pre-alignment Sim(3) (AUTHOR WRITES THIS -- section 3e).

    Solve for the up-direction and offset from the two vertical-only residuals
    (ground points -> plane, cameras -> height ``h``), recover the metric scale,
    gauge out the unobservable yaw / horizontal translation, and return the
    gauge-fixed :class:`GroundAnchor`. Robustify against segmentation bleed (RANSAC
    / IRLS) and gate on inlier support. The linear structure, the observability,
    and the weighting between the two families are written up in the README
    ("Ground-plane pre-alignment"); do not have the assistant implement the fit.
    """
    A = np.zeros((inputs.ground_points.shape[0]+inputs.camera_centres.shape[0], 4), dtype=np.float32)
    A[:inputs.ground_points.shape[0], :3] = inputs.ground_points/np.sqrt(inputs.ground_points.shape[0])
    A[inputs.ground_points.shape[0]:, :3] = inputs.camera_centres/np.sqrt(inputs.camera_centres.shape[0])
    A[:inputs.ground_points.shape[0], 3] = 1/np.sqrt(inputs.ground_points.shape[0])
    A[inputs.ground_points.shape[0]:, 3] = 1/np.sqrt(inputs.camera_centres.shape[0])

    b = np.zeros((inputs.ground_points.shape[0]+inputs.camera_centres.shape[0],), dtype=np.float32)
    b[inputs.ground_points.shape[0]:] = inputs.camera_height/np.sqrt(inputs.camera_centres.shape[0])

    reg = 1e-5    
    x = np.linalg.inv(A.T @ A + reg*np.eye(4, dtype=np.float32)) @ A.T @ b

    return GroundAnchor(x[:3]/np.linalg.norm(x[:3]), np.linalg.norm(x[:3]), x[3])


def ground_anchor_residual(anchor: GroundAnchor, inputs: GroundAnchorInputs) -> float:
    """RMS vertical residual of the fitted anchor, metres (infrastructure / eval).

    Scores the fit as returned: with ``m = scale * up`` and the fitted ``offset``,
    the leveled height of a ground point is ``m . p_i + offset`` (target 0) and of a
    camera ``m . c_j + offset`` (target ``h``); this returns the combined RMS. A
    diagnostic of how well the fitted up / scale / offset reconcile the ground plane
    with the known camera height -- not part of the fit.
    """
    up = np.asarray(anchor.up, dtype=float)
    m = float(anchor.scale) * up / np.linalg.norm(up)
    off = float(anchor.offset)
    g = np.asarray(inputs.ground_points, dtype=float) @ m + off    # ground -> target 0
    k = np.asarray(inputs.camera_centres, dtype=float) @ m + off   # camera -> target h
    h = float(inputs.camera_height)
    n = len(g) + len(k)
    if n == 0:
        return float("nan")
    r = np.concatenate([g, k - h])
    return float(np.sqrt(np.mean(r ** 2)))


def apply_ground_anchor(
    anchor: GroundAnchor,         # the fit's ``up`` + ``scale`` + ``offset``
    poses: np.ndarray,            # (M, 4, 4) camera->world to level
    points: np.ndarray,           # (P, 3) init cloud to level
) -> tuple[np.ndarray, np.ndarray]:
    """Level + scale the poses and the init cloud from the fitted anchor (infrastructure).

    Builds the pre-alignment Sim(3) from the fit: rotate the up-direction onto +z by
    the shortest-arc rotation (axis ``up x ez``, so yaw is untouched -- the
    minimal-displacement gauge), scale by ``anchor.scale``, and set the translation so
    the leveled vertical is ``m . x + offset`` (``m = scale * up``) -- i.e. ``t_z =
    offset``, placing the ground plane at ~0 and the cameras at ~``h``. The horizontal
    translation is the unobservable gauge, chosen to keep the camera centroid's x, y
    fixed. Returns ``(poses', points')``; call before training so the reconstruction
    starts levelled and metrically scaled.
    """
    up = np.asarray(anchor.up, dtype=float)
    up = up / np.linalg.norm(up)
    ez = np.array([0.0, 0.0, 1.0])
    v = np.cross(up, ez)
    c = float(np.dot(up, ez))
    if c > 1 - 1e-8:                     # already aligned
        R = np.eye(3)
    elif c < -1 + 1e-8:                  # antiparallel: 180 deg about any axis _|_ up
        a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(up, a); axis /= np.linalg.norm(axis)
        R = 2 * np.outer(axis, axis) - np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx / (1 + c)
    sR = float(anchor.scale) * R
    poses = np.asarray(poses, dtype=float)
    centroid = poses[:, :3, 3].mean(0)
    mapped = sR @ centroid
    t = np.array([centroid[0] - mapped[0],       # keep centroid x, y fixed (unobservable gauge)
                  centroid[1] - mapped[1],
                  float(anchor.offset)])          # vertical set by the fitted offset (t_z = b)
    # Similarity on camera->world poses: the ROTATION composes orthonormally (R . R_pose, no
    # scale -- a camera orientation stays a rotation), only the CENTRES scale/rotate/translate
    # (c' = sR c + t). The point cloud scales fully (x' = sR x + t).
    poses_out = poses.copy()
    poses_out[:, :3, :3] = R[None] @ poses[:, :3, :3]
    poses_out[:, :3, 3] = poses[:, :3, 3] @ sR.T + t
    points_out = np.asarray(points, dtype=float) @ sR.T + t
    return poses_out, points_out
