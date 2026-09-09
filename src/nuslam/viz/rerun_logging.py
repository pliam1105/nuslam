"""Rerun logging for nuScenes raw data and reconstruction output.

Rerun (https://rerun.io) is the viewer for this project: it renders camera frusta,
images, point clouds, poses/trajectories, and 3D Gaussian splats natively -- the
things a monocular splat reconstruction needs to inspect.

Entity hierarchy (transforms compose down the path):

    world                      global frame (right-handed, Z up)
    world/ego                  ego pose (ego -> global)
    world/ego/<cam>            sensor pose (sensor -> ego)
    world/ego/<cam>/image      Pinhole + the camera image (ground mask blended in)
    world/traj/gt|est          trajectories as line strips
    world/gaussians            the reconstruction (GaussianSplats3D)
    world/landmarks            explicit points (Points3D)

Visualization infrastructure only: it renders whatever geometry it is handed and
makes no modelling or estimation decisions.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rerun as rr

from ..types import Keyframe

WORLD = "world"


def _ensure_viewer_on_path() -> None:
    """rr.spawn() searches PATH for the viewer, but the binary ships inside the
    wheel (rerun_sdk/rerun_cli/rerun) and .venv/bin is not on PATH under a direct
    `.venv/bin/python` call. Prepend the bundled viewer's directory so spawn finds it.
    """
    try:
        import rerun_cli

        bindir = os.path.dirname(rerun_cli.__file__)
        if bindir not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass  # fall back to whatever the SDK can find; it will error clearly if absent


def init(app_id: str = "nuslam", *, spawn: bool = False,
         save: Path | str | None = None, connect: str | None = None) -> None:
    """Start a recording. One of: spawn a viewer, save a .rrd, or connect to one.

    ``save`` writes a recording file openable later with ``rerun <file>.rrd``;
    ``spawn`` opens the native viewer; ``connect`` (a gRPC url) attaches to a
    running viewer. Defaults to an in-memory recording if none is given.
    """
    if spawn:
        _ensure_viewer_on_path()
    rr.init(app_id, spawn=spawn)
    if save is not None:
        rr.save(str(save))
    elif connect is not None:
        rr.connect_grpc(connect)
    # nuScenes global frame is a local map frame: right-handed, Z up.
    rr.log(WORLD, rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def _quat_xyzw(wxyz: np.ndarray) -> list[float]:
    w, x, y, z = (float(v) for v in wxyz)
    return [x, y, z, w]


def set_time(frame_index: int, timestamp_us: int, t0_us: int) -> None:
    rr.set_time("frame", sequence=int(frame_index))
    rr.set_time("time", duration=(int(timestamp_us) - int(t0_us)) / 1e6)


_GROUND_RGB = np.array([80, 220, 120], dtype=np.float32)  # road overlay color


def log_keyframe(kf: Keyframe, t0_us: int, *, image: bool = True,
                 ground_mask: np.ndarray | None = None, mask_alpha: float = 0.45) -> None:
    """Log one keyframe: ego + camera poses, the pinhole, and the image.

    ``ground_mask`` (H, W bool) is blended onto the image in place -- one image
    plane, no separate overlay entity. Without it the JPEG is logged un-decoded.
    """
    set_time(kf.frame_index, kf.timestamp_us, t0_us)
    ego = kf.ego2global_gt
    rr.log(f"{WORLD}/ego", rr.Transform3D(
        translation=ego.t, quaternion=rr.Quaternion(xyzw=_quat_xyzw(ego.quaternion_wxyz()))))
    cam = f"{WORLD}/ego/{kf.calib.channel}"
    s2e = kf.calib.sensor2ego
    rr.log(cam, rr.Transform3D(
        translation=s2e.t, quaternion=rr.Quaternion(xyzw=_quat_xyzw(s2e.quaternion_wxyz()))))
    rr.log(f"{cam}/image", rr.Pinhole(
        image_from_camera=kf.calib.intrinsic,
        resolution=[kf.calib.width, kf.calib.height],
        camera_xyz=rr.ViewCoordinates.RDF,  # nuScenes camera: x right, y down, z forward
    ))
    if not image:
        return
    if ground_mask is None:
        rr.log(f"{cam}/image", rr.EncodedImage(contents=kf.image_bytes(), media_type="image/jpeg"))
    else:
        # blend the mask in, then re-encode to JPEG so the recording stays small
        # (a raw image is ~25x larger) while remaining a single image plane.
        img = kf.image().astype(np.float32)
        m = np.asarray(ground_mask, bool)
        img[m] = mask_alpha * _GROUND_RGB + (1.0 - mask_alpha) * img[m]
        rr.log(f"{cam}/image", rr.EncodedImage(contents=_jpeg(img.astype(np.uint8)),
                                               media_type="image/jpeg"))


def _jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def log_estimated_camera(name: str, world_from_cam: np.ndarray, K: np.ndarray,
                         width: int, height: int, *, channel: str = "cam",
                         image_plane_distance: float | None = None,
                         color: tuple[int, int, int] | None = None,
                         static: bool = False) -> None:
    """Log a recovered (estimated) camera: its pose + pinhole, under world/<name>.

    For the DA3 metric-upgrade path: ``world_from_cam`` is the recovered metric
    camera-to-world pose (4x4) and ``K`` the true intrinsics. Kept on a separate
    entity path from the GT ego chain (``log_keyframe``) so estimate and ground
    truth can be shown together. nuScenes camera convention is RDF.

    ``image_plane_distance`` sets how far out the frustum is drawn (scene units);
    pass a fraction of the cloud depth so the pyramid matches the points rather
    than dwarfing them. ``None`` leaves Rerun's default.
    """
    from ..transforms import SE3

    pose = SE3.from_matrix(np.asarray(world_from_cam, float))
    ent = f"{WORLD}/{name}/{channel}"
    rr.log(ent, rr.Transform3D(
        translation=pose.t, quaternion=rr.Quaternion(xyzw=_quat_xyzw(pose.quaternion_wxyz()))),
        static=static)
    kwargs = {} if image_plane_distance is None else {"image_plane_distance": float(image_plane_distance)}
    if color is not None:
        kwargs["color"] = color
    rr.log(f"{ent}/image", rr.Pinhole(
        image_from_camera=np.asarray(K, float),
        resolution=[int(width), int(height)],
        camera_xyz=rr.ViewCoordinates.RDF,
        **kwargs,
    ), static=static)


def log_trajectory(name: str, points_xyz: np.ndarray,
                   color: tuple[int, int, int] = (150, 150, 150), radius: float = 0.1,
                   *, static: bool = False) -> None:
    """Log an (N, 3) polyline (e.g. a trajectory) as a line strip under world/<name>.

    Pass ``static=True`` for a whole-scene path logged outside the per-frame loop, so
    it shows at every time cursor rather than only on the timeline value it happened
    to be logged at (a strip logged before any ``set_time`` is otherwise absent from
    the frame timeline the viewer scrubs)."""
    pts = np.asarray(points_xyz, dtype=np.float32)[:, :3]
    rr.log(f"{WORLD}/{name}", rr.LineStrips3D([pts], colors=[color], radii=radius), static=static)


def log_points(name: str, xyz: np.ndarray, *,
               colors: np.ndarray | tuple | None = None, radii: float | np.ndarray = 0.05,
               static: bool = False) -> None:
    """Log an (N, 3) point cloud under world/<name>.

    Pass ``static=True`` for a whole-scene cloud logged outside the per-frame loop
    (e.g. the Gaussian seed cloud), so it shows at every time cursor instead of only
    on the timeline value it happened to be logged at."""
    rr.log(f"{WORLD}/{name}", rr.Points3D(np.asarray(xyz, np.float32)[:, :3], colors=colors, radii=radii),
           static=static)


def log_image(name: str, image: np.ndarray, *, static: bool = False) -> None:
    """Log a 2D image (a rendered view or its GT keyframe) under ``<name>``.

    Use a 2D entity path (e.g. ``"render/est"``, ``"render/gt"``) so Rerun shows it
    in an image view beside the 3D scene -- the rendered-vs-GT comparison the ladder
    is scored on. ``image`` is (H, W, 3) uint8 or float; floats are shown as-is."""
    rr.log(name, rr.Image(np.asarray(image)), static=static)


def log_gaussians(name: str, means: np.ndarray, scales: np.ndarray,
                  quats_wxyz: np.ndarray, colors_rgba: np.ndarray, *, static: bool = False) -> None:
    """Log a 3D Gaussian-splat set (GaussianSplats3D archetype).

    ``means`` (N, 3), ``scales`` (N, 3) linear standard deviations in scene units,
    ``quats_wxyz`` (N, 4) unit quaternions (nuScenes/torch wxyz order), and
    ``colors_rgba`` (N, 4) uint8. Quaternions are reordered to Rerun's xyzw.
    """
    q = np.asarray(quats_wxyz, np.float32)[:, [1, 2, 3, 0]]  # wxyz -> xyzw
    rr.log(f"{WORLD}/{name}", rr.GaussianSplats3D(
        centers=np.asarray(means, np.float32),
        scales=np.asarray(scales, np.float32),
        quaternions=q,
        colors=np.asarray(colors_rgba, np.uint8),
    ), static=static)


