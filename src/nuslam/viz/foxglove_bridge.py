"""Live Foxglove WebSocket bridge for raw nuScenes visualization.

Runs a Foxglove WebSocket server in a background thread and exposes synchronous
``publish_*`` methods, so plain processing code can push frames as it walks a
scene. Connect the Foxglove app to ``ws://localhost:8765``.

Channels are created lazily on first publish. Each publish method maps onto a
Foxglove well-known schema (CompressedImage, PointCloud, FrameTransform,
SceneUpdate) so the app's Image and 3D panels render everything natively; the
3D panel composes the transform tree itself from the FrameTransforms published
here.

Extending: add a method that builds a dict matching a Foxglove schema, register
its schema once in ``_SCHEMAS``, and call ``_publish(topic, schema_name, msg)``.
"""
from __future__ import annotations

import asyncio
import base64
import json
import threading
from typing import Any, Iterable

import numpy as np

from foxglove_websocket.server import FoxgloveServer


def _stamp(timestamp_us: int) -> dict[str, int]:
    sec, us = divmod(int(timestamp_us), 1_000_000)
    return {"sec": sec, "nsec": us * 1000}


def _quat_wxyz_to_xyzw(q: np.ndarray) -> dict[str, float]:
    # nuScenes stores quaternions as wxyz; Foxglove messages want xyzw fields.
    w, x, y, z = (float(v) for v in q)
    return {"x": x, "y": y, "z": z, "w": w}


def _vec3(t: np.ndarray) -> dict[str, float]:
    return {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])}


# Foxglove well-known JSON schemas, trimmed to the fields used here. bytes fields
# are base64 strings under JSON encoding.
_SCHEMAS: dict[str, dict[str, Any]] = {
    "foxglove.CompressedImage": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
            },
            "frame_id": {"type": "string"},
            "data": {"type": "string", "contentEncoding": "base64"},
            "format": {"type": "string"},
        },
    },
    "foxglove.PointCloud": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
            },
            "frame_id": {"type": "string"},
            "pose": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                        },
                    },
                    "orientation": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "z": {"type": "number"},
                            "w": {"type": "number"},
                        },
                    },
                },
            },
            "point_stride": {"type": "integer"},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "offset": {"type": "integer"},
                        "type": {"type": "integer"},
                    },
                },
            },
            "data": {"type": "string", "contentEncoding": "base64"},
        },
    },
    "foxglove.FrameTransform": {
        "type": "object",
        "properties": {
            "timestamp": {
                "type": "object",
                "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}},
            },
            "parent_frame_id": {"type": "string"},
            "child_frame_id": {"type": "string"},
            "translation": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                },
            },
            "rotation": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                    "w": {"type": "number"},
                },
            },
        },
    },
    "foxglove.SceneUpdate": {
        "type": "object",
        "properties": {
            "deletions": {"type": "array", "items": {"type": "object"}},
            "entities": {"type": "array", "items": {"type": "object"}},
        },
    },
}

# foxglove.PackedElementField numeric type enum.
_FIELD_UINT8 = 1
_FIELD_FLOAT32 = 7


class FoxgloveBridge:
    """A running Foxglove WebSocket server with synchronous publish methods.

    Use as a context manager::

        with FoxgloveBridge() as bridge:
            bridge.publish_transform(...)
            bridge.publish_compressed_image(...)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, name: str = "nuslam-viz") -> None:
        self.host = host
        self.port = port
        self.name = name
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="foxglove-bridge", daemon=True)
        self._server: FoxgloveServer | None = None
        self._ready = threading.Event()
        self._stop_event: asyncio.Event | None = None
        self._channels: dict[str, int] = {}
        self._channels_lock = threading.Lock()

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        self._ready.wait()

    def stop(self) -> None:
        # Signal the coroutine to leave its `async with`, so FoxgloveServer
        # shuts down cleanly before the loop thread exits.
        if self._stop_event is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)

    def __enter__(self) -> "FoxgloveBridge":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_forever())
        finally:
            self._loop.close()

    async def _serve_forever(self) -> None:
        self._stop_event = asyncio.Event()
        async with FoxgloveServer(self.host, self.port, self.name) as server:
            self._server = server
            self._ready.set()
            await self._stop_event.wait()

    # ---- channel plumbing ------------------------------------------------

    def _channel(self, topic: str, schema_name: str) -> int:
        with self._channels_lock:
            if topic in self._channels:
                return self._channels[topic]
            fut = asyncio.run_coroutine_threadsafe(
                self._add_channel(topic, schema_name), self._loop
            )
            chan_id = fut.result(timeout=5)
            self._channels[topic] = chan_id
            return chan_id

    async def _add_channel(self, topic: str, schema_name: str) -> int:
        assert self._server is not None
        return await self._server.add_channel(
            {
                "topic": topic,
                "encoding": "json",
                "schemaName": schema_name,
                "schema": json.dumps(_SCHEMAS[schema_name]),
                "schemaEncoding": "jsonschema",
            }
        )

    def _publish(self, topic: str, schema_name: str, msg: dict[str, Any], timestamp_us: int) -> None:
        chan_id = self._channel(topic, schema_name)
        payload = json.dumps(msg).encode("utf-8")
        asyncio.run_coroutine_threadsafe(
            self._server.send_message(chan_id, int(timestamp_us) * 1000, payload),  # type: ignore[union-attr]
            self._loop,
        )

    # ---- publish methods -------------------------------------------------

    def publish_transform(
        self,
        topic: str,
        parent_frame: str,
        child_frame: str,
        translation: np.ndarray,
        rotation_wxyz: np.ndarray,
        timestamp_us: int,
    ) -> None:
        msg = {
            "timestamp": _stamp(timestamp_us),
            "parent_frame_id": parent_frame,
            "child_frame_id": child_frame,
            "translation": _vec3(translation),
            "rotation": _quat_wxyz_to_xyzw(rotation_wxyz),
        }
        self._publish(topic, "foxglove.FrameTransform", msg, timestamp_us)

    def publish_compressed_image(
        self,
        topic: str,
        frame_id: str,
        jpeg_bytes: bytes,
        timestamp_us: int,
        image_format: str = "jpeg",
    ) -> None:
        msg = {
            "timestamp": _stamp(timestamp_us),
            "frame_id": frame_id,
            "data": base64.b64encode(jpeg_bytes).decode("ascii"),
            "format": image_format,
        }
        self._publish(topic, "foxglove.CompressedImage", msg, timestamp_us)

    def publish_pointcloud(
        self,
        topic: str,
        frame_id: str,
        points_xyz: np.ndarray,
        timestamp_us: int,
        rgb: tuple[int, int, int] | None = None,
    ) -> None:
        """Publish an (N, >=3) float array (x, y, z).

        With `rgb` (0-255 per channel), a red/green/blue byte triple is packed
        per point so the color travels in the data and does not depend on the
        viewer's per-topic color setting.
        """
        xyz = np.ascontiguousarray(points_xyz[:, :3], dtype=np.float32)
        n = len(xyz)
        fields = [
            {"name": "x", "offset": 0, "type": _FIELD_FLOAT32},
            {"name": "y", "offset": 4, "type": _FIELD_FLOAT32},
            {"name": "z", "offset": 8, "type": _FIELD_FLOAT32},
        ]
        if rgb is None:
            stride = 12
            buffer = xyz.tobytes()
        else:
            stride = 16
            packed = np.zeros((n, stride), dtype=np.uint8)
            packed[:, :12] = xyz.view(np.uint8).reshape(n, 12)
            packed[:, 12:15] = np.asarray(rgb, dtype=np.uint8)
            packed[:, 15] = 255  # alpha
            fields += [
                {"name": "red", "offset": 12, "type": _FIELD_UINT8},
                {"name": "green", "offset": 13, "type": _FIELD_UINT8},
                {"name": "blue", "offset": 14, "type": _FIELD_UINT8},
                {"name": "alpha", "offset": 15, "type": _FIELD_UINT8},
            ]
            buffer = packed.tobytes()
        msg = {
            "timestamp": _stamp(timestamp_us),
            "frame_id": frame_id,
            "pose": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "point_stride": stride,
            "fields": fields,
            "data": base64.b64encode(buffer).decode("ascii"),
        }
        self._publish(topic, "foxglove.PointCloud", msg, timestamp_us)

    def publish_line(
        self,
        topic: str,
        frame_id: str,
        points_xyz: np.ndarray,
        timestamp_us: int,
        color: tuple[float, float, float, float] = (0.2, 1.0, 0.4, 1.0),
        thickness: float = 0.2,
        entity_id: str = "line",
    ) -> None:
        """Render an (N, >=3) polyline as a SceneUpdate line strip.

        Used for trajectories (estimated vs. ground truth). ``thickness`` is in
        world metres. Colors are RGBA in 0..1.
        """
        xyz = np.asarray(points_xyz, dtype=np.float64)[:, :3]
        r, g, b, a = color
        entity = {
            "timestamp": _stamp(timestamp_us),
            "frame_id": frame_id,
            "id": entity_id,
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": False,
            "lines": [
                {
                    "type": 0,  # LINE_STRIP
                    "thickness": float(thickness),
                    "scale_invariant": False,
                    "points": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in xyz],
                    "color": {"r": r, "g": g, "b": b, "a": a},
                }
            ],
        }
        self._publish(topic, "foxglove.SceneUpdate", {"deletions": [], "entities": [entity]}, timestamp_us)

    def publish_boxes(
        self,
        topic: str,
        frame_id: str,
        annotations: Iterable[Any],
        timestamp_us: int,
        color: tuple[float, float, float, float] = (0.2, 0.7, 1.0, 0.4),
    ) -> None:
        """Render annotation boxes as a SceneUpdate of cubes.

        Each item is a ``bev.types.Annotation`` whose ``.box`` is a nuScenes Box
        in ``frame_id``. nuScenes ``wlh`` (width, length, height) maps to a
        Foxglove cube size of (length, width, height) along x, y, z.
        """
        r, g, b, a = color
        cubes = []
        for ann in annotations:
            box = ann.box
            w, length, h = (float(v) for v in box.wlh)
            cubes.append(
                {
                    "pose": {
                        "position": _vec3(box.center),
                        "orientation": _quat_wxyz_to_xyzw(box.orientation.elements),
                    },
                    "size": {"x": length, "y": w, "z": h},
                    "color": {"r": r, "g": g, "b": b, "a": a},
                }
            )
        entity = {
            "timestamp": _stamp(timestamp_us),
            "frame_id": frame_id,
            "id": "boxes",
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": False,
            "cubes": cubes,
        }
        self._publish(topic, "foxglove.SceneUpdate", {"deletions": [], "entities": [entity]}, timestamp_us)
