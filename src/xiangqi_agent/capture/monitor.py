from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame, FrameSource
from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad


class MonitorStatus(StrEnum):
    CONNECTING = "connecting"
    WATCHING = "watching"
    GEOMETRY_INVALID = "geometry_invalid"
    CLOSED = "closed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CaptureMonitorUpdate:
    status: MonitorStatus
    message: str
    hwnd: int | None = None
    frame_size: tuple[int, int] | None = None
    point_count: int = 0


class CaptureMonitor:
    """Validate a visible capture stream and fixed manual board geometry."""

    def __init__(
        self,
        source: FrameSource,
        quad: NormalizedQuad,
        *,
        orientation: Orientation = Orientation.RED_BOTTOM,
        on_update: Callable[[CaptureMonitorUpdate], None] | None = None,
    ) -> None:
        self._source = source
        self._quad = quad
        self._orientation = orientation
        self._on_update = on_update or _ignore_update
        self._lock = Lock()
        self._started = False
        self._closed = False
        self._target_closed = False
        self._frame_size: tuple[int, int] | None = None
        self._geometry_invalid = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("capture monitor has already started")
            if self._closed:
                raise RuntimeError("capture monitor is closed")
            self._started = True
        self._emit(CaptureMonitorUpdate(MonitorStatus.CONNECTING, "waiting for first frame"))
        try:
            self._source.start(self._on_frame, self._on_closed)
        except (OSError, RuntimeError, ValueError) as exc:
            self._emit(CaptureMonitorUpdate(MonitorStatus.ERROR, str(exc)))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._source.close()

    def _on_frame(self, frame: CaptureFrame) -> None:
        with self._lock:
            if self._closed or self._geometry_invalid:
                return
            expected_size = self._frame_size
            if expected_size is not None and frame.size != expected_size:
                self._geometry_invalid = True
                invalid = True
            else:
                invalid = False
                if expected_size is None:
                    self._frame_size = frame.size
        if invalid:
            self._emit(
                CaptureMonitorUpdate(
                    MonitorStatus.GEOMETRY_INVALID,
                    "frame size changed; manual calibration is no longer valid",
                    hwnd=frame.hwnd,
                    frame_size=frame.size,
                )
            )
            return
        if expected_size is not None:
            return
        geometry = BoardGeometry.from_quad(self._quad, frame.size, self._orientation)
        point_count = len(geometry.grid_points())
        self._emit(
            CaptureMonitorUpdate(
                MonitorStatus.WATCHING,
                "visible capture is stable; automatic board commit remains disabled",
                hwnd=frame.hwnd,
                frame_size=frame.size,
                point_count=point_count,
            )
        )

    def _on_closed(self, error: CaptureClosedError) -> None:
        with self._lock:
            if self._closed or self._target_closed:
                return
            self._target_closed = True
        self._emit(CaptureMonitorUpdate(MonitorStatus.CLOSED, str(error)))

    def _emit(self, update: CaptureMonitorUpdate) -> None:
        self._on_update(update)


def _ignore_update(_update: CaptureMonitorUpdate) -> None:
    pass
