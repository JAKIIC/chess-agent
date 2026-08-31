from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame, FrameSource
from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.vision.geometry import BoardGeometry, GeometryError, NormalizedQuad


class MonitorStatus(StrEnum):
    CONNECTING = "connecting"
    WATCHING = "watching"
    GEOMETRY_REBOUND = "geometry_rebound"
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
        self._geometry: BoardGeometry | None = None
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
            geometry = self._geometry
            if geometry is None:
                geometry = BoardGeometry.from_quad(self._quad, frame.size, self._orientation)
                self._geometry = geometry
                self._frame_size = frame.size
                status = MonitorStatus.WATCHING
            elif frame.size != geometry.frame_size:
                try:
                    geometry = geometry.rebind(frame.size)
                except GeometryError as exc:
                    self._geometry_invalid = True
                    error = str(exc)
                    status = MonitorStatus.GEOMETRY_INVALID
                else:
                    self._geometry = geometry
                    self._frame_size = frame.size
                    status = MonitorStatus.GEOMETRY_REBOUND
            else:
                return
        if status is MonitorStatus.GEOMETRY_INVALID:
            self._emit(
                CaptureMonitorUpdate(
                    MonitorStatus.GEOMETRY_INVALID,
                    error,
                    hwnd=frame.hwnd,
                    frame_size=frame.size,
                )
            )
            return
        point_count = len(geometry.grid_points())
        if status is MonitorStatus.GEOMETRY_REBOUND:
            message = "frame size changed proportionally; normalized calibration was rebound"
        else:
            message = "visible capture is stable; automatic board commit remains disabled"
        self._emit(
            CaptureMonitorUpdate(
                status,
                message,
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
