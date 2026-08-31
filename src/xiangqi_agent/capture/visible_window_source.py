from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from ctypes import wintypes
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns
from typing import Any, Protocol, cast

import numpy as np

from xiangqi_agent.capture.protocol import (
    CaptureClosedError,
    CaptureFrame,
    ClosedCallback,
    FrameCallback,
)
from xiangqi_agent.platform.windows import WindowInfo


class _ScreenGrabber(Protocol):
    def grab(self, region: Mapping[str, int]) -> object: ...

    def close(self) -> None: ...


class _DxgiCamera(Protocol):
    def grab(
        self,
        region: tuple[int, int, int, int],
        *,
        copy: bool,
        new_frame_only: bool,
    ) -> object | None: ...

    def release(self) -> None: ...


type GrabberFactory = Callable[[], _ScreenGrabber]
type RegionProvider = Callable[[int], "CaptureRegion"]


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("capture region must have positive dimensions")

    def as_monitor(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


class DxgiDesktopGrabber:
    """Read a visible desktop region through DXGI Desktop Duplication."""

    def __init__(self, camera: _DxgiCamera) -> None:
        self._camera = camera
        self._closed = False

    def grab(self, region: Mapping[str, int]) -> object:
        left = int(region["left"])
        top = int(region["top"])
        right = left + int(region["width"])
        bottom = top + int(region["height"])
        frame = self._camera.grab(
            (left, top, right, bottom),
            copy=True,
            new_frame_only=False,
        )
        if frame is None:
            raise RuntimeError("desktop duplication returned no frame")
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._camera.release()


class VisibleWindowCaptureSource:
    """Capture a visible, unobscured window client area from the desktop."""

    def __init__(
        self,
        window: WindowInfo,
        fps: int = 20,
        *,
        burst_fps: int | None = None,
        grabber_factory: GrabberFactory | None = None,
        region_provider: RegionProvider | None = None,
        is_window: Callable[[int], bool] | None = None,
        is_visible: Callable[[int], bool] | None = None,
        is_minimized: Callable[[int], bool] | None = None,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        resolved_burst_fps = fps if burst_fps is None else burst_fps
        if (
            isinstance(resolved_burst_fps, bool)
            or not isinstance(resolved_burst_fps, int)
            or resolved_burst_fps < fps
        ):
            raise ValueError("burst_fps must be an integer at least as large as fps")
        self._window = window
        self._fps = fps
        self._burst_fps = resolved_burst_fps
        self._bursting = False
        self._grabber_factory = grabber_factory or _default_grabber_factory
        self._region_provider = region_provider or _client_region
        self._is_window = is_window or _default_is_window
        self._is_visible = is_visible or _default_is_visible
        self._is_minimized = is_minimized or _default_is_minimized
        self._lock = Lock()
        self._stop = Event()
        self._rate_changed = Event()
        self._started = False
        self._close_called = False
        self._closed_reported = False
        self._thread: Thread | None = None
        self._on_frame: FrameCallback | None = None
        self._on_closed: ClosedCallback | None = None

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def burst_fps(self) -> int:
        return self._burst_fps

    @property
    def bursting(self) -> bool:
        with self._lock:
            return self._bursting

    def set_bursting(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise TypeError("bursting state must be a boolean")
        with self._lock:
            if self._bursting == active:
                return
            self._bursting = active
        self._rate_changed.set()

    def start(
        self,
        on_frame: FrameCallback,
        on_closed_callback: ClosedCallback | None = None,
    ) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("frame source has already started")
            if self._close_called or not self._target_available():
                raise CaptureClosedError("target window is not visible and available")
            self._started = True
            self._on_frame = on_frame
            self._on_closed = on_closed_callback
            thread = Thread(
                target=self._run,
                name="xiangqi-visible-window-capture",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def close(self) -> None:
        with self._lock:
            if self._close_called:
                return
            self._close_called = True
            thread = self._thread
        self._stop.set()
        self._rate_changed.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2.0)

    def _run(self) -> None:
        grabber: _ScreenGrabber | None = None
        last_timestamp = -1
        try:
            grabber = self._grabber_factory()
            while not self._stop.is_set():
                if not self._target_available():
                    raise CaptureClosedError("target window is no longer visible and available")
                region = self._region_provider(self._window.hwnd)
                pixels = np.asarray(grabber.grab(region.as_monitor()))
                timestamp = max(perf_counter_ns(), last_timestamp + 1)
                last_timestamp = timestamp
                with self._lock:
                    callback = None if self._close_called else self._on_frame
                if callback is not None:
                    callback(
                        CaptureFrame(
                            timestamp_ns=timestamp,
                            hwnd=self._window.hwnd,
                            bgra=pixels,
                        )
                    )
                with self._lock:
                    active_fps = self._burst_fps if self._bursting else self._fps
                self._rate_changed.wait(1.0 / active_fps)
                self._rate_changed.clear()
                if self._stop.is_set():
                    break
        except (CaptureClosedError, OSError, RuntimeError, ValueError) as exc:
            self._report_closed(CaptureClosedError(str(exc)))
        finally:
            if grabber is not None:
                grabber.close()

    def _target_available(self) -> bool:
        hwnd = self._window.hwnd
        return self._is_window(hwnd) and self._is_visible(hwnd) and not self._is_minimized(hwnd)

    def _report_closed(self, error: CaptureClosedError) -> None:
        with self._lock:
            if self._close_called or self._closed_reported:
                return
            self._closed_reported = True
            callback = self._on_closed
        if callback is not None:
            callback(error)


def _default_grabber_factory() -> _ScreenGrabber:
    import dxcam

    camera = dxcam.create(
        output_color="BGRA",
        backend="dxgi",
        processor_backend="cv2",
    )
    return DxgiDesktopGrabber(cast(_DxgiCamera, camera))


def _client_region(hwnd: int) -> CaptureRegion:
    user32: Any = ctypes.windll.user32
    rect = wintypes.RECT()
    origin = wintypes.POINT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error(), "GetClientRect failed")
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise OSError(ctypes.get_last_error(), "ClientToScreen failed")
    return CaptureRegion(
        left=int(origin.x),
        top=int(origin.y),
        width=int(rect.right - rect.left),
        height=int(rect.bottom - rect.top),
    )


def _default_is_window(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsWindow(hwnd))


def _default_is_visible(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsWindowVisible(hwnd))


def _default_is_minimized(hwnd: int) -> bool:
    return bool(ctypes.windll.user32.IsIconic(hwnd))
