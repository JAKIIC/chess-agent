from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from time import perf_counter_ns
from typing import Any, Protocol, cast

from xiangqi_agent.capture.protocol import (
    CaptureClosedError,
    CaptureFrame,
    ClosedCallback,
    FrameCallback,
)
from xiangqi_agent.platform.windows import WindowInfo


class _CaptureControl(Protocol):
    def stop(self) -> None: ...

    def wait(self) -> None: ...


class _CaptureBackend(Protocol):
    def event(self, handler: Callable[..., None]) -> Callable[..., None]: ...

    def start_free_threaded(self) -> _CaptureControl: ...


type CaptureFactory = Callable[..., _CaptureBackend]


def _default_factory(**kwargs: Any) -> _CaptureBackend:
    from windows_capture import WindowsCapture

    return cast(_CaptureBackend, WindowsCapture(**kwargs))


def _default_is_window(hwnd: int) -> bool:
    import ctypes

    return bool(ctypes.windll.user32.IsWindow(hwnd))


class WindowsCaptureSource:
    def __init__(
        self,
        window: WindowInfo,
        fps: int = 2,
        *,
        capture_factory: CaptureFactory = _default_factory,
        is_window: Callable[[int], bool] = _default_is_window,
    ) -> None:
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("fps must be a positive integer")
        self._window = window
        self._fps = fps
        self._capture_factory = capture_factory
        self._is_window = is_window
        self._lock = Lock()
        self._started = False
        self._close_called = False
        self._backend_closed = False
        self._control: _CaptureControl | None = None
        self._on_frame: FrameCallback | None = None
        self._on_closed: ClosedCallback | None = None

    def start(
        self,
        on_frame: FrameCallback,
        on_closed_callback: ClosedCallback | None = None,
    ) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("frame source has already started")
            if self._close_called or not self._is_window(self._window.hwnd):
                raise CaptureClosedError("target window is not available")
            self._started = True
            self._on_frame = on_frame
            self._on_closed = on_closed_callback

        backend = self._capture_factory(
            cursor_capture=False,
            draw_border=False,
            minimum_update_interval=max(1, round(1000 / self._fps)),
            window_hwnd=self._window.hwnd,
        )

        def on_frame_arrived(frame: Any, _control: Any) -> None:
            with self._lock:
                callback = None if self._close_called or self._backend_closed else self._on_frame
            if callback is not None:
                callback(
                    CaptureFrame(
                        timestamp_ns=perf_counter_ns(),
                        hwnd=self._window.hwnd,
                        bgra=frame.frame_buffer,
                    )
                )

        def on_closed() -> None:
            with self._lock:
                if self._close_called or self._backend_closed:
                    return
                self._backend_closed = True
                callback = self._on_closed
            if callback is not None:
                callback(CaptureClosedError("target window or capture session closed"))

        backend.event(on_frame_arrived)
        backend.event(on_closed)
        control = backend.start_free_threaded()
        with self._lock:
            self._control = control

    def close(self) -> None:
        with self._lock:
            if self._close_called:
                return
            self._close_called = True
            control = self._control
        if control is not None:
            control.stop()
            control.wait()
