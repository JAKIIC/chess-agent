from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.capture.windows_capture_source import WindowsCaptureSource
from xiangqi_agent.platform.windows import WindowInfo


class FakeControl:
    def __init__(self) -> None:
        self.stop_count = 0
        self.wait_count = 0

    def stop(self) -> None:
        self.stop_count += 1

    def wait(self) -> None:
        self.wait_count += 1


class FakeBackend:
    def __init__(self) -> None:
        self.handlers: dict[str, Callable[..., None]] = {}
        self.control = FakeControl()

    def event(self, handler: Callable[..., None]) -> Callable[..., None]:
        self.handlers[handler.__name__] = handler
        return handler

    def start_free_threaded(self) -> FakeControl:
        return self.control

    def emit(self, pixels: np.ndarray) -> None:
        frame = SimpleNamespace(frame_buffer=pixels)
        self.handlers["on_frame_arrived"](frame, SimpleNamespace())

    def close_target(self) -> None:
        self.handlers["on_closed"]()


def test_windows_source_captures_by_hwnd_at_two_fps_and_owns_frame() -> None:
    backend = FakeBackend()
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> FakeBackend:
        factory_calls.append(kwargs)
        return backend

    window = WindowInfo(hwnd=42, title="target", process_name="Weixin.exe", client_size=(30, 20))
    received: list[CaptureFrame] = []
    source = WindowsCaptureSource(window, capture_factory=factory, is_window=lambda _hwnd: True)
    source.start(received.append)
    pixels = np.zeros((20, 30, 4), dtype=np.uint8)

    backend.emit(pixels)
    pixels.fill(255)
    source.close()
    source.close()

    assert factory_calls == [{
        "cursor_capture": False,
        "draw_border": False,
        "minimum_update_interval": 500,
        "window_hwnd": 42,
    }]
    assert received[0].bgra.flags["OWNDATA"]
    assert np.count_nonzero(received[0].bgra) == 0
    assert backend.control.stop_count == 1
    assert backend.control.wait_count == 1


def test_windows_source_rejects_closed_window_and_reports_backend_close() -> None:
    backend = FakeBackend()
    window = WindowInfo(hwnd=42, title="target", process_name="Weixin.exe", client_size=(30, 20))
    dead = WindowsCaptureSource(window, capture_factory=lambda **_kwargs: backend, is_window=lambda _hwnd: False)
    with pytest.raises(CaptureClosedError, match="not available"):
        dead.start(lambda _frame: None)

    errors: list[CaptureClosedError] = []
    alive = WindowsCaptureSource(window, capture_factory=lambda **_kwargs: backend, is_window=lambda _hwnd: True)
    alive.start(lambda _frame: None, errors.append)
    backend.close_target()
    backend.close_target()
    alive.close()

    assert len(errors) == 1
    assert "closed" in str(errors[0]).lower()


def test_windows_source_allows_size_change_for_probe_reporting() -> None:
    backend = FakeBackend()
    window = WindowInfo(hwnd=42, title="target", process_name="Weixin.exe", client_size=(30, 20))
    received: list[CaptureFrame] = []
    source = WindowsCaptureSource(
        window,
        capture_factory=lambda **_kwargs: backend,
        is_window=lambda _hwnd: True,
    )
    source.start(received.append)

    backend.emit(np.zeros((20, 30, 4), dtype=np.uint8))
    backend.emit(np.zeros((25, 40, 4), dtype=np.uint8))
    source.close()

    assert [frame.size for frame in received] == [(30, 20), (40, 25)]
