from __future__ import annotations

import sys
from importlib import import_module
from threading import Event
from types import ModuleType

import numpy as np
import pytest

from xiangqi_agent.capture.protocol import CaptureFrame
from xiangqi_agent.platform.windows import WindowInfo


class FakeGrabber:
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        self._frames = frames
        self._index = 0
        self.regions: list[dict[str, int]] = []
        self.closed = False

    def grab(self, region: dict[str, int]) -> np.ndarray:
        self.regions.append(region)
        frame = self._frames[min(self._index, len(self._frames) - 1)]
        self._index += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeDxgiCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.calls: list[tuple[tuple[int, int, int, int], bool, bool]] = []
        self.released = False

    def grab(
        self,
        region: tuple[int, int, int, int],
        *,
        copy: bool,
        new_frame_only: bool,
    ) -> np.ndarray:
        self.calls.append((region, copy, new_frame_only))
        return self.frame

    def release(self) -> None:
        self.released = True


def _visible_capture_module() -> ModuleType:
    try:
        return import_module("xiangqi_agent.capture.visible_window_source")
    except ModuleNotFoundError:
        pytest.fail("visible-window capture source is not implemented")


def test_visible_source_emits_updated_owned_bgra_frames_from_client_region() -> None:
    capture = _visible_capture_module()
    region = capture.CaptureRegion(left=11, top=22, width=30, height=20)
    first = np.zeros((20, 30, 4), dtype=np.uint8)
    second = np.full((20, 30, 4), 255, dtype=np.uint8)
    grabber = FakeGrabber((first, second))
    received: list[CaptureFrame] = []
    ready = Event()

    def receive(frame: CaptureFrame) -> None:
        received.append(frame)
        if len(received) == 2:
            ready.set()

    window = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (30, 20))
    source = capture.VisibleWindowCaptureSource(
        window,
        fps=200,
        grabber_factory=lambda: grabber,
        region_provider=lambda _hwnd: region,
        is_window=lambda _hwnd: True,
        is_visible=lambda _hwnd: True,
        is_minimized=lambda _hwnd: False,
    )

    source.start(receive)
    assert ready.wait(1.0)
    source.close()

    assert grabber.regions[:2] == [
        {"left": 11, "top": 22, "width": 30, "height": 20},
        {"left": 11, "top": 22, "width": 30, "height": 20},
    ]
    assert [frame.size for frame in received[:2]] == [(30, 20), (30, 20)]
    assert received[0].timestamp_ns < received[1].timestamp_ns
    assert np.count_nonzero(received[0].bgra) == 0
    assert np.count_nonzero(received[1].bgra) == 20 * 30 * 4
    assert received[0].bgra.flags["OWNDATA"]
    assert received[1].bgra.flags["OWNDATA"]
    assert grabber.closed


def test_visible_source_wakes_and_increases_capture_rate_during_a_burst() -> None:
    capture = _visible_capture_module()
    region = capture.CaptureRegion(left=0, top=0, width=4, height=4)
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    grabber = FakeGrabber((pixels,))
    received: list[CaptureFrame] = []
    first_frame = Event()
    burst_frames = Event()

    def receive(frame: CaptureFrame) -> None:
        received.append(frame)
        first_frame.set()
        if len(received) >= 3:
            burst_frames.set()

    source = capture.VisibleWindowCaptureSource(
        WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (4, 4)),
        fps=1,
        burst_fps=100,
        grabber_factory=lambda: grabber,
        region_provider=lambda _hwnd: region,
        is_window=lambda _hwnd: True,
        is_visible=lambda _hwnd: True,
        is_minimized=lambda _hwnd: False,
    )

    source.start(receive)
    assert first_frame.wait(1.0)
    source.set_bursting(True)
    try:
        assert burst_frames.wait(0.5)
    finally:
        source.close()

    assert source.bursting
    assert source.fps == 1
    assert source.burst_fps == 100


@pytest.mark.parametrize("burst_fps", [0, 1, True, 2.5])
def test_visible_source_rejects_an_invalid_burst_rate(burst_fps: object) -> None:
    with pytest.raises(ValueError, match="burst_fps"):
        _visible_capture_module().VisibleWindowCaptureSource(
            WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (4, 4)),
            fps=2,
            burst_fps=burst_fps,
        )


def test_visible_source_requires_a_boolean_burst_state() -> None:
    source = _visible_capture_module().VisibleWindowCaptureSource(
        WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (4, 4)),
    )

    with pytest.raises(TypeError, match="boolean"):
        source.set_bursting(1)


def test_dxgi_grabber_crops_the_visible_desktop_and_releases_camera() -> None:
    capture = _visible_capture_module()
    frame = np.full((40, 30, 4), 127, dtype=np.uint8)
    camera = FakeDxgiCamera(frame)
    grabber = capture.DxgiDesktopGrabber(camera)

    pixels = grabber.grab({"left": 11, "top": 22, "width": 30, "height": 40})
    grabber.close()

    assert pixels is frame
    assert camera.calls == [((11, 22, 41, 62), True, False)]
    assert camera.released


def test_default_grabber_selects_bgra_dxgi_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = _visible_capture_module()
    camera = FakeDxgiCamera(np.zeros((1, 1, 4), dtype=np.uint8))
    create_calls: list[dict[str, str]] = []
    fake_dxcam = ModuleType("dxcam")

    def create(**kwargs: str) -> FakeDxgiCamera:
        create_calls.append(kwargs)
        return camera

    fake_dxcam.create = create  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dxcam", fake_dxcam)

    grabber = capture._default_grabber_factory()
    try:
        assert create_calls == [
            {
                "output_color": "BGRA",
                "backend": "dxgi",
                "processor_backend": "cv2",
            }
        ]
    finally:
        grabber.close()

    assert camera.released
