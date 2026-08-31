from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.ui.capture_panel import CapturePanel


class FakeCatalog:
    def __init__(self, windows: tuple[WindowInfo, ...]) -> None:
        self.windows = windows

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return self.windows


def test_capture_panel_selects_window_and_reports_ninety_points(qtbot: object) -> None:
    window = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))
    source = FakeFrameSource(hwnd=42)
    panel = CapturePanel(catalog=FakeCatalog((window,)), source_factory=lambda _: source)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    qtbot.mouseClick(panel.refresh_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    qtbot.mouseClick(panel.connect_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)

    qtbot.waitUntil(lambda: "90" in panel.status_label.text(), timeout=1000)  # type: ignore[attr-defined]
    assert panel.window_combo.count() == 1
    assert "天天象棋" in panel.window_combo.itemText(0)
    assert "自动同步未启用" in panel.status_label.text()
    assert panel.connect_button.text() == "断开"

    source.push(np.zeros((300, 450, 4), dtype=np.uint8), timestamp_ns=2)
    qtbot.waitUntil(lambda: "几何已自动适配" in panel.status_label.text(), timeout=1000)  # type: ignore[attr-defined]
    assert "同步尚未确认" in panel.status_label.text()

    source.push(np.zeros((200, 450, 4), dtype=np.uint8), timestamp_ns=3)
    qtbot.waitUntil(lambda: "重新标定" in panel.status_label.text(), timeout=1000)  # type: ignore[attr-defined]
    panel.close_capture()
    panel.close_capture()


def test_capture_panel_handles_no_window_and_invalid_quad_without_starting(
    qtbot: object,
) -> None:
    calls: list[WindowInfo] = []
    panel = CapturePanel(
        catalog=FakeCatalog(()),
        source_factory=lambda window: calls.append(window) or FakeFrameSource(window.hwnd),
    )
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.refresh_button.click()
    assert "没有" in panel.status_label.text()
    assert not panel.connect_button.isEnabled()

    window = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))
    panel._catalog = FakeCatalog((window,))
    panel.refresh_button.click()
    panel.quad_input.setText("bad quad")
    panel.connect_button.click()

    assert "四角" in panel.status_label.text()
    assert calls == []


def test_capture_panel_prefers_the_explicit_tiantian_xiangqi_title(qtbot: object) -> None:
    generic = WindowInfo(41, "微信", "WeChat.exe", (800, 600))
    board = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))
    panel = CapturePanel(catalog=FakeCatalog((generic, board)))
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.refresh_button.click()

    assert "天天象棋" in panel.window_combo.currentText()
