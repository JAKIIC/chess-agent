from __future__ import annotations

from threading import Thread

import numpy as np
from PySide6.QtCore import Qt

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.visible_window_source import VisibleWindowCaptureSource
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.sync.live_session import LiveSyncStatus
from xiangqi_agent.ui.capture_panel import CapturePanel, _default_source_factory

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
TEST_QUAD = "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540"


class FakeCatalog:
    def __init__(self, windows: tuple[WindowInfo, ...]) -> None:
        self.windows = windows

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return self.windows


def test_default_capture_backend_reads_the_visible_wechat_window() -> None:
    window = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))

    source = _default_source_factory(window)

    assert isinstance(source, VisibleWindowCaptureSource)
    assert source.fps == 2
    assert source.burst_fps == 20


def _render(board: BoardState) -> np.ndarray:
    frame = np.zeros((240, 216, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = PALETTE[symbol]
    return frame


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


def test_capture_panel_emits_rule_confirmed_board_updates(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    source = FakeFrameSource(target.hwnd)
    panel = CapturePanel(
        catalog=FakeCatalog((target,)),
        source_factory=lambda _: source,
        board_provider=lambda: board,
        patch_size=CELL,
    )
    updates = []
    panel.sync_update.connect(updates.append)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.refresh_button.click()
    panel.quad_input.setText(TEST_QUAD)
    panel.connect_button.click()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates),
        timeout=2000,
    )

    moved = _render(after)
    source.push(moved, 150_000_000)
    source.push(moved.copy(), 260_000_000)
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates),
        timeout=2000,
    )

    accepted = next(update for update in updates if update.status is LiveSyncStatus.MOVE_ACCEPTED)
    assert accepted.board == after
    assert "已同步" in panel.status_label.text()
    panel.close_capture()


def test_capture_panel_ignores_queued_updates_from_a_closed_generation(qtbot: object) -> None:
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))
    old_source = FakeFrameSource(target.hwnd)
    current_source = FakeFrameSource(target.hwnd)
    sources = iter((old_source, current_source))
    panel = CapturePanel(
        catalog=FakeCatalog((target,)),
        source_factory=lambda _: next(sources),
    )
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.refresh_button.click()
    panel.connect_button.click()
    old_close = Thread(target=old_source.simulate_target_close)
    old_close.start()
    old_close.join()

    panel.close_capture()
    panel.connect_button.click()
    qtbot.wait(100)  # type: ignore[attr-defined]

    assert panel.connect_button.text() == "断开"
    current_source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)
    qtbot.waitUntil(lambda: "90" in panel.status_label.text(), timeout=1000)  # type: ignore[attr-defined]
    panel.close_capture()
