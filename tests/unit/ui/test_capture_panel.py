from __future__ import annotations

from pathlib import Path
from threading import Thread

import numpy as np
from PySide6.QtCore import Qt

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.visible_window_source import VisibleWindowCaptureSource
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.ui.capture_panel import CapturePanel, _default_source_factory
from xiangqi_agent.vision.occupancy import CircularOccupancyObserver

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
TEST_QUAD = "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540"


class FakeCatalog:
    def __init__(self, windows: tuple[WindowInfo, ...]) -> None:
        self.windows = windows

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return self.windows


class RecordingEvidenceWriter:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.calls: list[tuple[LiveSyncUpdate, dict[str, object]]] = []

    def record(self, update: LiveSyncUpdate, **context: object) -> Path:
        self.calls.append((update, context))
        return self.output


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


def test_capture_panel_requires_explicit_mode_and_locks_it_while_connected(
    qtbot: object,
) -> None:
    board = parse_fen(START)
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

    assert panel.mode_combo.currentData() == SyncMode.STRICT_SINGLE.value
    human_index = panel.mode_combo.findData(SyncMode.HUMAN_VS_AI.value)
    panel.mode_combo.setCurrentIndex(human_index)
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

    ready = next(update for update in updates if update.status is LiveSyncStatus.BASELINE_READY)
    assert ready.sync_mode is SyncMode.HUMAN_VS_AI
    assert not panel.mode_combo.isEnabled()

    panel.close_capture()
    assert panel.mode_combo.isEnabled()


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


def test_local_evidence_mode_is_off_by_default_and_requires_human_ai_mode(
    qtbot: object,
) -> None:
    board = parse_fen(START)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    sources = iter((FakeFrameSource(target.hwnd), FakeFrameSource(target.hwnd)))
    panel = CapturePanel(
        catalog=FakeCatalog((target,)),
        source_factory=lambda _: next(sources),
        board_provider=lambda: board,
        patch_size=CELL,
    )
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    assert not panel.evidence_checkbox.isChecked()
    panel.refresh_button.click()
    panel.evidence_checkbox.setChecked(True)
    panel.connect_button.click()

    assert panel._session is None
    assert "人机练习" in panel.status_label.text()

    human_index = panel.mode_combo.findData(SyncMode.HUMAN_VS_AI.value)
    panel.mode_combo.setCurrentIndex(human_index)
    panel.connect_button.click()

    assert panel._session is not None
    assert panel._session._capture_transition_evidence is True
    assert panel._session._require_matching_baseline is True
    assert isinstance(panel._session._occupancy_observer, CircularOccupancyObserver)
    assert not panel.evidence_checkbox.isEnabled()

    panel.close_capture()
    assert panel.evidence_checkbox.isEnabled()


def test_pending_review_pauses_additional_event_recording_until_completion(
    qtbot: object,
    tmp_path: Path,
) -> None:
    board = parse_fen(START)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    writer = RecordingEvidenceWriter(
        tmp_path / ".local" / "stage-c-quarantine" / "session-ui" / "event-ui"
    )
    panel = CapturePanel(
        catalog=FakeCatalog((target,)),
        source_factory=lambda _: FakeFrameSource(target.hwnd),
        board_provider=lambda: board,
        evidence_writer=writer,
        session_id_factory=lambda: "session-ui",
        event_id_factory=lambda: "event-ui",
    )
    ready: list[Path] = []
    panel.review_event_ready.connect(ready.append)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.refresh_button.click()
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData(SyncMode.HUMAN_VS_AI.value))
    panel.evidence_checkbox.setChecked(True)
    panel.connect_button.click()
    update = LiveSyncUpdate(
        status=LiveSyncStatus.PAUSED_AMBIGUOUS,
        board=board,
        message="paused",
        sync_mode=SyncMode.HUMAN_VS_AI,
        frame_size=(216, 240),
        point_count=90,
    )

    panel._show_live_update(update)
    panel._show_live_update(update)

    assert len(writer.calls) == 1
    assert ready == [writer.output]
    assert writer.calls[0][1]["session_id"] == "session-ui"
    assert writer.calls[0][1]["event_id"] == "event-ui"

    panel.finish_review("event-ui", "discarded")
    panel._show_live_update(update)

    assert len(writer.calls) == 2
    panel.close_capture()
    panel.close_capture()
    panel.mode_combo.setCurrentIndex(
        panel.mode_combo.findData(SyncMode.STRICT_SINGLE.value)
    )
    panel._show_live_update(update)

    assert len(writer.calls) == 2
