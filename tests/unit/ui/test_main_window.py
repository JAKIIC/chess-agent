from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureClosedError
from xiangqi_agent.coach.client import DeepSeekClient
from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.engine.service import AnalysisFailure
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.ui.capture_panel import CapturePanel
from xiangqi_agent.ui.main_window import MainWindow

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
TEST_QUAD = "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540"


class ImmediateEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []
        self.closed = False

    def start(self) -> None:
        pass

    def analyse(
        self,
        board: BoardState,
        *,
        movetime_ms: int,
        multipv: int,
        timeout: float | None = None,
    ) -> EngineAnalysis:
        self.calls.append((board.position_id, movetime_ms))
        line = EngineLine(
            position_id=board.position_id,
            depth=10 if movetime_ms < 100 else 16,
            seldepth=20,
            multipv=1,
            score_cp=28,
            mate_in=None,
            nodes=1000,
            nps=10_000,
            time_ms=movetime_ms,
            pv=("h2e2",),
        )
        return EngineAnalysis(
            position_id=board.position_id,
            duration_ms=movetime_ms,
            depth=line.depth,
            nodes=1000,
            lines=(line,),
            bestmove="h2e2",
            engine_name="fake",
        )

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class OneWindowCatalog:
    def __init__(self, window: WindowInfo) -> None:
        self.window = window

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return (self.window,)


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


def test_main_window_loads_fen_and_replaces_quick_with_deep_result(qtbot: object) -> None:
    engine = ImmediateEngine()
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        quick_ms=20,
        deep_ms=100,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.show()

    window.fen_input.setText(START)
    qtbot.mouseClick(window.analyse_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: window.phase_label.text().startswith("加深分析"), timeout=2000
    )

    assert window.board_widget.board is not None
    assert sum(piece != "." for piece in window.board_widget.board.pieces) == 32
    assert window.results.rowCount() == 1
    assert window.results.item(0, 0).text() == "炮二平五"
    assert window.results.item(0, 1).text() == "+0.28"
    assert window.results.item(0, 2).text() == "16"
    assert [call[1] for call in engine.calls] == [20, 100]

    window.coach_panel.question_input.setText("为什么推荐这一步？")
    qtbot.mouseClick(window.coach_panel.ask_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: window.coach_panel.visible_sections() == ("position_summary",), timeout=2000
    )
    assert "本地" in window.coach_panel.source_label.text()
    assert "炮二平五" in window.coach_panel.candidate_label.text()

    window.close()
    assert engine.closed


def test_invalid_fen_never_reaches_engine(qtbot: object) -> None:
    engine = ImmediateEngine()
    window = MainWindow(engine=engine, coach_client=DeepSeekClient(api_key=None))
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.fen_input.setText("not a fen")

    qtbot.mouseClick(window.analyse_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert "FEN 无效" in window.phase_label.text()
    assert engine.calls == []
    window.close()


def test_missing_local_engine_shows_install_guidance_without_crashing(
    qtbot: object, tmp_path: Path
) -> None:
    window = MainWindow(
        runtime_root=tmp_path,
        coach_client=DeepSeekClient(api_key=None),
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    assert window.analyse_button.isEnabled()
    assert "未安装" in window.phase_label.text()
    assert "install_pikafish.py" in window.guidance_label.text()
    assert window.board_widget.board is not None
    assert window.board_widget.board.fen == START

    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    window.fen_input.setText(after.fen)
    qtbot.mouseClick(window.analyse_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    assert window.board_widget.board is not None
    assert window.board_widget.board.position_id == after.position_id
    assert "Pikafish 未安装" in window.phase_label.text()

    window.close()


def test_main_window_closes_connected_capture_panel(qtbot: object) -> None:
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (300, 200))
    source = FakeFrameSource(target.hwnd)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(target),
        source_factory=lambda _: source,
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    capture_panel.refresh_button.click()
    capture_panel.connect_button.click()
    source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=1)

    window.close()

    with pytest.raises(CaptureClosedError, match="closed"):
        source.push(np.zeros((200, 300, 4), dtype=np.uint8), timestamp_ns=2)


def test_confirmed_live_move_updates_board_without_starting_gated_analysis(
    qtbot: object,
) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    source = FakeFrameSource(target.hwnd)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(target),
        source_factory=lambda _: source,
        patch_size=CELL,
    )
    engine = ImmediateEngine()
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
        quick_ms=20,
        deep_ms=100,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    capture_panel.refresh_button.click()
    capture_panel.quad_input.setText(TEST_QUAD)
    capture_panel.connect_button.click()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    moved = _render(after)
    source.push(moved, 150_000_000)
    source.push(moved.copy(), 260_000_000)

    qtbot.waitUntil(lambda: window.board_widget.board == after, timeout=3000)  # type: ignore[attr-defined]

    assert window.fen_input.text() == after.fen
    assert engine.calls == []
    assert window.results.rowCount() == 0
    assert "盲测" in window.phase_label.text()
    window.close()


def test_two_ply_sync_replays_both_notations_and_adopts_only_the_final_board(
    qtbot: object,
) -> None:
    board = parse_fen(START)
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    final = apply_move(middle, second)
    engine = ImmediateEngine()
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.MOVE_ACCEPTED,
            board=final,
            message="atomic double",
            moves=(first, second),
            before_position_id=board.position_id,
            after_position_id=final.position_id,
            sync_mode=SyncMode.HUMAN_VS_AI,
        )
    )

    assert window.board_widget.board == final
    assert window.board_widget.last_move == second
    assert "炮二平五 · 炮8平5" in window.phase_label.text()
    assert engine.calls == []
    window.close()


def test_two_ply_sync_submits_only_the_final_position_to_live_analysis(
    qtbot: object,
) -> None:
    board = parse_fen(START)
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    final = apply_move(middle, second)
    engine = ImmediateEngine()
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
        quick_ms=20,
        deep_ms=100,
        enable_live_analysis=True,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.MOVE_ACCEPTED,
            board=final,
            message="atomic double",
            moves=(first, second),
            before_position_id=board.position_id,
            after_position_id=final.position_id,
            sync_mode=SyncMode.HUMAN_VS_AI,
        )
    )
    qtbot.waitUntil(lambda: len(engine.calls) == 2, timeout=2000)  # type: ignore[attr-defined]

    assert {position_id for position_id, _time_ms in engine.calls} == {
        final.position_id
    }
    assert middle.position_id not in {position_id for position_id, _time_ms in engine.calls}
    window.close()


def test_main_window_rejects_two_ply_event_from_strict_mode(qtbot: object) -> None:
    board = parse_fen(START)
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    final = apply_move(middle, second)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.MOVE_ACCEPTED,
            board=final,
            message="unsafe strict double",
            moves=(first, second),
            before_position_id=board.position_id,
            after_position_id=final.position_id,
            sync_mode=SyncMode.STRICT_SINGLE,
        )
    )

    assert window.board_widget.board == board
    window.close()


def test_main_window_rejects_move_whose_before_position_is_stale(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    engine = ImmediateEngine()
    capture_panel = CapturePanel(catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1))))
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.MOVE_ACCEPTED,
            board=after,
            message="stale move",
            moves=(move,),
            before_position_id="not-the-current-position",
        )
    )

    assert window.board_widget.board == board
    assert window.fen_input.text() == board.fen
    assert engine.calls == []
    window.close()


def test_manual_fen_waits_for_stable_capture_recovery_before_adoption(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    source = FakeFrameSource(target.hwnd)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(target),
        source_factory=lambda _: source,
        patch_size=CELL,
    )
    engine = ImmediateEngine()
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
        quick_ms=20,
        deep_ms=100,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    capture_panel.refresh_button.click()
    capture_panel.quad_input.setText(TEST_QUAD)
    capture_panel.connect_button.click()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    qtbot.waitUntil(lambda: "实时同步已启动" in capture_panel.status_label.text(), timeout=2000)  # type: ignore[attr-defined]

    window.fen_input.setText(after.fen)
    qtbot.mouseClick(window.analyse_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert window.board_widget.board == board
    assert engine.calls == []
    assert "稳定画面" in window.phase_label.text()

    moved = _render(after)
    source.push(moved, 150_000_000)
    source.push(moved.copy(), 210_000_000)
    source.push(moved.copy(), 270_000_000)
    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: window.board_widget.board is not None
        and window.board_widget.board.position_id == after.position_id
        and len(engine.calls) == 2,
        timeout=3000,
    )
    window.close()


def test_black_bottom_baseline_updates_mirror_orientation_without_analysis(qtbot: object) -> None:
    board = parse_fen(START)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    source = FakeFrameSource(target.hwnd)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(target),
        source_factory=lambda _: source,
        patch_size=CELL,
    )
    engine = ImmediateEngine()
    window = MainWindow(
        engine=engine,
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    capture_panel.refresh_button.click()
    capture_panel.quad_input.setText(TEST_QUAD)
    capture_panel.orientation_combo.setCurrentIndex(1)
    capture_panel.connect_button.click()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)

    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: window.board_widget.board is not None
        and window.board_widget.board.orientation is Orientation.BLACK_BOTTOM,
        timeout=2000,
    )
    assert engine.calls == []
    window.close()


def test_main_window_ignores_analysis_failure_for_an_old_position(qtbot: object) -> None:
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    assert window.board_widget.board is not None
    window._adopt_board(window.board_widget.board, "current analysis")
    assert window._analysis_generation is not None
    generation = window._analysis_generation
    before = window.phase_label.text()

    window._show_error(
        AnalysisFailure(window.board_widget.board.position_id, generation - 1, "old failure")
    )

    assert window.phase_label.text() == before
    window._show_error(
        AnalysisFailure(window.board_widget.board.position_id, generation, "current failure")
    )
    assert window.phase_label.text() == "分析暂停：current failure"
    window.close()


def test_capture_failure_cancels_pending_manual_recovery_without_adopting(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    pending = apply_move(board, move)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window._pending_manual_board = pending

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.ERROR,
            board=board,
            message="capture failed",
        )
    )

    assert window._pending_manual_board is None
    assert window.board_widget.board == board
    assert "原局面" in window.phase_label.text()
    window.close()


def test_stale_recovery_acceptance_cannot_satisfy_a_newer_request(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    pending = apply_move(board, move)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window._pending_manual_board = pending
    window._pending_recovery_id = 2

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.RECOVERY_ACCEPTED,
            board=pending,
            message="old recovery",
            recovery_id=1,
        )
    )

    assert window.board_widget.board == board
    assert window._pending_manual_board == pending
    assert window._pending_recovery_id == 2
    window.close()


def test_live_move_is_ignored_while_manual_recovery_is_pending(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(WindowInfo(42, "天天象棋", "x", (1, 1)))
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window._pending_manual_board = after
    window._pending_recovery_id = 1

    capture_panel.sync_update.emit(
        LiveSyncUpdate(
            status=LiveSyncStatus.MOVE_ACCEPTED,
            board=after,
            message="old move",
            moves=(move,),
            before_position_id=board.position_id,
        )
    )

    assert window.board_widget.board == board
    assert window._pending_manual_board == after
    window.close()


def test_disconnect_cancels_pending_recovery_and_reconnect_accepts_moves(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    target = WindowInfo(42, "天天象棋", "WeChatAppEx.exe", (216, 240))
    first_source = FakeFrameSource(target.hwnd)
    second_source = FakeFrameSource(target.hwnd)
    sources = iter((first_source, second_source))
    capture_panel = CapturePanel(
        catalog=OneWindowCatalog(target),
        source_factory=lambda _: next(sources),
        patch_size=CELL,
    )
    window = MainWindow(
        engine=ImmediateEngine(),
        coach_client=DeepSeekClient(api_key=None),
        capture_panel=capture_panel,
    )
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    capture_panel.refresh_button.click()
    capture_panel.quad_input.setText(TEST_QUAD)
    capture_panel.connect_button.click()
    baseline = _render(board)
    first_source.push(baseline, 0)
    first_source.push(baseline.copy(), 50_000_000)
    first_source.push(baseline.copy(), 100_000_000)
    qtbot.waitUntil(lambda: "实时同步已启动" in capture_panel.status_label.text(), timeout=2000)  # type: ignore[attr-defined]

    window.fen_input.setText(after.fen)
    window.analyse_button.click()
    assert window._pending_manual_board is not None
    capture_panel.connect_button.click()

    assert window._pending_manual_board is None
    assert window._pending_recovery_id is None

    capture_panel.connect_button.click()
    second_source.push(baseline, 0)
    second_source.push(baseline.copy(), 50_000_000)
    second_source.push(baseline.copy(), 100_000_000)
    qtbot.waitUntil(lambda: "实时同步已启动" in capture_panel.status_label.text(), timeout=2000)  # type: ignore[attr-defined]
    moved = _render(after)
    second_source.push(moved, 150_000_000)
    second_source.push(moved.copy(), 260_000_000)
    qtbot.waitUntil(lambda: window.board_widget.board == after, timeout=2000)  # type: ignore[attr-defined]
    window.close()
