from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureClosedError
from xiangqi_agent.coach.client import DeepSeekClient
from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo
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

    assert not window.analyse_button.isEnabled()
    assert "未安装" in window.phase_label.text()
    assert "install_pikafish.py" in window.guidance_label.text()
    assert window.board_widget.board is not None
    assert window.board_widget.board.fen == START

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


def test_confirmed_live_move_updates_board_and_starts_analysis(qtbot: object) -> None:
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

    qtbot.waitUntil(  # type: ignore[attr-defined]
        lambda: window.board_widget.board == after
        and len([call for call in engine.calls if call[0] == after.position_id]) == 2
        and window.phase_label.text().startswith("加深分析"),
        timeout=3000,
    )

    assert window.fen_input.text() == after.fen
    assert window.results.rowCount() == 1
    assert window.phase_label.text().startswith("加深分析")
    window.close()
