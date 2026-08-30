from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt

from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.ui.main_window import MainWindow

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


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


def test_main_window_loads_fen_and_replaces_quick_with_deep_result(qtbot: object) -> None:
    engine = ImmediateEngine()
    window = MainWindow(engine=engine, quick_ms=20, deep_ms=100)
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

    window.close()
    assert engine.closed


def test_invalid_fen_never_reaches_engine(qtbot: object) -> None:
    engine = ImmediateEngine()
    window = MainWindow(engine=engine)
    qtbot.addWidget(window)  # type: ignore[attr-defined]
    window.fen_input.setText("not a fen")

    qtbot.mouseClick(window.analyse_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert "FEN 无效" in window.phase_label.text()
    assert engine.calls == []
    window.close()


def test_missing_local_engine_shows_install_guidance_without_crashing(
    qtbot: object, tmp_path: Path
) -> None:
    window = MainWindow(runtime_root=tmp_path)
    qtbot.addWidget(window)  # type: ignore[attr-defined]

    assert not window.analyse_button.isEnabled()
    assert "未安装" in window.phase_label.text()
    assert "install_pikafish.py" in window.guidance_label.text()
    assert window.board_widget.board is not None
    assert window.board_widget.board.fen == START

    window.close()
