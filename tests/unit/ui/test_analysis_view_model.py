from __future__ import annotations

from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.ui.analysis_view_model import analysis_rows

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_analysis_rows_render_chinese_notation_red_score_and_variation() -> None:
    board = parse_fen(START)
    line = EngineLine(
        position_id=board.position_id,
        depth=18,
        seldepth=24,
        multipv=1,
        score_cp=35,
        mate_in=None,
        nodes=10_000,
        nps=100_000,
        time_ms=500,
        pv=("h2e2", "h9g7"),
    )
    analysis = EngineAnalysis(
        position_id=board.position_id,
        duration_ms=500,
        depth=18,
        nodes=10_000,
        lines=(line,),
        bestmove="h2e2",
        engine_name="Pikafish test",
    )

    rows = analysis_rows(board, analysis)

    assert rows[0].rank == 1
    assert rows[0].uci == "h2e2"
    assert rows[0].notation == "炮二平五"
    assert rows[0].score == "+0.35"
    assert rows[0].depth == 18
    assert rows[0].variation.startswith("炮二平五")


def test_analysis_rows_render_mate_and_reject_a_different_position() -> None:
    board = parse_fen(START)
    wrong = parse_fen(START.replace(" w", " b"))
    line = EngineLine(
        position_id=wrong.position_id,
        depth=12,
        seldepth=None,
        multipv=1,
        score_cp=None,
        mate_in=-4,
        nodes=None,
        nps=None,
        time_ms=None,
        pv=("b7b0",),
    )
    analysis = EngineAnalysis(
        position_id=wrong.position_id,
        duration_ms=50,
        depth=12,
        nodes=0,
        lines=(line,),
        bestmove="b7b0",
        engine_name="test",
    )

    try:
        analysis_rows(board, analysis)
    except ValueError as exc:
        assert "position" in str(exc)
    else:
        raise AssertionError("stale analysis must be rejected")
