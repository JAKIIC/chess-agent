from __future__ import annotations

from dataclasses import replace

import pytest

from xiangqi_agent.coach.evidence import build_evidence
from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _analysis() -> tuple[BoardState, EngineAnalysis]:
    board = parse_fen(START)
    moves = (("h2e2", "h9g7"), ("b0c2", "b9c7"), ("g3g4", "g6g5"))
    lines = tuple(
        EngineLine(
            position_id=board.position_id,
            depth=18 - index,
            seldepth=24,
            multipv=index + 1,
            score_cp=35 - index * 8,
            mate_in=None,
            nodes=20_000,
            nps=200_000,
            time_ms=500,
            pv=pv,
        )
        for index, pv in enumerate(moves)
    )
    return board, EngineAnalysis(
        position_id=board.position_id,
        duration_ms=500,
        depth=18,
        nodes=20_000,
        lines=lines,
        bestmove="h2e2",
        engine_name="Pikafish test",
    )


def test_evidence_contains_only_rule_and_engine_verified_text() -> None:
    board, analysis = _analysis()

    evidence = build_evidence(board, analysis, user_side="w", recent_moves=("a3a4",))
    serialized = evidence.model_dump_json()

    assert evidence.position_id == board.position_id
    assert evidence.fen == board.fen
    assert evidence.phase == "opening"
    assert evidence.recent_moves == ("a3a4",)
    assert evidence.allowed_move_map == {
        "candidate_1": "炮二平五",
        "candidate_2": "马八进七",
        "candidate_3": "兵三进一",
    }
    assert [candidate.uci for candidate in evidence.candidates] == ["h2e2", "b0c2", "g3g4"]
    assert evidence.candidates[0].pv_notation == ("炮二平五", "马8进7")
    assert evidence.material_facts["red_pawn"] == 5
    assert evidence.king_safety_facts == {"side_to_move_in_check": False}
    assert all(not tactic.is_capture for tactic in evidence.immediate_tactics)
    assert "screenshot" not in serialized.lower()
    assert "api_key" not in serialized.lower()


def test_evidence_rejects_stale_analysis() -> None:
    _board, analysis = _analysis()
    stale_board = parse_fen(START.replace(" w", " b"))

    with pytest.raises(ValueError, match="position"):
        build_evidence(stale_board, analysis, user_side="w")


def test_evidence_rejects_an_engine_candidate_outside_legal_moves() -> None:
    board, analysis = _analysis()
    bad_line = replace(analysis.lines[0], pv=("a0a9",))
    bad_analysis = replace(analysis, lines=(bad_line, *analysis.lines[1:]))

    with pytest.raises(ValueError, match="legal"):
        build_evidence(board, bad_analysis, user_side="w")


def test_evidence_rejects_unvalidated_recent_move_text() -> None:
    board, analysis = _analysis()

    with pytest.raises(ValueError, match="recent"):
        build_evidence(board, analysis, user_side="w", recent_moves=("炮二平五",))
