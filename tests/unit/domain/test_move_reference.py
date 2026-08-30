from __future__ import annotations

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import resolve_move_reference

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_resolve_exact_chinese_move_from_legal_moves() -> None:
    move = resolve_move_reference(parse_fen(START), "我想走炮二平五，可以吗？")

    assert move is not None
    assert move.uci == "h2e2"


def test_resolve_move_reference_returns_none_for_unknown_or_multiple_mentions() -> None:
    board = parse_fen(START)

    assert resolve_move_reference(board, "我想稳一点") is None
    assert resolve_move_reference(board, "炮二平五和马八进七哪个好？") is None
