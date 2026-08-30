import pytest

from xiangqi_agent.domain.board import Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_rule_committer_advances_only_a_legal_move() -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")

    after = RuleStateCommitter().commit(board, move)

    assert after.position_id != board.position_id
    assert after.side_to_move == "b"
    assert after.ply == board.ply + 1


def test_rule_committer_rejects_an_illegal_move_without_mutating_board() -> None:
    board = parse_fen(START)

    with pytest.raises(ValueError, match="legal move"):
        RuleStateCommitter().commit(board, Move("a0a9", 81, 0))

    assert board.position_id == parse_fen(START).position_id
