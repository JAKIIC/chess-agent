import pytest

from xiangqi_agent.domain.board import Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _move(board, uci: str) -> Move:
    return next(move for move in legal_moves(board) if move.uci == uci)


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


def test_rule_committer_projects_two_legal_plies_without_mutating_input() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    after_first = RuleStateCommitter().commit(board, first)
    second = _move(after_first, "h7e7")

    projected = RuleStateCommitter().project(board, (first, second))

    assert projected == RuleStateCommitter().commit(after_first, second)
    assert board == parse_fen(START)


def test_rule_committer_rejects_whole_sequence_when_second_move_is_illegal() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    illegal_second = Move("a0a9", 81, 0)

    with pytest.raises(ValueError, match="legal move"):
        RuleStateCommitter().commit_sequence(board, (first, illegal_second))

    assert board == parse_fen(START)


def test_rule_committer_projects_every_two_ply_chain_through_one_rule_boundary() -> None:
    board = parse_fen(START)
    expected_first = _move(board, "h2e2")
    middle = RuleStateCommitter().commit(board, expected_first)
    expected_second = _move(middle, "h7e7")
    expected_final = RuleStateCommitter().commit(middle, expected_second)

    projections = tuple(RuleStateCommitter().project_two_ply(board))
    matching = [
        final
        for moves, final in projections
        if moves == (expected_first, expected_second)
    ]

    assert matching == [expected_final]
    assert all(len(moves) == 2 for moves, _final in projections)


def test_rule_committer_projects_only_replies_after_a_confirmed_first_move() -> None:
    board = parse_fen(START)
    expected_first = _move(board, "h2e2")
    middle = RuleStateCommitter().commit(board, expected_first)
    expected_second = _move(middle, "h7e7")
    expected_final = RuleStateCommitter().commit(middle, expected_second)

    projections = tuple(
        RuleStateCommitter().project_replies(board, expected_first)
    )

    assert projections
    assert all(moves[0] == expected_first for moves, _final in projections)
    assert [
        final
        for moves, final in projections
        if moves == (expected_first, expected_second)
    ] == [expected_final]
