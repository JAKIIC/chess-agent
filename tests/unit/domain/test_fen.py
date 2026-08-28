from dataclasses import FrozenInstanceError

import pytest

from xiangqi_agent.domain.board import BoardState, Move, Orientation
from xiangqi_agent.domain.fen import parse_fen, to_fen

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def test_fen_round_trip_and_position_id_is_stable_across_extended_fields() -> None:
    board = parse_fen(START)

    assert len(board.pieces) == 90
    assert board.side_to_move == "w"
    assert to_fen(board) == START
    assert board.fen == START
    assert board.position_id == parse_fen(START + " - - 0 1").position_id
    assert len(board.position_id) == 32


def test_parse_fen_normalizes_extended_fields_to_board_and_side() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 b - - 17 42")

    assert board.fen == "4k4/9/9/9/9/9/9/9/9/4K4 b"
    assert board.pieces[4] == "k"
    assert board.pieces[85] == "K"
    assert board.orientation is Orientation.RED_BOTTOM
    assert board.ply == 0


def test_board_state_and_move_are_immutable() -> None:
    board = parse_fen(START)
    move = Move(uci="a0a1", from_index=81, to_index=72)

    with pytest.raises(FrozenInstanceError):
        board.side_to_move = "b"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        move.uci = "a0a2"  # type: ignore[misc]


def test_board_state_requires_exactly_ninety_valid_intersections_and_side() -> None:
    with pytest.raises(ValueError):
        BoardState(pieces=(".",) * 89, side_to_move="w")
    with pytest.raises(ValueError):
        BoardState(pieces=(".",) * 89 + ("x",), side_to_move="w")
    with pytest.raises(ValueError):
        BoardState(pieces=(".",) * 90, side_to_move="x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    (
        "9/9/9/9/9/9/9/9/9 w",
        "9/9/9/9/9/9/9/9/9/9/9 w",
        "8/9/9/9/9/9/9/9/9/9 w",
        "10/9/9/9/9/9/9/9/9/9 w",
        "x8/9/9/9/9/9/9/9/9/9 w",
        "9/9/9/9/9/9/9/9/9/9",
        "9/9/9/9/9/9/9/9/9/9 x",
        "9/9/9/9/9/9/9/9/9/9 w - - 0",
        "9/9/9/9/9/9/9/9/9/9 w - - 0 1 extra",
    ),
)
def test_parse_fen_rejects_malformed_input(text: str) -> None:
    with pytest.raises(ValueError):
        parse_fen(text)
