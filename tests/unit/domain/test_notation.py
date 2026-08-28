import pytest

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import legal_moves


def notation(board_fen: str, uci: str) -> str:
    board = parse_fen(board_fen)
    move = next(move for move in legal_moves(board) if move.uci == uci)
    return to_chinese(board, move)


def test_red_and_black_notation_use_side_specific_names_and_numerals() -> None:
    assert notation("4k4/9/9/9/9/4P4/9/9/9/4K3R w", "i0i1") == "车一进一"
    assert notation("r3k4/9/9/9/9/4p4/9/9/9/4K4 b", "a9a8") == "车1进1"


def test_notation_uses_destination_file_for_horse_advisor_and_elephant() -> None:
    assert notation("4k4/9/9/9/9/4P4/9/9/4N4/4K4 w", "e1f3") == "马五进四"
    assert notation("4k4/9/9/9/9/4P4/9/9/4A4/4K4 w", "e1f2") == "仕五进四"
    assert notation("4k4/9/9/9/9/4P4/9/4B4/9/4K4 w", "e2g4") == "相五进三"


def test_standard_file_numbering_is_red_right_to_left_and_black_left_to_right() -> None:
    assert notation("4k4/9/9/9/9/9/9/7C1/9/3K5 w", "h2e2") == "炮二平五"
    assert notation("3k5/9/7c1/9/9/9/9/9/9/4K4 b", "h7e7") == "炮8平5"


def test_same_file_rooks_get_front_and_rear_prefixes_from_mover_perspective() -> None:
    board = "4k4/9/9/9/9/4R4/9/4R4/9/4K4 w"
    assert notation(board, "e4e5") == "前车进一"
    assert notation(board, "e2e3") == "后车进一"

    black_board = "4k4/9/4r4/9/4r4/9/9/9/9/4K4 b"
    assert notation(black_board, "e5e4") == "前车进1"
    assert notation(black_board, "e7e6") == "后车进1"


def test_three_same_file_pawns_get_front_middle_and_rear_prefixes() -> None:
    board = "4k4/9/4P4/9/4P4/9/4P4/9/9/4K4 w"
    assert notation(board, "e7e8") == "前兵进一"
    assert notation(board, "e5e6") == "中兵进一"
    assert notation(board, "e3e4") == "后兵进一"


@pytest.mark.parametrize(
    ("board", "moves", "expected"),
    (
        (
            "3k5/4P4/4P4/4P4/4P4/9/9/9/9/4K4 w",
            ("e8d8", "e7d7", "e6d6", "e5d5"),
            ("前一兵平六", "前二兵平六", "后一兵平六", "后二兵平六"),
        ),
        (
            "3kP4/4P4/4P4/4P4/4P4/9/9/9/9/4K4 w",
            ("e9f9", "e8f8", "e7f7", "e6f6", "e5f5"),
            ("前一兵平四", "前二兵平四", "中兵平四", "后一兵平四", "后二兵平四"),
        ),
        (
            "3k5/9/9/9/9/4p4/4p4/4p4/4p4/4K4 b",
            ("e1f1", "e2f2", "e3f3", "e4f4"),
            ("前1卒平6", "前2卒平6", "后1卒平6", "后2卒平6"),
        ),
        (
            "3k5/9/9/9/9/4p4/4p4/4p4/4p4/4p1K2 b",
            ("e0f0", "e1f1", "e2f2", "e3f3", "e4f4"),
            ("前1卒平6", "前2卒平6", "中卒平6", "后1卒平6", "后2卒平6"),
        ),
    ),
)
def test_four_and_five_same_file_pawns_use_group_relative_labels(
    board: str, moves: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert tuple(notation(board, move) for move in moves) == expected


def test_notation_rejects_illegal_move() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    with pytest.raises(ValueError, match="legal move"):
        to_chinese(board, next(iter(legal_moves(board))).__class__("e0e2", 85, 67))
