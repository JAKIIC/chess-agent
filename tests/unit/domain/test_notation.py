import pytest

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import legal_moves


def notation(board_fen: str, uci: str) -> str:
    board = parse_fen(board_fen)
    move = next(move for move in legal_moves(board) if move.uci == uci)
    return to_chinese(board, move)


def test_red_and_black_notation_use_side_specific_names_and_numerals() -> None:
    assert notation("4k4/9/9/9/9/4P4/9/9/9/4K3R w", "i0i1") == "车九进一"
    assert notation("r3k4/9/9/9/9/4p4/9/9/9/4K4 b", "a9a8") == "车9进1"


def test_notation_uses_destination_file_for_horse_advisor_and_elephant() -> None:
    assert notation("4k4/9/9/9/9/4P4/9/9/4N4/4K4 w", "e1f3") == "马五进六"
    assert notation("4k4/9/9/9/9/4P4/9/9/4A4/4K4 w", "e1f2") == "仕五进六"
    assert notation("4k4/9/9/9/9/4P4/9/4B4/9/4K4 w", "e2g4") == "相五进七"


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


def test_notation_rejects_illegal_move() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    with pytest.raises(ValueError, match="legal move"):
        to_chinese(board, next(iter(legal_moves(board))).__class__("e0e2", 85, 67))
