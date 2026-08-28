import pytest

from xiangqi_agent.domain.board import Move, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, detect_unique_move, is_in_check, legal_moves


def moves(board_fen: str) -> set[str]:
    return {move.uci for move in legal_moves(parse_fen(board_fen))}


def test_horse_leg_blocks_move() -> None:
    assert "d0c2" not in moves("4k4/9/9/9/9/4P4/9/9/3P5/3NK4 w")


def test_cannon_requires_one_screen_to_capture_and_none_to_move() -> None:
    assert "e1e4" in moves("4k4/9/9/9/9/4p4/9/4P4/4C4/3K5 w")
    assert "e1e4" in moves("4k4/9/9/9/9/9/9/9/4C4/3K5 w")
    assert "e1e4" not in moves("4k4/9/9/9/9/9/9/4P4/4C4/3K5 w")
    assert "e1e4" not in moves("4k4/9/9/9/9/4p4/9/9/4C4/3K5 w")


def test_elephant_eye_and_river_limit_moves() -> None:
    assert "e2g4" not in moves("4k4/9/9/9/9/9/4P4/9/4B4/4K4 w")
    assert "e4g6" not in moves("4k4/9/9/9/4B4/9/9/9/9/4K4 w")


def test_advisor_and_general_stay_inside_their_palaces() -> None:
    board = "4k4/9/9/9/9/4P4/9/9/3A5/4K4 w"
    assert "d1c2" not in moves(board)
    assert "e0e1" in moves(board)
    assert "e0f0" in moves(board)


def test_pawn_only_moves_forward_before_river_and_can_move_sideways_after() -> None:
    before_river = moves("4k4/9/9/9/9/9/4P4/9/9/4K4 w")
    after_river = moves("4k4/9/9/9/4P4/9/4P4/9/9/4K4 w")
    assert "e3e4" in before_river
    assert "e3d3" not in before_river
    assert "e3f3" not in before_river
    assert {"e5e6", "e5d5", "e5f5"} <= after_river
    assert "e5e4" not in after_river


def test_black_pawn_uses_the_opposite_direction_and_river_boundary() -> None:
    legal = moves("4k4/9/9/4p4/9/4p4/9/9/9/4K4 b")
    assert "e6e5" in legal
    assert "e6d6" not in legal
    assert "e6f6" not in legal
    assert {"e4e3", "e4d4", "e4f4"} <= legal
    assert "e4e5" not in legal


def test_flying_generals_are_check_and_illegal_to_expose() -> None:
    facing = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    assert is_in_check(facing, "w")
    assert is_in_check(facing, "b")

    pinned = moves("4k4/9/9/9/9/9/9/9/4R4/4K4 w")
    assert "e1d1" not in pinned


def test_self_check_filter_rejects_moving_a_rook_that_shields_king() -> None:
    legal = moves("4k4/9/9/9/9/9/9/9/4R4/4K4 w")
    assert "e1d1" not in legal
    assert "e1e2" in legal


def test_legal_moves_reject_friendly_capture_and_other_side_pieces() -> None:
    legal = moves("4k4/9/9/9/9/9/9/9/3PR4/4K4 w")
    assert "e1d1" not in legal
    black_to_move = moves("4k4/9/9/9/9/9/9/9/9/4K4 b")
    assert black_to_move == {"e9d9", "e9f9"}


def test_apply_move_updates_capture_turn_ply_and_preserves_orientation() -> None:
    board = parse_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w")
    board = board.__class__(
        pieces=board.pieces,
        side_to_move=board.side_to_move,
        orientation=Orientation.BLACK_BOTTOM,
        ply=7,
    )
    move = next(move for move in legal_moves(board) if move.uci == "e4e5")

    after = apply_move(board, move)

    assert move.captured == "p"
    assert after.pieces[40] == "R"
    assert after.pieces[49] == "."
    assert after.side_to_move == "b"
    assert after.ply == 8
    assert after.orientation is Orientation.BLACK_BOTTOM


def test_apply_move_records_no_capture_for_a_normal_move() -> None:
    board = parse_fen("4k4/9/9/9/4P4/9/9/9/9/4K4 w")
    move = next(move for move in legal_moves(board) if move.uci == "e5e6")

    after = apply_move(board, move)

    assert move.captured is None
    assert after.pieces[31] == "P"
    assert after.pieces[40] == "."


def test_apply_move_rejects_non_legal_or_mismatched_move() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    with pytest.raises(ValueError, match="legal move"):
        apply_move(board, Move("e0e2", 85, 67))


def test_legal_moves_have_deterministic_order() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    assert tuple(move.uci for move in legal_moves(board)) == ("e0d0", "e0f0")


def test_detect_unique_capture_ignores_orientation_and_ply() -> None:
    before = parse_fen("4k4/9/9/9/4p4/4R4/9/9/9/4K4 w")
    after = parse_fen("4k4/9/9/9/4R4/9/9/9/9/4K4 b")
    after = after.__class__(
        pieces=after.pieces,
        side_to_move=after.side_to_move,
        orientation=Orientation.BLACK_BOTTOM,
        ply=99,
    )
    assert detect_unique_move(before, after).uci == "e4e5"


@pytest.mark.parametrize(
    "after_fen",
    (
        "4k4/9/9/9/9/9/9/9/9/4K4 b",
        "3k5/9/9/9/9/9/9/9/9/5K3 b",
    ),
)
def test_detect_unique_move_rejects_zero_or_multiple_frame_changes(after_fen: str) -> None:
    before = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    with pytest.raises(ValueError, match="unique legal move"):
        detect_unique_move(before, parse_fen(after_fen))
