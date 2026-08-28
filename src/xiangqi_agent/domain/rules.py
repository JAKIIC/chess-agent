"""Pure, deterministic Xiangqi move rules for the normalized board orientation."""

from collections.abc import Iterator
from typing import cast

from xiangqi_agent.domain.board import BoardState, Move, Side

_BOARD_ROWS = 10
_BOARD_COLUMNS = 9
_ORTHOGONAL_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))
_HORSE_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))
_ELEPHANT_DELTAS = ((-2, -2), (-2, 2), (2, -2), (2, 2))
_ADVISOR_DELTAS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_GENERAL_COUNT_ERROR = "board must contain exactly one K and one k"


def legal_moves(board: BoardState) -> tuple[Move, ...]:
    """Return all legal moves for the side to move in a stable order."""
    _validate_generals(board)
    side = board.side_to_move
    legal = (
        move
        for move in _pseudo_legal_moves(board, side)
        if not is_in_check(_apply_unchecked(board, move), side)
    )
    return tuple(sorted(legal, key=lambda move: (move.from_index, move.uci)))


def apply_move(board: BoardState, move: Move) -> BoardState:
    """Apply one legal move while preserving display orientation and advancing the ply."""
    _validate_generals(board)
    if not isinstance(move, Move) or move not in legal_moves(board):
        raise ValueError("move must be a legal move for the board")
    return _apply_unchecked(board, move)


def is_in_check(board: BoardState, side: Side) -> bool:
    """Return whether *side*'s general is attacked in the supplied position."""
    _validate_generals(board)
    if side not in ("w", "b"):
        raise ValueError("side must be 'w' or 'b'")

    general_index = _find_general(board, side)
    opponent = _other_side(side)
    if _generals_face(board):
        return True
    return any(
        _piece_attacks_square(board, source_index, general_index)
        for source_index, piece in enumerate(board.pieces)
        if _piece_side(piece) == opponent
    )


def detect_unique_move(before: BoardState, after: BoardState) -> Move:
    """Find the one legal move that changes ``before`` into ``after``.

    Observation metadata (orientation and ply) is intentionally excluded because a frame
    comparison represents only the pieces and the player to move.
    """
    _validate_generals(before)
    _validate_generals(after)
    matches = tuple(
        move
        for move in legal_moves(before)
        if _same_frame(_apply_unchecked(before, move), after)
    )
    if len(matches) != 1:
        raise ValueError("frame change does not contain a unique legal move")
    return matches[0]


def _pseudo_legal_moves(board: BoardState, side: Side) -> Iterator[Move]:
    for source_index, piece in enumerate(board.pieces):
        if _piece_side(piece) != side:
            continue
        row, column = divmod(source_index, _BOARD_COLUMNS)
        kind = piece.upper()
        if kind == "K":
            yield from _king_moves(board, source_index, row, column, side)
        elif kind == "A":
            yield from _advisor_moves(board, source_index, row, column, side)
        elif kind == "B":
            yield from _elephant_moves(board, source_index, row, column, side)
        elif kind == "N":
            yield from _horse_moves(board, source_index, row, column)
        elif kind == "R":
            yield from _rook_moves(board, source_index, row, column)
        elif kind == "C":
            yield from _cannon_moves(board, source_index, row, column)
        elif kind == "P":
            yield from _pawn_moves(board, source_index, row, column, side)


def _king_moves(
    board: BoardState, source_index: int, row: int, column: int, side: Side
) -> Iterator[Move]:
    for row_delta, column_delta in _ORTHOGONAL_DIRECTIONS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        if _in_palace(destination_row, destination_column, side):
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move


def _advisor_moves(
    board: BoardState, source_index: int, row: int, column: int, side: Side
) -> Iterator[Move]:
    for row_delta, column_delta in _ADVISOR_DELTAS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        if _in_palace(destination_row, destination_column, side):
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move


def _elephant_moves(
    board: BoardState, source_index: int, row: int, column: int, side: Side
) -> Iterator[Move]:
    for row_delta, column_delta in _ELEPHANT_DELTAS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        eye_row = row + row_delta // 2
        eye_column = column + column_delta // 2
        if (
            _in_bounds(destination_row, destination_column)
            and _on_own_side_of_river(destination_row, side)
            and _piece_at(board, eye_row, eye_column) == "."
        ):
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move


def _horse_moves(board: BoardState, source_index: int, row: int, column: int) -> Iterator[Move]:
    for row_delta, column_delta in _HORSE_DELTAS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        leg_row, leg_column = (
            (row + row_delta // 2, column)
            if abs(row_delta) == 2
            else (row, column + column_delta // 2)
        )
        if (
            _in_bounds(destination_row, destination_column)
            and _piece_at(board, leg_row, leg_column) == "."
        ):
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move


def _rook_moves(board: BoardState, source_index: int, row: int, column: int) -> Iterator[Move]:
    for row_delta, column_delta in _ORTHOGONAL_DIRECTIONS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        while _in_bounds(destination_row, destination_column):
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move
            if _piece_at(board, destination_row, destination_column) != ".":
                break
            destination_row += row_delta
            destination_column += column_delta


def _cannon_moves(board: BoardState, source_index: int, row: int, column: int) -> Iterator[Move]:
    for row_delta, column_delta in _ORTHOGONAL_DIRECTIONS:
        destination_row = row + row_delta
        destination_column = column + column_delta
        while _in_bounds(destination_row, destination_column):
            if _piece_at(board, destination_row, destination_column) != ".":
                break
            move = _make_move(board, source_index, destination_row, destination_column)
            if move is not None:
                yield move
            destination_row += row_delta
            destination_column += column_delta

        destination_row += row_delta
        destination_column += column_delta
        while _in_bounds(destination_row, destination_column):
            if _piece_at(board, destination_row, destination_column) != ".":
                move = _make_move(board, source_index, destination_row, destination_column)
                if move is not None:
                    yield move
                break
            destination_row += row_delta
            destination_column += column_delta


def _pawn_moves(
    board: BoardState, source_index: int, row: int, column: int, side: Side
) -> Iterator[Move]:
    forward = -1 if side == "w" else 1
    destinations = [(row + forward, column)]
    if _has_crossed_river(row, side):
        destinations.extend(((row, column - 1), (row, column + 1)))
    for destination_row, destination_column in destinations:
        move = _make_move(board, source_index, destination_row, destination_column)
        if move is not None:
            yield move


def _piece_attacks_square(board: BoardState, source_index: int, target_index: int) -> bool:
    source_row, source_column = divmod(source_index, _BOARD_COLUMNS)
    target_row, target_column = divmod(target_index, _BOARD_COLUMNS)
    piece = board.pieces[source_index]
    side = cast(Side, _piece_side(piece))
    kind = piece.upper()
    row_delta = target_row - source_row
    column_delta = target_column - source_column

    if kind == "K":
        return (
            (abs(row_delta), abs(column_delta)) in ((1, 0), (0, 1))
            and _in_palace(target_row, target_column, side)
        )
    if kind == "A":
        return (
            (abs(row_delta), abs(column_delta)) == (1, 1)
            and _in_palace(target_row, target_column, side)
        )
    if kind == "B":
        return (
            (abs(row_delta), abs(column_delta)) == (2, 2)
            and _on_own_side_of_river(target_row, side)
            and _piece_at(board, source_row + row_delta // 2, source_column + column_delta // 2) == "."
        )
    if kind == "N":
        if (abs(row_delta), abs(column_delta)) not in ((2, 1), (1, 2)):
            return False
        leg_row, leg_column = (
            (source_row + row_delta // 2, source_column)
            if abs(row_delta) == 2
            else (source_row, source_column + column_delta // 2)
        )
        return _piece_at(board, leg_row, leg_column) == "."
    if kind == "R":
        return _is_clear_orthogonal_path(board, source_row, source_column, target_row, target_column)
    if kind == "C":
        return _screen_count(board, source_row, source_column, target_row, target_column) == 1
    if kind == "P":
        forward = -1 if side == "w" else 1
        if (row_delta, column_delta) == (forward, 0):
            return True
        return _has_crossed_river(source_row, side) and row_delta == 0 and abs(column_delta) == 1
    return False


def _make_move(
    board: BoardState, source_index: int, destination_row: int, destination_column: int
) -> Move | None:
    if not _in_bounds(destination_row, destination_column):
        return None
    destination_index = _index(destination_row, destination_column)
    source_piece = board.pieces[source_index]
    destination_piece = board.pieces[destination_index]
    if _piece_side(destination_piece) == _piece_side(source_piece) or destination_piece.upper() == "K":
        return None
    return Move(
        uci=_uci(source_index) + _uci(destination_index),
        from_index=source_index,
        to_index=destination_index,
        captured=None if destination_piece == "." else destination_piece,
    )


def _apply_unchecked(board: BoardState, move: Move) -> BoardState:
    pieces = list(board.pieces)
    pieces[move.to_index] = pieces[move.from_index]
    pieces[move.from_index] = "."
    next_side: Side = "b" if board.side_to_move == "w" else "w"
    return BoardState(
        pieces=tuple(pieces),
        side_to_move=next_side,
        orientation=board.orientation,
        ply=board.ply + 1,
    )


def _same_frame(left: BoardState, right: BoardState) -> bool:
    return left.pieces == right.pieces and left.side_to_move == right.side_to_move


def _find_general(board: BoardState, side: Side) -> int:
    _validate_generals(board)
    general = "K" if side == "w" else "k"
    return board.pieces.index(general)


def _validate_generals(board: BoardState) -> None:
    if board.pieces.count("K") != 1 or board.pieces.count("k") != 1:
        raise ValueError(_GENERAL_COUNT_ERROR)


def _generals_face(board: BoardState) -> bool:
    red_general = _find_general(board, "w")
    black_general = _find_general(board, "b")
    red_row, red_column = divmod(red_general, _BOARD_COLUMNS)
    black_row, black_column = divmod(black_general, _BOARD_COLUMNS)
    return red_column == black_column and _is_clear_orthogonal_path(
        board, red_row, red_column, black_row, black_column
    )


def _is_clear_orthogonal_path(
    board: BoardState, start_row: int, start_column: int, end_row: int, end_column: int
) -> bool:
    if start_row != end_row and start_column != end_column:
        return False
    row_step = (end_row > start_row) - (end_row < start_row)
    column_step = (end_column > start_column) - (end_column < start_column)
    if row_step == 0 and column_step == 0:
        return False
    row = start_row + row_step
    column = start_column + column_step
    while (row, column) != (end_row, end_column):
        if _piece_at(board, row, column) != ".":
            return False
        row += row_step
        column += column_step
    return True


def _screen_count(
    board: BoardState, start_row: int, start_column: int, end_row: int, end_column: int
) -> int:
    if start_row != end_row and start_column != end_column:
        return 0
    row_step = (end_row > start_row) - (end_row < start_row)
    column_step = (end_column > start_column) - (end_column < start_column)
    if row_step == 0 and column_step == 0:
        return 0
    row = start_row + row_step
    column = start_column + column_step
    screens = 0
    while (row, column) != (end_row, end_column):
        if _piece_at(board, row, column) != ".":
            screens += 1
        row += row_step
        column += column_step
    return screens


def _piece_at(board: BoardState, row: int, column: int) -> str:
    return board.pieces[_index(row, column)]


def _piece_side(piece: str) -> Side | None:
    if piece == ".":
        return None
    return "w" if piece.isupper() else "b"


def _other_side(side: Side) -> Side:
    return "b" if side == "w" else "w"


def _in_bounds(row: int, column: int) -> bool:
    return 0 <= row < _BOARD_ROWS and 0 <= column < _BOARD_COLUMNS


def _in_palace(row: int, column: int, side: Side) -> bool:
    return 3 <= column <= 5 and (7 <= row <= 9 if side == "w" else 0 <= row <= 2)


def _on_own_side_of_river(row: int, side: Side) -> bool:
    return row >= 5 if side == "w" else row <= 4


def _has_crossed_river(row: int, side: Side) -> bool:
    return row <= 4 if side == "w" else row >= 5


def _index(row: int, column: int) -> int:
    return row * _BOARD_COLUMNS + column


def _uci(index: int) -> str:
    row, column = divmod(index, _BOARD_COLUMNS)
    return f"{chr(ord('a') + column)}{9 - row}"
