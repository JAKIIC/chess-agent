"""Chinese Xiangqi notation rendered from a validated legal move."""

from xiangqi_agent.domain.board import BoardState, Move, Side
from xiangqi_agent.domain.rules import legal_moves

_RED_NAMES = {"K": "帅", "A": "仕", "B": "相", "N": "马", "R": "车", "C": "炮", "P": "兵"}
_BLACK_NAMES = {"K": "将", "A": "士", "B": "象", "N": "马", "R": "车", "C": "炮", "P": "卒"}
_RED_NUMERALS = ("一", "二", "三", "四", "五", "六", "七", "八", "九")
_BLACK_NUMERALS = ("1", "2", "3", "4", "5", "6", "7", "8", "9")


def to_chinese(board: BoardState, move: Move) -> str:
    """Render one legal move using standard Red/Black Chinese Xiangqi notation."""
    if not isinstance(move, Move) or move not in legal_moves(board):
        raise ValueError("move must be a legal move for the board")

    piece = board.pieces[move.from_index]
    side: Side = "w" if piece.isupper() else "b"
    source_row, source_column = divmod(move.from_index, 9)
    target_row, target_column = divmod(move.to_index, 9)
    kind = piece.upper()
    prefix = _disambiguator(board, move.from_index, kind, side)
    name = (_RED_NAMES if side == "w" else _BLACK_NAMES)[kind]
    origin = "" if prefix else _file_numeral(source_column, side)

    if source_row == target_row:
        action = "平"
        detail = _file_numeral(target_column, side)
    else:
        action = "进" if _moves_toward_opponent(source_row, target_row, side) else "退"
        detail = (
            _file_numeral(target_column, side)
            if kind in {"N", "A", "B"}
            else _distance_numeral(abs(target_row - source_row), side)
        )
    return f"{prefix}{name}{origin}{action}{detail}"


def resolve_move_reference(board: BoardState, text: str) -> Move | None:
    """Resolve one exact Chinese-notation move mention from the current legal moves."""
    if not isinstance(text, str):
        raise TypeError("move reference text must be a string")
    normalized = "".join(text.split())
    matches = tuple(
        move for move in legal_moves(board) if to_chinese(board, move) in normalized
    )
    return matches[0] if len(matches) == 1 else None


def _disambiguator(board: BoardState, source_index: int, kind: str, side: Side) -> str:
    _, source_column = divmod(source_index, 9)
    same_file = [
        index
        for index, piece in enumerate(board.pieces)
        if piece.upper() == kind
        and (piece.isupper() if side == "w" else piece.islower())
        and index % 9 == source_column
    ]
    if len(same_file) == 1:
        return ""
    same_file.sort(key=lambda index: index // 9, reverse=side == "b")
    position = same_file.index(source_index)
    if len(same_file) == 2:
        return "前" if position == 0 else "后"
    if len(same_file) == 3:
        return ("前", "中", "后")[position]
    front_count = len(same_file) // 2
    if len(same_file) % 2 and position == front_count:
        return "中"
    if position < front_count:
        return f"前{_distance_numeral(position + 1, side)}"
    rear_position = position - front_count - (len(same_file) % 2)
    return f"后{_distance_numeral(rear_position + 1, side)}"


def _file_numeral(column: int, side: Side) -> str:
    number = 9 - column if side == "w" else column + 1
    return (_RED_NUMERALS if side == "w" else _BLACK_NUMERALS)[number - 1]


def _distance_numeral(distance: int, side: Side) -> str:
    return (_RED_NUMERALS if side == "w" else _BLACK_NUMERALS)[distance - 1]


def _moves_toward_opponent(source_row: int, target_row: int, side: Side) -> bool:
    return target_row < source_row if side == "w" else target_row > source_row
