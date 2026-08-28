from typing import cast

from xiangqi_agent.domain.board import VALID_PIECES, BoardState, Side

_FEN_PIECES = VALID_PIECES - {"."}


def parse_fen(text: str) -> BoardState:
    fields = text.split()
    if len(fields) not in (2, 6):
        raise ValueError("FEN must contain board and side, with optional four extension fields")

    board_field, side = fields[:2]
    if side not in ("w", "b"):
        raise ValueError("side to move must be 'w' or 'b'")
    if len(fields) == 6:
        _validate_extension_fields(fields[2:])

    ranks = board_field.split("/")
    if len(ranks) != 10:
        raise ValueError("FEN board must contain exactly 10 ranks")

    pieces: list[str] = []
    for rank in ranks:
        pieces.extend(_parse_rank(rank))

    return BoardState(pieces=tuple(pieces), side_to_move=cast(Side, side))


def to_fen(board: BoardState) -> str:
    ranks = (_serialize_rank(board.pieces[start : start + 9]) for start in range(0, 90, 9))
    return f"{'/'.join(ranks)} {board.side_to_move}"


def _validate_extension_fields(fields: list[str]) -> None:
    castling, en_passant, halfmove, fullmove = fields
    if castling != "-" or en_passant != "-":
        raise ValueError("Xiangqi FEN extension markers must be '-'")
    if not halfmove.isascii() or not halfmove.isdecimal():
        raise ValueError("FEN halfmove clock must be a non-negative integer")
    if not fullmove.isascii() or not fullmove.isdecimal() or int(fullmove) < 1:
        raise ValueError("FEN fullmove number must be a positive integer")


def _parse_rank(rank: str) -> tuple[str, ...]:
    pieces: list[str] = []
    previous_was_digit = False
    for symbol in rank:
        if "1" <= symbol <= "9":
            if previous_was_digit:
                raise ValueError("FEN rank contains consecutive empty counts")
            pieces.extend("." for _ in range(int(symbol)))
            previous_was_digit = True
        elif symbol in _FEN_PIECES:
            pieces.append(symbol)
            previous_was_digit = False
        else:
            raise ValueError("FEN rank contains an unknown piece symbol")

    if len(pieces) != 9:
        raise ValueError("FEN rank must expand to exactly 9 files")
    return tuple(pieces)


def _serialize_rank(rank: tuple[str, ...]) -> str:
    fields: list[str] = []
    empty_count = 0
    for piece in rank:
        if piece == ".":
            empty_count += 1
            continue
        if empty_count:
            fields.append(str(empty_count))
            empty_count = 0
        fields.append(piece)
    if empty_count:
        fields.append(str(empty_count))
    return "".join(fields)
