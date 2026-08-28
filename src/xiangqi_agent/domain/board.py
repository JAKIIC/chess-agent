from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Literal

type Side = Literal["w", "b"]
VALID_PIECES = frozenset("KABNRCPkabnrcp.")


class Orientation(StrEnum):
    RED_BOTTOM = "red_bottom"
    BLACK_BOTTOM = "black_bottom"


@dataclass(frozen=True, slots=True)
class Move:
    uci: str
    from_index: int
    to_index: int
    captured: str | None = None


@dataclass(frozen=True, slots=True)
class BoardState:
    pieces: tuple[str, ...]
    side_to_move: Side
    orientation: Orientation = Orientation.RED_BOTTOM
    ply: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.pieces, tuple)
            or len(self.pieces) != 90
            or any(not isinstance(piece, str) or piece not in VALID_PIECES for piece in self.pieces)
        ):
            raise ValueError("board must contain exactly 90 valid intersections")
        if self.side_to_move not in ("w", "b"):
            raise ValueError("side to move must be 'w' or 'b'")
        if not isinstance(self.orientation, Orientation):
            raise ValueError("orientation must be an Orientation")  # noqa: TRY004
        if isinstance(self.ply, bool) or not isinstance(self.ply, int) or self.ply < 0:
            raise ValueError("ply must be a non-negative integer")

    @property
    def fen(self) -> str:
        from xiangqi_agent.domain.fen import to_fen

        return to_fen(self)

    @property
    def position_id(self) -> str:
        return sha256(self.fen.encode("ascii")).hexdigest()[:32]
