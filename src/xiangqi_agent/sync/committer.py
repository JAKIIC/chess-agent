from __future__ import annotations

from typing import Protocol

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import apply_move


class StateCommitter(Protocol):
    def commit(self, board: BoardState, move: Move) -> BoardState: ...


class RuleStateCommitter:
    """Advance an immutable board only through the domain legality boundary."""

    def commit(self, board: BoardState, move: Move) -> BoardState:
        return apply_move(board, move)
