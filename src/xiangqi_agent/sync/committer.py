from __future__ import annotations

from typing import Protocol

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import apply_move


class StateCommitter(Protocol):
    def commit(self, board: BoardState, move: Move) -> BoardState: ...

    def project(self, board: BoardState, moves: tuple[Move, ...]) -> BoardState: ...

    def commit_sequence(
        self,
        board: BoardState,
        moves: tuple[Move, Move],
    ) -> BoardState: ...


class RuleStateCommitter:
    """Advance an immutable board only through the domain legality boundary."""

    def commit(self, board: BoardState, move: Move) -> BoardState:
        return apply_move(board, move)

    def project(self, board: BoardState, moves: tuple[Move, ...]) -> BoardState:
        if not moves or len(moves) > 2:
            raise ValueError("projection must contain one or two moves")
        projected = board
        for move in moves:
            projected = apply_move(projected, move)
        return projected

    def commit_sequence(
        self,
        board: BoardState,
        moves: tuple[Move, Move],
    ) -> BoardState:
        if len(moves) != 2:
            raise ValueError("atomic sequence must contain exactly two moves")
        return self.project(board, moves)
