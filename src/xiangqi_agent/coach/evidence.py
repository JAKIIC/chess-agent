from __future__ import annotations

import re

from xiangqi_agent.domain.analysis import EngineAnalysis
from xiangqi_agent.domain.board import BoardState, Side
from xiangqi_agent.domain.coach import (
    CoachCandidate,
    CoachEvidence,
    GamePhase,
    ImmediateTactic,
)
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import apply_move, is_in_check, legal_moves

_UCI_MOVE = re.compile(r"[a-i][0-9][a-i][0-9]")
_MATERIAL_NAMES = {
    "K": "red_general",
    "A": "red_advisor",
    "B": "red_elephant",
    "N": "red_horse",
    "R": "red_rook",
    "C": "red_cannon",
    "P": "red_pawn",
    "k": "black_general",
    "a": "black_advisor",
    "b": "black_elephant",
    "n": "black_horse",
    "r": "black_rook",
    "c": "black_cannon",
    "p": "black_pawn",
}


def build_evidence(
    board: BoardState,
    analysis: EngineAnalysis,
    *,
    user_side: Side,
    recent_moves: tuple[str, ...] = (),
) -> CoachEvidence:
    if not isinstance(board, BoardState) or not isinstance(analysis, EngineAnalysis):
        raise TypeError("coach evidence requires a confirmed board and engine analysis")
    if user_side not in ("w", "b"):
        raise ValueError("user side must be red or black")
    if analysis.position_id != board.position_id:
        raise ValueError("analysis position does not match the confirmed board")
    if any(_UCI_MOVE.fullmatch(move) is None for move in recent_moves):
        raise ValueError("recent moves must be committed UCI coordinates")

    candidates: list[CoachCandidate] = []
    tactics: list[ImmediateTactic] = []
    seen_uci: set[str] = set()
    for index, line in enumerate(analysis.lines[:3], start=1):
        candidate_id = f"candidate_{index}"
        pv_notation = _validate_and_render_pv(board, line.pv)
        first_move = next(move for move in legal_moves(board) if move.uci == line.pv[0])
        if first_move.uci in seen_uci:
            raise ValueError("engine candidates must be unique legal moves")
        seen_uci.add(first_move.uci)
        after = apply_move(board, first_move)
        candidates.append(
            CoachCandidate(
                candidate_id=candidate_id,
                uci=first_move.uci,
                notation=pv_notation[0],
                score_cp=line.score_cp,
                mate_in=line.mate_in,
                depth=line.depth,
                pv_uci=line.pv,
                pv_notation=pv_notation,
            )
        )
        tactics.append(
            ImmediateTactic(
                candidate_id=candidate_id,
                is_capture=first_move.captured is not None,
                captured_piece=first_move.captured,
                gives_check=is_in_check(after, after.side_to_move),
            )
        )

    material = {name: 0 for name in _MATERIAL_NAMES.values()}
    for piece in board.pieces:
        if piece != ".":
            material[_MATERIAL_NAMES[piece]] += 1
    allowed = {candidate.candidate_id: candidate.notation for candidate in candidates}
    return CoachEvidence(
        position_id=board.position_id,
        fen=board.fen,
        user_side=user_side,
        phase=_phase(board),
        recent_moves=recent_moves,
        material_facts=material,
        king_safety_facts={"side_to_move_in_check": is_in_check(board, board.side_to_move)},
        immediate_tactics=tuple(tactics),
        candidates=tuple(candidates),
        allowed_move_map=allowed,
    )


def _validate_and_render_pv(board: BoardState, pv: tuple[str, ...]) -> tuple[str, ...]:
    current = board
    rendered: list[str] = []
    for uci in pv:
        move = next((candidate for candidate in legal_moves(current) if candidate.uci == uci), None)
        if move is None:
            raise ValueError(f"engine PV contains a move that is not legal: {uci}")
        rendered.append(to_chinese(current, move))
        current = apply_move(current, move)
    return tuple(rendered)


def _phase(board: BoardState) -> GamePhase:
    remaining_non_generals = sum(piece not in (".", "K", "k") for piece in board.pieces)
    if board.ply < 20 and remaining_non_generals >= 24:
        return "opening"
    if remaining_non_generals >= 12:
        return "middlegame"
    return "endgame"
