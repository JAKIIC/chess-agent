from __future__ import annotations

from dataclasses import dataclass

from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import apply_move, legal_moves


@dataclass(frozen=True, slots=True)
class AnalysisRow:
    rank: int
    notation: str
    uci: str
    score: str
    depth: int
    variation: str


def analysis_rows(board: BoardState, analysis: EngineAnalysis) -> tuple[AnalysisRow, ...]:
    if analysis.position_id != board.position_id:
        raise ValueError("analysis position does not match the confirmed board")
    return tuple(_line_to_row(board, line) for line in analysis.lines)


def _line_to_row(board: BoardState, line: EngineLine) -> AnalysisRow:
    first_move = line.pv[0]
    notation = _notation_for_uci(board, first_move)
    return AnalysisRow(
        rank=line.multipv,
        notation=notation,
        uci=first_move,
        score=_format_score(line),
        depth=line.depth,
        variation="  ".join(_variation_notation(board, line.pv[:8])),
    )


def _variation_notation(board: BoardState, variation: tuple[str, ...]) -> tuple[str, ...]:
    current = board
    rendered: list[str] = []
    for uci in variation:
        move = next((candidate for candidate in legal_moves(current) if candidate.uci == uci), None)
        if move is None:
            rendered.append(uci)
            break
        rendered.append(to_chinese(current, move))
        current = apply_move(current, move)
    return tuple(rendered)


def _notation_for_uci(board: BoardState, uci: str) -> str:
    move = next((candidate for candidate in legal_moves(board) if candidate.uci == uci), None)
    return uci if move is None else to_chinese(board, move)


def _format_score(line: EngineLine) -> str:
    if line.score_cp is not None:
        return f"{line.score_cp / 100:+.2f}"
    if line.mate_in is None:
        raise ValueError("analysis line has no score")
    side = "红方" if line.mate_in > 0 else "黑方"
    return f"{side}杀{abs(line.mate_in)}步"
