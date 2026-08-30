from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineLine:
    position_id: str
    depth: int
    seldepth: int | None
    multipv: int
    score_cp: int | None
    mate_in: int | None
    nodes: int | None
    nps: int | None
    time_ms: int | None
    pv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.position_id:
            raise ValueError("position_id must be non-empty")
        if self.depth < 0 or self.multipv <= 0:
            raise ValueError("depth must be non-negative and multipv must be positive")
        if (self.score_cp is None) == (self.mate_in is None):
            raise ValueError("an engine line must contain exactly one cp or mate score")
        if not self.pv:
            raise ValueError("an engine line must contain a principal variation")
        optional_non_negative = (self.seldepth, self.nodes, self.nps, self.time_ms)
        if any(value is not None and value < 0 for value in optional_non_negative):
            raise ValueError("engine counters must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineAnalysis:
    position_id: str
    duration_ms: int
    depth: int
    nodes: int
    lines: tuple[EngineLine, ...]
    bestmove: str | None
    engine_name: str

    def __post_init__(self) -> None:
        if not self.position_id or not self.engine_name:
            raise ValueError("analysis identity and engine name must be non-empty")
        if self.duration_ms < 0 or self.depth < 0 or self.nodes < 0:
            raise ValueError("analysis counters must be non-negative")
        if not self.lines:
            raise ValueError("analysis must contain at least one engine line")
        if tuple(sorted(line.multipv for line in self.lines)) != tuple(
            line.multipv for line in self.lines
        ):
            raise ValueError("analysis lines must be ordered by multipv")
        if any(line.position_id != self.position_id for line in self.lines):
            raise ValueError("analysis lines must match the analysis position")
