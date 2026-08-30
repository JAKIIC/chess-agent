from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from xiangqi_agent.domain.board import Move


class ObservationStatus(StrEnum):
    ACCEPTED = "accepted"
    NO_CHANGE = "no_change"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    move: Move
    source_difference: float
    destination_difference: float
    unexpected_difference: float
    source_expected_distance: float
    destination_expected_distance: float
    semantic_margin: float
    source_semantic_evidence_score: float
    destination_semantic_evidence_score: float
    score: float


@dataclass(frozen=True, slots=True)
class MoveEvidence:
    candidates: tuple[CandidateEvidence, ...]
    local_differences: tuple[float, ...]
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MoveProposal:
    status: ObservationStatus
    move: Move | None
    evidence_score: float
    evidence: MoveEvidence
