from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from xiangqi_agent.domain.board import Move
from xiangqi_agent.vision.endpoint_features import EndpointFeatures


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
    endpoint_features: EndpointFeatures | None = None


@dataclass(frozen=True, slots=True)
class MoveProposal:
    status: ObservationStatus
    move: Move | None
    evidence_score: float
    evidence: MoveEvidence


@dataclass(frozen=True, slots=True)
class SequenceCandidateEvidence:
    moves: tuple[Move, Move]
    changed_points: tuple[int, ...]
    expected_change_floor: float
    unexpected_difference: float
    maximum_template_distance: float
    minimum_template_margin: float
    minimum_template_confidence: float
    score: float
    final_position_id: str


@dataclass(frozen=True, slots=True)
class MoveSequenceEvidence:
    candidates: tuple[SequenceCandidateEvidence, ...]
    local_differences: tuple[float, ...]
    rejection_reasons: tuple[str, ...]
    feature_version: str


@dataclass(frozen=True, slots=True)
class MoveSequenceProposal:
    status: ObservationStatus
    moves: tuple[Move, ...]
    evidence_score: float
    evidence: MoveSequenceEvidence

    def __post_init__(self) -> None:
        if self.status is ObservationStatus.ACCEPTED and len(self.moves) != 2:
            raise ValueError("accepted sequence must contain exactly two moves")
        if self.status is not ObservationStatus.ACCEPTED and self.moves:
            raise ValueError("rejected sequence must not expose moves")
