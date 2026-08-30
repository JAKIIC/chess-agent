from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from xiangqi_agent.sync.evidence import CandidateEvidence
from xiangqi_agent.vision.endpoint_features import EndpointFeatures


@dataclass(frozen=True, slots=True)
class SemanticThresholds:
    max_source_empty_distance: float
    min_source_empty_evidence_score: float
    max_destination_side_distance: float
    min_destination_side_evidence_score: float
    max_instance_distance: float
    min_instance_evidence_score: float
    min_semantic_margin: float

    def __post_init__(self) -> None:
        distances = (
            self.max_source_empty_distance,
            self.max_destination_side_distance,
            self.max_instance_distance,
            self.min_semantic_margin,
        )
        scores = (
            self.min_source_empty_evidence_score,
            self.min_destination_side_evidence_score,
            self.min_instance_evidence_score,
        )
        if any(not isfinite(value) or value <= 0 for value in distances):
            raise ValueError("semantic distances and margins must be finite and positive")
        if any(not isfinite(value) or not 0 < value <= 1 for value in scores):
            raise ValueError("semantic evidence scores must be finite and between zero and one")


@dataclass(frozen=True, slots=True)
class SemanticGateResult:
    accepted: bool
    rejection_reasons: tuple[str, ...]


class MoveSemanticGate:
    """Evaluate independent endpoint facts without weighted compensation."""

    def __init__(self, thresholds: SemanticThresholds) -> None:
        self._thresholds = thresholds

    def evaluate(
        self,
        candidate: CandidateEvidence,
        features: EndpointFeatures,
    ) -> SemanticGateResult:
        reasons: list[str] = []
        if (
            candidate.source_expected_distance
            > self._thresholds.max_source_empty_distance
            or candidate.source_semantic_evidence_score
            < self._thresholds.min_source_empty_evidence_score
        ):
            reasons.append("source_empty")
        if (
            candidate.destination_expected_distance
            > self._thresholds.max_destination_side_distance
            or candidate.destination_semantic_evidence_score
            < self._thresholds.min_destination_side_evidence_score
        ):
            reasons.append("side")
        if candidate.semantic_margin < self._thresholds.min_semantic_margin:
            reasons.append("semantic_margin")
        if (
            features.instance_distance > self._thresholds.max_instance_distance
            or features.instance_evidence_score
            < self._thresholds.min_instance_evidence_score
        ):
            reasons.append("instance")
        return SemanticGateResult(not reasons, tuple(reasons))
