from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from xiangqi_agent.sync.evidence import SequenceCandidateEvidence

_SEMANTIC_ARTIFACT_PROFILE_VERSION = "human-ai-two-ply-v2"
_MAX_RELATIVE_ARTIFACT_DIFFERENCE = 0.30


@dataclass(frozen=True, slots=True, kw_only=True)
class SequenceThresholdProfile:
    min_local_difference: float
    max_unexpected_difference: float
    min_score: float
    min_margin: float
    max_template_distance: float
    min_template_margin: float
    min_template_confidence: float
    profile_version: str

    def __post_init__(self) -> None:
        positive_thresholds = (
            self.min_local_difference,
            self.max_unexpected_difference,
            self.min_score,
            self.min_margin,
            self.max_template_distance,
            self.min_template_margin,
        )
        if any(not isfinite(value) or value <= 0 for value in positive_thresholds):
            raise ValueError("sequence thresholds must be finite and positive")
        if (
            not isfinite(self.min_template_confidence)
            or not 0 < self.min_template_confidence <= 1
        ):
            raise ValueError(
                "min_template_confidence must be finite and between zero and one"
            )
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise ValueError("profile_version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SequenceDecision:
    accepted: bool
    candidate: SequenceCandidateEvidence | None
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.accepted and self.candidate is None:
            raise ValueError("accepted sequence decision must expose one candidate")
        if not self.accepted and self.candidate is not None:
            raise ValueError("rejected sequence decision must not expose a candidate")
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted sequence decision must not contain rejection reasons")


class SequenceDecisionGate:
    def __init__(self, profile: SequenceThresholdProfile) -> None:
        if not isinstance(profile, SequenceThresholdProfile):
            raise TypeError("profile must be a SequenceThresholdProfile")
        self._profile = profile

    @property
    def profile(self) -> SequenceThresholdProfile:
        return self._profile

    def evaluate(
        self,
        candidates: tuple[SequenceCandidateEvidence, ...],
        *,
        template_unavailable: bool = False,
    ) -> SequenceDecision:
        if not isinstance(candidates, tuple) or any(
            not isinstance(candidate, SequenceCandidateEvidence) for candidate in candidates
        ):
            raise TypeError("candidates must be a tuple of SequenceCandidateEvidence")
        if not isinstance(template_unavailable, bool):
            raise TypeError("template_unavailable must be a boolean")
        if not candidates:
            reason = "template_unavailable" if template_unavailable else "no_legal_candidates"
            return SequenceDecision(False, None, (reason,))

        best = candidates[0]
        next_score = candidates[1].score if len(candidates) > 1 else 0.0
        profile = self._profile
        semantic_artifact_profile = (
            profile.profile_version == _SEMANTIC_ARTIFACT_PROFILE_VERSION
        )
        unexpected_limit = profile.max_unexpected_difference
        if semantic_artifact_profile:
            # V2 observers fold every above-baseline outside patch into the
            # template distance/confidence minima. The relative allowance is
            # therefore only for presentation artifacts that still preserve
            # the confirmed empty/red/black semantic class.
            unexpected_limit = max(
                unexpected_limit,
                best.expected_change_floor * _MAX_RELATIVE_ARTIFACT_DIFFERENCE,
            )
        reasons: list[str] = []
        if best.expected_change_floor < profile.min_local_difference:
            reasons.append("expected_change")
        if best.unexpected_difference > unexpected_limit:
            reasons.append("outside_change")
        if best.score < profile.min_score:
            reasons.append("candidate_score")
        if best.score - next_score < profile.min_margin:
            reasons.append("candidate_margin")
        if best.maximum_template_distance > profile.max_template_distance:
            reasons.append("template_distance")
        if (
            best.minimum_template_margin < profile.min_template_margin
            and not semantic_artifact_profile
        ):
            # Exact piece-class margin is advisory in V2. Legal-chain
            # uniqueness identifies the piece; distance and semantic
            # confidence remain hard gates under last-move tinting.
            reasons.append("template_margin")
        if best.minimum_template_confidence < profile.min_template_confidence:
            reasons.append("template_confidence")
        if reasons:
            return SequenceDecision(False, None, tuple(reasons))
        return SequenceDecision(True, best, ())
