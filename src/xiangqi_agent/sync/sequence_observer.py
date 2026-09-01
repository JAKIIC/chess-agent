from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.sync.committer import RuleStateCommitter, StateCommitter
from xiangqi_agent.sync.evidence import (
    MoveSequenceEvidence,
    MoveSequenceProposal,
    ObservationStatus,
    SequenceCandidateEvidence,
)
from xiangqi_agent.sync.sequence_gate import (
    SequenceDecisionGate,
    SequenceThresholdProfile,
)
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import PieceTemplateBank, TemplateMatch

_FEATURE_VERSION = "two-ply-template-v1"


class MoveSequenceObserver(Protocol):
    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveSequenceProposal: ...


class LegalTwoPlyDiffObserver:
    """Accept only one legal two-ply chain that explains a stable final frame."""

    def __init__(
        self,
        *,
        patch_size: int = 48,
        min_local_difference: float = 5.0,
        max_unexpected_difference: float = 3.0,
        min_score: float = 5.0,
        min_margin: float = 5.0,
        max_template_distance: float = 0.18,
        min_template_margin: float = 0.02,
        min_template_confidence: float = 0.8,
        committer: StateCommitter | None = None,
    ) -> None:
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        self._patch_size = patch_size
        self._gate = SequenceDecisionGate(
            SequenceThresholdProfile(
                min_local_difference=min_local_difference,
                max_unexpected_difference=max_unexpected_difference,
                min_score=min_score,
                min_margin=min_margin,
                max_template_distance=max_template_distance,
                min_template_margin=min_template_margin,
                min_template_confidence=min_template_confidence,
                profile_version="human-ai-two-ply-v1",
            )
        )
        self._committer = committer or RuleStateCommitter()

    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveSequenceProposal:
        before_patches = geometry.crop_intersections(before, size=self._patch_size)
        after_patches = geometry.crop_intersections(after, size=self._patch_size)
        local = _local_differences(before_patches, after_patches)
        if max(local, default=0.0) < self._gate.profile.min_local_difference:
            return _proposal(ObservationStatus.NO_CHANGE, (), local, ())

        templates = PieceTemplateBank.from_position(
            board,
            geometry,
            before,
            patch_size=self._patch_size,
        )
        match_cache: dict[tuple[int, str], TemplateMatch] = {}
        candidates: list[SequenceCandidateEvidence] = []
        template_unavailable = False
        for moves, final in self._committer.project_two_ply(board):
            changed_points = tuple(
                index
                for index, (left, right) in enumerate(
                    zip(board.pieces, final.pieces, strict=True)
                )
                if left != right
            )
            if not changed_points:
                continue
            matches: list[TemplateMatch] = []
            try:
                for index in changed_points:
                    symbol = final.pieces[index]
                    key = (index, symbol)
                    match = match_cache.get(key)
                    if match is None:
                        match = templates.match(symbol, after_patches[index])
                        match_cache[key] = match
                    matches.append(match)
            except ValueError:
                template_unavailable = True
                continue
            expected_floor = min(local[index] for index in changed_points)
            changed = frozenset(changed_points)
            unexpected = max(
                (
                    difference
                    for index, difference in enumerate(local)
                    if index not in changed
                ),
                default=0.0,
            )
            candidates.append(
                SequenceCandidateEvidence(
                    moves=moves,
                    changed_points=changed_points,
                    expected_change_floor=expected_floor,
                    unexpected_difference=unexpected,
                    maximum_template_distance=max(match.distance for match in matches),
                    minimum_template_margin=min(match.margin for match in matches),
                    minimum_template_confidence=min(match.confidence for match in matches),
                    score=expected_floor - unexpected,
                    final_position_id=final.position_id,
                )
            )

        ranked = tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    -candidate.score,
                    candidate.moves[0].uci,
                    candidate.moves[1].uci,
                ),
            )
        )
        decision = self._gate.evaluate(
            ranked,
            template_unavailable=template_unavailable,
        )
        if not decision.accepted:
            return _proposal(
                ObservationStatus.AMBIGUOUS,
                ranked,
                local,
                decision.rejection_reasons,
            )

        best = decision.candidate
        if best is None:
            raise RuntimeError("accepted sequence decision did not expose a candidate")

        visual_confidence = best.expected_change_floor / (
            best.expected_change_floor + best.unexpected_difference + 1.0
        )
        return MoveSequenceProposal(
            status=ObservationStatus.ACCEPTED,
            moves=best.moves,
            evidence_score=min(visual_confidence, best.minimum_template_confidence),
            evidence=MoveSequenceEvidence(ranked, local, (), _FEATURE_VERSION),
        )


def _local_differences(
    before_patches: tuple[NDArray[np.uint8], ...],
    after_patches: tuple[NDArray[np.uint8], ...],
) -> tuple[float, ...]:
    return tuple(
        float(
            np.abs(left[..., :3].astype(np.int16) - right[..., :3].astype(np.int16)).mean()
        )
        for left, right in zip(before_patches, after_patches, strict=True)
    )


def _proposal(
    status: ObservationStatus,
    candidates: tuple[SequenceCandidateEvidence, ...],
    local: tuple[float, ...],
    reasons: tuple[str, ...],
) -> MoveSequenceProposal:
    return MoveSequenceProposal(
        status=status,
        moves=(),
        evidence_score=0.0,
        evidence=MoveSequenceEvidence(candidates, local, reasons, _FEATURE_VERSION),
    )
