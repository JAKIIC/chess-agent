from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import legal_moves
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
from xiangqi_agent.sync.two_ply_profile import (
    TWO_PLY_FEATURE_VERSION,
    TWO_PLY_INSTANCE_TRANSFER_MAX_SHIFT,
    TWO_PLY_MINIMUM_SEMANTIC_CONFIDENCE,
)
from xiangqi_agent.vision.endpoint_features import InstanceTransferExtractor
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import (
    PieceTemplateBankCache,
    TemplateClassification,
    TemplateMatch,
)

_FEATURE_VERSION = TWO_PLY_FEATURE_VERSION
_REPLY_CONSTRAINED_FEATURE_VERSION = _FEATURE_VERSION


class MoveSequenceObserver(Protocol):
    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveSequenceProposal: ...


@runtime_checkable
class ReplyConstrainedMoveSequenceObserver(Protocol):
    def observe_after_first(
        self,
        board: BoardState,
        first: Move,
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
        min_template_confidence: float = TWO_PLY_MINIMUM_SEMANTIC_CONFIDENCE,
        committer: StateCommitter | None = None,
        template_cache: PieceTemplateBankCache | None = None,
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
                profile_version="human-ai-two-ply-v2",
            )
        )
        self._committer = committer or RuleStateCommitter()
        if template_cache is not None and not isinstance(
            template_cache,
            PieceTemplateBankCache,
        ):
            raise TypeError("template_cache must be a PieceTemplateBankCache")
        self._template_cache = template_cache or PieceTemplateBankCache()

    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveSequenceProposal:
        return self._observe_projections(
            board,
            before,
            after,
            geometry,
            None,
            feature_version=_FEATURE_VERSION,
        )

    def observe_after_first(
        self,
        board: BoardState,
        first: Move,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveSequenceProposal:
        """Score only legal replies after an already observed first move."""
        return self._observe_projections(
            board,
            before,
            after,
            geometry,
            self._committer.project_replies(board, first),
            feature_version=_REPLY_CONSTRAINED_FEATURE_VERSION,
        )

    def _observe_projections(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
        projections: Iterable[tuple[tuple[Move, Move], BoardState]] | None,
        *,
        feature_version: str,
    ) -> MoveSequenceProposal:
        before_patches = geometry.crop_intersections(before, size=self._patch_size)
        after_patches = geometry.crop_intersections(after, size=self._patch_size)
        local = _local_differences(before_patches, after_patches)
        if max(local, default=0.0) < self._gate.profile.min_local_difference:
            return _proposal(
                ObservationStatus.NO_CHANGE,
                (),
                local,
                (),
                feature_version,
            )

        if projections is None:
            projections = self._project_from_changed_sources(board, local)

        templates = self._template_cache.get(
            board,
            geometry,
            before,
            patch_size=self._patch_size,
        )
        classification_cache: dict[int, TemplateClassification] = {}
        match_cache: dict[tuple[int, str], TemplateMatch] = {}
        confidence_components: dict[tuple[Move, Move], tuple[float, float]] = {}
        candidates: list[SequenceCandidateEvidence] = []
        template_unavailable = False
        for moves, final in projections:
            changed_points = tuple(
                index
                for index, (left, right) in enumerate(
                    zip(board.pieces, final.pieces, strict=True)
                )
                if left != right
            )
            if not changed_points:
                continue
            endpoint_matches: list[TemplateMatch] = []
            try:
                for index in changed_points:
                    symbol = final.pieces[index]
                    key = (index, symbol)
                    match = match_cache.get(key)
                    if match is None:
                        classification = classification_cache.get(index)
                        if classification is None:
                            classification = templates.classify(after_patches[index])
                            classification_cache[index] = classification
                        match = classification.match(symbol)
                        match_cache[key] = match
                    endpoint_matches.append(match)
            except ValueError:
                template_unavailable = True
                continue
            expected_floor = min(local[index] for index in changed_points)
            changed = frozenset(changed_points)
            artifact_points = tuple(
                index
                for index, difference in enumerate(local)
                if index not in changed
                and difference > self._gate.profile.max_unexpected_difference
            )
            semantic_artifacts: set[int] = set()
            artifact_matches: list[TemplateMatch] = []
            try:
                # A bounded outside difference can only be treated as UI
                # highlight/shadow when the final patch still matches the
                # unchanged board symbol's occupancy/side semantics.
                for index in artifact_points:
                    symbol = board.pieces[index]
                    expected_symbols = (
                        frozenset({"."})
                        if symbol == "."
                        else frozenset(
                            candidate
                            for candidate in templates.symbols
                            if candidate != "."
                            and candidate.isupper() == symbol.isupper()
                        )
                    )
                    key = (index, "".join(sorted(expected_symbols)))
                    match = match_cache.get(key)
                    if match is None:
                        classification = classification_cache.get(index)
                        if classification is None:
                            classification = templates.classify(after_patches[index])
                            classification_cache[index] = classification
                        match = classification.match_any(expected_symbols)
                        match_cache[key] = match
                    artifact_matches.append(match)
                    if (
                        match.distance <= self._gate.profile.max_template_distance
                        and match.confidence
                        >= self._gate.profile.min_template_confidence
                    ):
                        semantic_artifacts.add(index)
            except ValueError:
                template_unavailable = True
                continue
            unexpected = max(
                (
                    difference
                    for index, difference in enumerate(local)
                    if index not in changed and index not in semantic_artifacts
                ),
                default=0.0,
            )
            matches = (*endpoint_matches, *artifact_matches)
            confidence_components[moves] = (
                min(match.confidence for match in endpoint_matches),
                min(
                    (match.confidence for match in artifact_matches),
                    default=1.0,
                ),
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
        if (
            not decision.accepted
            and decision.rejection_reasons == ("template_confidence",)
            and ranked
        ):
            # Persistent source/destination markers can make an otherwise
            # correct empty endpoint visually close to an occupied template.
            # Only when every other hard gate already passed may the same
            # physical pieces' source-before -> target-after evidence satisfy
            # the unchanged semantic-confidence threshold.
            transfer_confidence = _piece_transfer_confidence(
                ranked[0].moves,
                before_patches,
                after_patches,
            )
            endpoint_confidence, artifact_confidence = confidence_components[
                ranked[0].moves
            ]
            effective_confidence = min(
                max(endpoint_confidence, transfer_confidence),
                artifact_confidence,
            )
            if effective_confidence > ranked[0].minimum_template_confidence:
                adjusted = replace(
                    ranked[0],
                    minimum_template_confidence=effective_confidence,
                )
                ranked = (adjusted, *ranked[1:])
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
                feature_version,
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
            evidence=MoveSequenceEvidence(ranked, local, (), feature_version),
        )

    def _project_from_changed_sources(
        self,
        board: BoardState,
        local: tuple[float, ...],
    ) -> Iterable[tuple[tuple[Move, Move], BoardState]]:
        """Skip chains that cannot satisfy the existing expected-change gate."""
        minimum = self._gate.profile.min_local_difference
        for first in legal_moves(board):
            # A legal first mover's source cannot contain its original piece after
            # the opponent replies. If that source did not visibly change, every
            # chain beginning with this move would fail expected_change anyway.
            if local[first.from_index] < minimum:
                continue
            yield from self._committer.project_replies(board, first)


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


def _piece_transfer_confidence(
    moves: tuple[Move, Move],
    before_patches: tuple[NDArray[np.uint8], ...],
    after_patches: tuple[NDArray[np.uint8], ...],
) -> float:
    first, second = moves
    surviving_moves = (
        (first, second)
        if second.to_index != first.to_index
        else (second,)
    )
    extractor = InstanceTransferExtractor(
        max_shift=TWO_PLY_INSTANCE_TRANSFER_MAX_SHIFT
    )
    scores = tuple(
        extractor.extract(
            EndpointCrops(
                source_before=_feature_patch(before_patches[move.from_index]),
                source_after=_feature_patch(after_patches[move.from_index]),
                target_before=_feature_patch(before_patches[move.to_index]),
                target_after=_feature_patch(after_patches[move.to_index]),
            )
        ).instance_evidence_score
        for move in surviving_moves
    )
    return min(scores)


def _feature_patch(patch: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if patch.shape == (48, 48, 4):
        return np.array(patch, dtype=np.uint8, copy=True)
    resized = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _proposal(
    status: ObservationStatus,
    candidates: tuple[SequenceCandidateEvidence, ...],
    local: tuple[float, ...],
    reasons: tuple[str, ...],
    feature_version: str,
) -> MoveSequenceProposal:
    return MoveSequenceProposal(
        status=status,
        moves=(),
        evidence_score=0.0,
        evidence=MoveSequenceEvidence(candidates, local, reasons, feature_version),
    )
