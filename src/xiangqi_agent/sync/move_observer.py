from __future__ import annotations

from math import isfinite
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.sync.evidence import (
    CandidateEvidence,
    MoveEvidence,
    MoveProposal,
    ObservationStatus,
)
from xiangqi_agent.sync.semantic_gate import MoveSemanticGate, SemanticThresholds
from xiangqi_agent.vision.endpoint_features import (
    EndpointFeatureExtractor,
    EndpointFeatures,
    InstanceTransferExtractor,
)
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import PieceTemplateBank


class MoveObserver(Protocol):
    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveProposal: ...


class LegalMoveDiffObserver:
    """Accept a frame transition only when it uniquely supports one legal move."""

    def __init__(
        self,
        *,
        patch_size: int = 48,
        min_local_difference: float = 5.0,
        max_unexpected_difference: float = 3.0,
        min_score: float = 5.0,
        min_margin: float = 5.0,
        max_semantic_distance: float = 0.18,
        min_semantic_margin: float = 0.02,
        min_evidence_score: float = 0.8,
        max_instance_distance: float = 0.35,
        min_instance_evidence_score: float = 0.4,
        feature_extractor: EndpointFeatureExtractor | None = None,
    ) -> None:
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError("patch_size must be a positive integer")
        thresholds = (
            min_local_difference,
            max_unexpected_difference,
            min_score,
            min_margin,
            max_semantic_distance,
            min_semantic_margin,
            max_instance_distance,
        )
        if any(not isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("observer thresholds must be finite and positive")
        if not isfinite(min_evidence_score) or not 0 < min_evidence_score <= 1:
            raise ValueError("min_evidence_score must be finite and between zero and one")
        if (
            not isfinite(min_instance_evidence_score)
            or not 0 < min_instance_evidence_score <= 1
        ):
            raise ValueError(
                "min_instance_evidence_score must be finite and between zero and one"
            )
        self._patch_size = patch_size
        self._min_local_difference = min_local_difference
        self._max_unexpected_difference = max_unexpected_difference
        self._min_score = min_score
        self._min_margin = min_margin
        self._min_evidence_score = min_evidence_score
        self._max_semantic_distance = max_semantic_distance
        self._min_semantic_margin = min_semantic_margin
        self._feature_extractor = feature_extractor or InstanceTransferExtractor()
        self._semantic_gate = MoveSemanticGate(
            SemanticThresholds(
                max_source_empty_distance=max_semantic_distance,
                min_source_empty_evidence_score=min_evidence_score,
                max_destination_side_distance=max_semantic_distance,
                min_destination_side_evidence_score=min_evidence_score,
                max_instance_distance=max_instance_distance,
                min_instance_evidence_score=min_instance_evidence_score,
                min_semantic_margin=min_semantic_margin,
            )
        )

    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveProposal:
        before_patches = geometry.crop_intersections(before, size=self._patch_size)
        after_patches = geometry.crop_intersections(after, size=self._patch_size)
        local = _local_differences(before_patches, after_patches)
        if max(local, default=0.0) < self._min_local_difference:
            return MoveProposal(
                status=ObservationStatus.NO_CHANGE,
                move=None,
                evidence_score=0.0,
                evidence=MoveEvidence((), local, ()),
            )

        templates = PieceTemplateBank.from_position(
            board,
            geometry,
            before,
            patch_size=self._patch_size,
        )
        semantically_unchanged = _semantic_unchanged_mask(
            board,
            local,
            after_patches,
            templates,
            min_visual_difference=self._max_unexpected_difference,
            max_semantic_distance=self._max_semantic_distance,
            min_semantic_margin=self._min_semantic_margin,
            min_semantic_evidence_score=self._min_evidence_score,
        )
        candidates = tuple(
            sorted(
                (
                    _score_candidate(
                        board,
                        move,
                        local,
                        after_patches,
                        templates,
                        semantically_unchanged,
                    )
                    for move in legal_moves(board)
                ),
                key=lambda candidate: (-candidate.score, candidate.move.uci),
            )
        )
        if not candidates:
            return _ambiguous(candidates, local, ("no_legal_candidates",))

        best = candidates[0]
        next_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - next_score
        endpoint_features = self._feature_extractor.extract(
            _endpoint_crops(best.move, before_patches, after_patches)
        )
        visual_confidence = min(
            1.0,
            min(best.source_difference, best.destination_difference)
            / (
                min(best.source_difference, best.destination_difference)
                + best.unexpected_difference
                + 1.0
            ),
        )
        semantic_evidence_score = min(
            best.source_semantic_evidence_score,
            best.destination_semantic_evidence_score,
        )
        evidence_score = min(visual_confidence, semantic_evidence_score)
        rejection_reasons = list(
            _visual_rejection_reasons(
                best,
                margin=margin,
                min_local_difference=self._min_local_difference,
                max_unexpected_difference=self._max_unexpected_difference,
                min_score=self._min_score,
                min_margin=self._min_margin,
            )
        )
        rejection_reasons.extend(
            self._semantic_gate.evaluate(best, endpoint_features).rejection_reasons
        )
        evidence_score = min(evidence_score, endpoint_features.instance_evidence_score)
        if evidence_score < self._min_evidence_score:
            rejection_reasons.append("evidence_score")
        if rejection_reasons:
            return _ambiguous(
                candidates,
                local,
                tuple(dict.fromkeys(rejection_reasons)),
                endpoint_features,
            )

        return MoveProposal(
            status=ObservationStatus.ACCEPTED,
            move=best.move,
            evidence_score=evidence_score,
            evidence=MoveEvidence(candidates, local, (), endpoint_features),
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


def _score_candidate(
    board: BoardState,
    move: Move,
    local: tuple[float, ...],
    after_patches: tuple[NDArray[np.uint8], ...],
    templates: PieceTemplateBank,
    semantically_unchanged: tuple[bool, ...],
) -> CandidateEvidence:
    source = local[move.from_index]
    destination = local[move.to_index]
    excluded = {move.from_index, move.to_index}
    unexpected = max(
        (
            difference
            for index, difference in enumerate(local)
            if index not in excluded and not semantically_unchanged[index]
        ),
        default=0.0,
    )
    source_match = templates.match_any(frozenset({"."}), after_patches[move.from_index])
    moving_symbol = board.pieces[move.from_index]
    same_side_symbols = frozenset(
        symbol
        for symbol in templates.symbols
        if symbol != "." and symbol.isupper() == moving_symbol.isupper()
    )
    destination_match = templates.match_any(
        same_side_symbols,
        after_patches[move.to_index],
    )
    return CandidateEvidence(
        move=move,
        source_difference=source,
        destination_difference=destination,
        unexpected_difference=unexpected,
        source_expected_distance=source_match.distance,
        destination_expected_distance=destination_match.distance,
        semantic_margin=min(source_match.margin, destination_match.margin),
        source_semantic_evidence_score=source_match.confidence,
        destination_semantic_evidence_score=destination_match.confidence,
        score=min(source, destination) - unexpected,
    )


def _semantic_unchanged_mask(
    board: BoardState,
    local: tuple[float, ...],
    after_patches: tuple[NDArray[np.uint8], ...],
    templates: PieceTemplateBank,
    *,
    min_visual_difference: float,
    max_semantic_distance: float,
    min_semantic_margin: float,
    min_semantic_evidence_score: float,
) -> tuple[bool, ...]:
    unchanged: list[bool] = []
    for index, (difference, patch) in enumerate(zip(local, after_patches, strict=True)):
        if difference <= min_visual_difference:
            unchanged.append(True)
            continue
        expected_symbol = board.pieces[index]
        expected_symbols = (
            frozenset({"."})
            if expected_symbol == "."
            else frozenset(
                symbol
                for symbol in templates.symbols
                if symbol != "." and symbol.isupper() == expected_symbol.isupper()
            )
        )
        match = templates.match_any(expected_symbols, patch)
        unchanged.append(
            match.distance <= max_semantic_distance
            and match.margin >= min_semantic_margin
            and match.confidence >= min_semantic_evidence_score
        )
    return tuple(unchanged)


def _ambiguous(
    candidates: tuple[CandidateEvidence, ...],
    local: tuple[float, ...],
    rejection_reasons: tuple[str, ...],
    endpoint_features: EndpointFeatures | None = None,
) -> MoveProposal:
    return MoveProposal(
        status=ObservationStatus.AMBIGUOUS,
        move=None,
        evidence_score=0.0,
        evidence=MoveEvidence(
            candidates,
            local,
            rejection_reasons,
            endpoint_features,
        ),
    )


def _visual_rejection_reasons(
    best: CandidateEvidence,
    *,
    margin: float,
    min_local_difference: float,
    max_unexpected_difference: float,
    min_score: float,
    min_margin: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if best.source_difference < min_local_difference:
        reasons.append("source_change")
    if best.destination_difference < min_local_difference:
        reasons.append("destination_change")
    if best.unexpected_difference > max_unexpected_difference:
        reasons.append("outside_change")
    if best.score < min_score:
        reasons.append("candidate_score")
    if margin < min_margin:
        reasons.append("candidate_margin")
    return tuple(reasons)


def _endpoint_crops(
    move: Move,
    before_patches: tuple[NDArray[np.uint8], ...],
    after_patches: tuple[NDArray[np.uint8], ...],
) -> EndpointCrops:
    return EndpointCrops(
        source_before=_feature_patch(before_patches[move.from_index]),
        source_after=_feature_patch(after_patches[move.from_index]),
        target_before=_feature_patch(before_patches[move.to_index]),
        target_after=_feature_patch(after_patches[move.to_index]),
    )


def _feature_patch(patch: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if patch.shape == (48, 48, 4):
        return np.array(patch, dtype=np.uint8, copy=True)
    resized = cv2.resize(patch, (48, 48), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.uint8)
