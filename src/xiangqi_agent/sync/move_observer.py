from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import PieceTemplateBank


class ObservationStatus(StrEnum):
    ACCEPTED = "accepted"
    NO_CHANGE = "no_change"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    move: Move
    source_difference: float
    destination_difference: float
    unexpected_difference: float
    source_expected_distance: float
    destination_expected_distance: float
    semantic_margin: float
    source_semantic_confidence: float
    destination_semantic_confidence: float
    score: float


@dataclass(frozen=True, slots=True)
class MoveObservation:
    status: ObservationStatus
    move: Move | None
    after: BoardState | None
    confidence: float
    candidates: tuple[CandidateScore, ...]
    local_differences: tuple[float, ...]


class MoveObserver(Protocol):
    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveObservation: ...


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
        min_confidence: float = 0.8,
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
        )
        if any(not isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("observer thresholds must be finite and positive")
        if not isfinite(min_confidence) or not 0 < min_confidence <= 1:
            raise ValueError("min_confidence must be finite and between zero and one")
        self._patch_size = patch_size
        self._min_local_difference = min_local_difference
        self._max_unexpected_difference = max_unexpected_difference
        self._min_score = min_score
        self._min_margin = min_margin
        self._max_semantic_distance = max_semantic_distance
        self._min_semantic_margin = min_semantic_margin
        self._min_confidence = min_confidence

    def observe(
        self,
        board: BoardState,
        before: NDArray[np.generic],
        after: NDArray[np.generic],
        geometry: BoardGeometry,
    ) -> MoveObservation:
        before_patches = geometry.crop_intersections(before, size=self._patch_size)
        after_patches = geometry.crop_intersections(after, size=self._patch_size)
        local = _local_differences(before_patches, after_patches)
        if max(local, default=0.0) < self._min_local_difference:
            return MoveObservation(
                ObservationStatus.NO_CHANGE,
                None,
                None,
                0.0,
                (),
                local,
            )

        templates = PieceTemplateBank.from_position(
            board,
            geometry,
            before,
            patch_size=self._patch_size,
        )
        candidates = tuple(
            sorted(
                (
                    _score_candidate(board, move, local, after_patches, templates)
                    for move in legal_moves(board)
                ),
                key=lambda candidate: (-candidate.score, candidate.move.uci),
            )
        )
        if not candidates:
            return _ambiguous(candidates, local)

        best = candidates[0]
        next_score = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - next_score
        semantic_distance = max(
            best.source_expected_distance,
            best.destination_expected_distance,
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
        semantic_confidence = min(
            best.source_semantic_confidence,
            best.destination_semantic_confidence,
        )
        confidence = min(visual_confidence, semantic_confidence)
        supported = (
            best.source_difference >= self._min_local_difference
            and best.destination_difference >= self._min_local_difference
            and best.unexpected_difference <= self._max_unexpected_difference
            and best.score >= self._min_score
            and margin >= self._min_margin
            and semantic_distance <= self._max_semantic_distance
            and best.semantic_margin >= self._min_semantic_margin
            and confidence >= self._min_confidence
        )
        if not supported:
            return _ambiguous(candidates, local)

        return MoveObservation(
            ObservationStatus.ACCEPTED,
            best.move,
            apply_move(board, best.move),
            confidence,
            candidates,
            local,
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
) -> CandidateScore:
    source = local[move.from_index]
    destination = local[move.to_index]
    excluded = {move.from_index, move.to_index}
    unexpected = max(
        (difference for index, difference in enumerate(local) if index not in excluded),
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
    return CandidateScore(
        move=move,
        source_difference=source,
        destination_difference=destination,
        unexpected_difference=unexpected,
        source_expected_distance=source_match.distance,
        destination_expected_distance=destination_match.distance,
        semantic_margin=min(source_match.margin, destination_match.margin),
        source_semantic_confidence=source_match.confidence,
        destination_semantic_confidence=destination_match.confidence,
        score=min(source, destination) - unexpected,
    )


def _ambiguous(
    candidates: tuple[CandidateScore, ...], local: tuple[float, ...]
) -> MoveObservation:
    return MoveObservation(
        ObservationStatus.AMBIGUOUS,
        None,
        None,
        0.0,
        candidates,
        local,
    )
