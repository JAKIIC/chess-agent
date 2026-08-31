from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import PieceTemplateBank


@dataclass(frozen=True, slots=True)
class PositionValidationResult:
    accepted: bool
    matched_count: int
    mismatched_indices: tuple[int, ...]


def validate_fixed_theme_position(
    board: BoardState,
    reference_frame: NDArray[np.generic],
    reference_geometry: BoardGeometry,
    observed_frame: NDArray[np.generic],
    observed_geometry: BoardGeometry,
    *,
    patch_size: int = 48,
    max_distance: float = 0.18,
    min_margin: float = 0.02,
    min_evidence_score: float = 0.8,
) -> PositionValidationResult:
    """Verify that a resized fixed-theme board retains every occupied side."""
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if not isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be finite and positive")
    if not isfinite(min_margin) or min_margin <= 0:
        raise ValueError("min_margin must be finite and positive")
    if not isfinite(min_evidence_score) or not 0 < min_evidence_score <= 1:
        raise ValueError("min_evidence_score must be finite and between zero and one")

    templates = PieceTemplateBank.from_position(
        board,
        reference_geometry,
        reference_frame,
        patch_size=patch_size,
    )
    observed_patches = observed_geometry.crop_intersections(
        observed_frame,
        size=patch_size,
    )
    mismatches: list[int] = []
    for index, (symbol, patch) in enumerate(
        zip(board.pieces, observed_patches, strict=True)
    ):
        expected = _semantic_templates(symbol, templates)
        match = templates.match_any(expected, patch)
        if (
            match.distance > max_distance
            or match.margin < min_margin
            or match.confidence < min_evidence_score
        ):
            mismatches.append(index)
    mismatch_indices = tuple(mismatches)
    return PositionValidationResult(
        accepted=not mismatch_indices,
        matched_count=90 - len(mismatch_indices),
        mismatched_indices=mismatch_indices,
    )


def _semantic_templates(
    symbol: str,
    templates: PieceTemplateBank,
) -> frozenset[str]:
    if symbol == ".":
        return frozenset({"."})
    return frozenset(
        candidate
        for candidate in templates.symbols
        if candidate != "." and candidate.isupper() == symbol.isupper()
    )
