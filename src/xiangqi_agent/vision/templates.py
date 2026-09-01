from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import VALID_PIECES, BoardState
from xiangqi_agent.vision.geometry import BoardGeometry

_REDNESS_DISTANCE_WEIGHT = 0.1


class TemplateExtractionError(ValueError):
    """The confirmed position cannot seed a complete fixed-theme template bank."""


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    expected_symbol: str
    distance: float
    margin: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PieceTemplateBank:
    """In-memory visual examples extracted from one confirmed fixed-theme board."""

    _examples: dict[str, tuple[_TemplateFeature, ...]]

    @classmethod
    def from_position(
        cls,
        board: BoardState,
        geometry: BoardGeometry,
        frame: NDArray[np.generic],
        *,
        patch_size: int = 48,
        require_complete: bool = False,
    ) -> PieceTemplateBank:
        patches = geometry.crop_intersections(frame, size=patch_size)
        present = frozenset(board.pieces)
        if require_complete and present != VALID_PIECES:
            raise TemplateExtractionError(
                "the confirmed position must contain all 15 fixed-theme classes"
            )

        grouped: dict[str, list[_TemplateFeature]] = {
            symbol: [] for symbol in sorted(present)
        }
        for symbol, patch in zip(board.pieces, patches, strict=True):
            grouped[symbol].append(_feature(patch))
        return cls({symbol: tuple(examples) for symbol, examples in grouped.items()})

    @property
    def symbols(self) -> frozenset[str]:
        return frozenset(self._examples)

    def example_count(self, symbol: str) -> int:
        return len(self._examples[_validate_symbol(symbol, self._examples)])

    def distance(self, symbol: str, patch: NDArray[np.generic]) -> float:
        examples = self._examples[_validate_symbol(symbol, self._examples)]
        candidate = _feature(patch)
        return _minimum_distance(examples, candidate)

    def occupancy_distances(
        self,
        patch: NDArray[np.generic],
    ) -> tuple[float, float]:
        """Return the closest empty and occupied distances from one feature pass."""
        if "." not in self._examples:
            raise ValueError("template bank has no empty examples")
        occupied_examples = tuple(
            example
            for symbol, examples in self._examples.items()
            if symbol != "."
            for example in examples
        )
        if not occupied_examples:
            raise ValueError("template bank has no occupied examples")
        candidate = _feature(patch)
        return (
            _minimum_distance(self._examples["."], candidate),
            _minimum_distance(occupied_examples, candidate),
        )

    def match(self, expected_symbol: str, patch: NDArray[np.generic]) -> TemplateMatch:
        expected = _validate_symbol(expected_symbol, self._examples)
        return self.match_any(frozenset({expected}), patch)

    def match_any(
        self,
        expected_symbols: frozenset[str],
        patch: NDArray[np.generic],
    ) -> TemplateMatch:
        if not expected_symbols:
            raise ValueError("expected template group must not be empty")
        unknown = expected_symbols - self.symbols
        if unknown:
            raise ValueError("expected template group contains an unknown symbol")
        expected_groups = {_semantic_group(symbol) for symbol in expected_symbols}
        if len(expected_groups) != 1:
            raise ValueError("expected templates must belong to one semantic group")
        candidate = _feature(patch)
        distances = {
            symbol: _minimum_distance(examples, candidate)
            for symbol, examples in self._examples.items()
        }
        expected, expected_distance = min(
            ((symbol, distances[symbol]) for symbol in expected_symbols),
            key=lambda item: (item[1], item[0]),
        )
        alternatives = tuple(
            distance for symbol, distance in distances.items() if symbol not in expected_symbols
        )
        margin = min(alternatives) - expected_distance if alternatives else float("inf")
        group_distances = {
            group: min(
                distance
                for symbol, distance in distances.items()
                if _semantic_group(symbol) == group
            )
            for group in {_semantic_group(symbol) for symbol in distances}
        }
        floor = min(group_distances.values())
        group_weights = {
            group: exp(-(distance - floor) / 0.008)
            for group, distance in group_distances.items()
        }
        expected_group = expected_groups.pop()
        confidence = group_weights[expected_group] / sum(group_weights.values())
        return TemplateMatch(expected, expected_distance, margin, confidence)


def _validate_symbol(symbol: str, examples: Mapping[str, object]) -> str:
    if symbol not in examples:
        raise ValueError("unknown Xiangqi template symbol")
    return symbol


def _semantic_group(symbol: str) -> str:
    if symbol == ".":
        return "empty"
    return "red" if symbol.isupper() else "black"


@dataclass(frozen=True, slots=True)
class _TemplateFeature:
    ordered_colors: NDArray[np.float32]
    redness: float


def _feature(patch: NDArray[np.generic]) -> _TemplateFeature:
    pixels = np.asarray(patch)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("template patch must be a BGRA uint8 image")
    colors = np.asarray(pixels[..., :3], dtype=np.float32).reshape(-1, 3)
    order = np.lexsort((colors[:, 2], colors[:, 1], colors[:, 0]))
    ordered_colors = colors[order].reshape(-1)
    red_excess = np.maximum(
        colors[:, 2] - np.maximum(colors[:, 0], colors[:, 1]),
        np.float32(0.0),
    )
    sample_count = max(1, red_excess.size // 10)
    strongest_red = np.partition(red_excess, red_excess.size - sample_count)[-sample_count:]
    return _TemplateFeature(
        ordered_colors=np.asarray(
            ordered_colors / np.float32(255.0),
            dtype=np.float32,
        ),
        redness=float(strongest_red.mean() / np.float32(255.0)),
    )


def _minimum_distance(
    examples: tuple[_TemplateFeature, ...], candidate: _TemplateFeature
) -> float:
    if candidate.ordered_colors.shape != examples[0].ordered_colors.shape:
        raise ValueError("template feature shape differs from the extracted theme")
    return min(
        float(np.abs(example.ordered_colors - candidate.ordered_colors).mean())
        + _REDNESS_DISTANCE_WEIGHT * abs(example.redness - candidate.redness)
        for example in examples
    )
