from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import VALID_PIECES, BoardState
from xiangqi_agent.vision.geometry import BoardGeometry


class TemplateExtractionError(ValueError):
    """The confirmed position cannot seed a complete fixed-theme template bank."""


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    expected_symbol: str
    distance: float
    margin: float


@dataclass(frozen=True, slots=True)
class PieceTemplateBank:
    """In-memory visual examples extracted from one confirmed fixed-theme board."""

    _examples: dict[str, tuple[NDArray[np.float32], ...]]

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

        grouped: dict[str, list[NDArray[np.float32]]] = {symbol: [] for symbol in sorted(present)}
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
        if candidate.shape != examples[0].shape:
            raise ValueError("template patch shape differs from the extracted theme")
        return _minimum_distance(examples, candidate)

    def match(self, expected_symbol: str, patch: NDArray[np.generic]) -> TemplateMatch:
        expected = _validate_symbol(expected_symbol, self._examples)
        candidate = _feature(patch)
        distances = {
            symbol: _minimum_distance(examples, candidate)
            for symbol, examples in self._examples.items()
        }
        expected_distance = distances[expected]
        alternatives = tuple(
            distance for symbol, distance in distances.items() if symbol != expected
        )
        margin = min(alternatives) - expected_distance if alternatives else float("inf")
        return TemplateMatch(expected, expected_distance, margin)


def _validate_symbol(symbol: str, examples: Mapping[str, object]) -> str:
    if symbol not in examples:
        raise ValueError("unknown Xiangqi template symbol")
    return symbol


def _feature(patch: NDArray[np.generic]) -> NDArray[np.float32]:
    pixels = np.asarray(patch)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("template patch must be a BGRA uint8 image")
    return np.asarray(pixels[..., :3], dtype=np.float32) / np.float32(255.0)


def _minimum_distance(
    examples: tuple[NDArray[np.float32], ...], candidate: NDArray[np.float32]
) -> float:
    if candidate.shape != examples[0].shape:
        raise ValueError("template patch shape differs from the extracted theme")
    return min(float(np.abs(example - candidate).mean()) for example in examples)
