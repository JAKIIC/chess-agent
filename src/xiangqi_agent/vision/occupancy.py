from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, runtime_checkable

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.templates import PieceTemplateBank

_PATCH_SIZE = 48
_OCCUPIED_THRESHOLD = 0.52
_CONFIDENCE_WIDTH = 0.30
_KNOWN_POSITION_PATCH_RATIO = 0.78
_MINIMUM_TEMPLATE_SEPARATION = 0.004
_KNOWN_POSITION_ALGORITHM = "known-position-template-occupancy-v1"


@dataclass(frozen=True, slots=True)
class OccupancyEvidence:
    occupied: tuple[bool, ...]
    confidences: tuple[float, ...]
    algorithm_version: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.occupied, tuple)
            or len(self.occupied) != 90
            or any(not isinstance(value, bool) for value in self.occupied)
        ):
            raise ValueError("occupied must contain exactly 90 booleans")
        if (
            not isinstance(self.confidences, tuple)
            or len(self.confidences) != 90
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                for value in self.confidences
            )
        ):
            raise ValueError("confidences must contain exactly 90 finite confidences")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.confidences):
            raise ValueError("occupancy confidences must be between zero and one")
        if not isinstance(self.algorithm_version, str) or not self.algorithm_version.strip():
            raise ValueError("algorithm_version must be a non-empty string")
        object.__setattr__(self, "confidences", tuple(float(value) for value in self.confidences))


@dataclass(frozen=True, slots=True)
class OccupancyComparison:
    mismatched_points: tuple[int, ...]
    low_confidence_points: tuple[int, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("mismatched_points", self.mismatched_points),
            ("low_confidence_points", self.low_confidence_points),
        ):
            if (
                not isinstance(values, tuple)
                or tuple(sorted(set(values))) != values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < 90
                    for value in values
                )
            ):
                raise ValueError(f"{name} must contain stable board indices")

    @property
    def accepted(self) -> bool:
        return not self.mismatched_points and not self.low_confidence_points


class OccupancyObserver(Protocol):
    def observe(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
    ) -> OccupancyEvidence: ...


@runtime_checkable
class IncrementalOccupancyObserver(OccupancyObserver, Protocol):
    def observe_changed(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
        baseline: OccupancyEvidence,
        point_indices: tuple[int, ...],
    ) -> OccupancyEvidence: ...


class CircularOccupancyObserver:
    algorithm_version = "circular-occupancy-v1"

    def observe(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        if not isinstance(geometry, BoardGeometry):
            raise TypeError("geometry must be a BoardGeometry")
        patches = geometry.crop_intersections(frame, size=_PATCH_SIZE)
        scores = tuple(_occupancy_score(patch) for patch in patches)
        occupied = tuple(score >= _OCCUPIED_THRESHOLD for score in scores)
        confidences = tuple(_score_confidence(score) for score in scores)
        return OccupancyEvidence(occupied, confidences, self.algorithm_version)


class KnownPositionOccupancyObserver:
    """Calibrate fixed-theme occupancy from one user-confirmed position.

    Only compact template features are retained. If the initial frame does not
    clearly separate occupied intersections from empty ones, calibration stays
    unset and every point is reported with zero confidence.
    """

    algorithm_version = _KNOWN_POSITION_ALGORITHM

    def __init__(self, board: BoardState) -> None:
        if not isinstance(board, BoardState):
            raise TypeError("board must be a BoardState")
        if "." not in board.pieces or all(piece == "." for piece in board.pieces):
            raise ValueError("board must contain both empty and occupied intersections")
        self._board = board
        self._bank: PieceTemplateBank | None = None
        self._patch_size: int | None = None

    def observe(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        if not isinstance(geometry, BoardGeometry):
            raise TypeError("geometry must be a BoardGeometry")
        patch_size = self._patch_size or _adaptive_patch_size(geometry)
        bank = self._bank or PieceTemplateBank.from_position(
            self._board,
            geometry,
            frame,
            patch_size=patch_size,
        )
        patches = geometry.crop_intersections(frame, size=patch_size)
        evidence, separations = _classify_template_occupancy(bank, patches)

        if self._bank is None:
            expected = tuple(piece != "." for piece in self._board.pieces)
            separated = min(separations) >= _MINIMUM_TEMPLATE_SEPARATION
            if evidence.occupied != expected or not separated:
                return OccupancyEvidence(
                    expected,
                    (0.0,) * 90,
                    self.algorithm_version,
                )
            self._bank = bank
            self._patch_size = patch_size
        return evidence

    def observe_changed(
        self,
        frame: NDArray[np.uint8],
        geometry: BoardGeometry,
        baseline: OccupancyEvidence,
        point_indices: tuple[int, ...],
    ) -> OccupancyEvidence:
        if not isinstance(geometry, BoardGeometry):
            raise TypeError("geometry must be a BoardGeometry")
        if not isinstance(baseline, OccupancyEvidence):
            raise TypeError("baseline must be OccupancyEvidence")
        if baseline.algorithm_version != self.algorithm_version:
            raise ValueError("baseline occupancy algorithm does not match observer")
        if (
            not isinstance(point_indices, tuple)
            or not point_indices
            or tuple(sorted(set(point_indices))) != point_indices
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < 90
                for index in point_indices
            )
        ):
            raise ValueError("point_indices must contain unique sorted board indices")
        bank = self._bank
        patch_size = self._patch_size
        if bank is None or patch_size is None:
            raise RuntimeError("known-position occupancy must be calibrated first")
        patches = geometry.crop_selected_intersections(
            frame,
            point_indices,
            size=patch_size,
        )
        occupied = list(baseline.occupied)
        confidences = list(baseline.confidences)
        for index, patch in zip(point_indices, patches, strict=True):
            empty_distance, occupied_distance = bank.occupancy_distances(patch)
            separation = abs(empty_distance - occupied_distance)
            occupied[index] = occupied_distance < empty_distance
            confidences[index] = _clamp(
                separation / (empty_distance + occupied_distance + 1e-6)
            )
        return OccupancyEvidence(
            tuple(occupied),
            tuple(confidences),
            self.algorithm_version,
        )


def compare_occupancy(
    evidence: OccupancyEvidence,
    board: BoardState,
    *,
    minimum_confidence: float,
) -> OccupancyComparison:
    if not isinstance(evidence, OccupancyEvidence):
        raise TypeError("evidence must be OccupancyEvidence")
    if not isinstance(board, BoardState):
        raise TypeError("board must be a BoardState")
    if isinstance(minimum_confidence, bool) or not isinstance(
        minimum_confidence, (int, float)
    ):
        raise TypeError("minimum_confidence must be a number")
    threshold = float(minimum_confidence)
    if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("minimum_confidence must be between zero and one")

    expected = tuple(piece != "." for piece in board.pieces)
    low_confidence = tuple(
        index
        for index, confidence in enumerate(evidence.confidences)
        if confidence < threshold
    )
    low_set = frozenset(low_confidence)
    mismatched = tuple(
        index
        for index, (observed, wanted) in enumerate(
            zip(evidence.occupied, expected, strict=True)
        )
        if index not in low_set and observed is not wanted
    )
    return OccupancyComparison(mismatched, low_confidence)


def _occupancy_score(patch: NDArray[np.uint8]) -> float:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGRA2GRAY)
    luminance = cv2.cvtColor(patch[..., :3], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    coordinates = np.indices((_PATCH_SIZE, _PATCH_SIZE), dtype=np.float32)
    center = (_PATCH_SIZE - 1) / 2.0
    dy = coordinates[0] - center
    dx = coordinates[1] - center
    radius = np.hypot(dx, dy)
    annulus = (radius >= 14.0) & (radius <= 22.0)

    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(np.mean(edges[annulus] > 0))
    annulus_score = _clamp((edge_density - 0.035) / 0.20)

    center_disk = radius <= 9.0
    corners = radius >= 27.0
    center_mean = float(np.mean(luminance[center_disk]))
    corner_mean = float(np.mean(luminance[corners]))
    contrast_score = _clamp(abs(center_mean - corner_mean) / 55.0)

    gradient_x = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    safe_radius = np.maximum(radius, 1.0)
    radial_component = np.abs((gradient_x * dx + gradient_y * dy) / safe_radius)
    radial_strength = float(np.mean(radial_component[annulus]))
    radial_score = _clamp((radial_strength - 6.0) / 80.0)

    return 0.50 * annulus_score + 0.30 * contrast_score + 0.20 * radial_score


def _score_confidence(score: float) -> float:
    return _clamp(abs(score - _OCCUPIED_THRESHOLD) / _CONFIDENCE_WIDTH)


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _adaptive_patch_size(geometry: BoardGeometry) -> int:
    points = geometry.grid_points()
    horizontal = float(np.hypot(
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
    ))
    vertical = float(np.hypot(
        points[9][0] - points[0][0],
        points[9][1] - points[0][1],
    ))
    return max(8, round(min(horizontal, vertical) * _KNOWN_POSITION_PATCH_RATIO))


def _classify_template_occupancy(
    bank: PieceTemplateBank,
    patches: tuple[NDArray[np.uint8], ...],
) -> tuple[OccupancyEvidence, tuple[float, ...]]:
    distances = tuple(bank.occupancy_distances(patch) for patch in patches)
    occupied = tuple(
        occupied_distance < empty_distance
        for empty_distance, occupied_distance in distances
    )
    separations = tuple(
        abs(empty_distance - occupied_distance)
        for empty_distance, occupied_distance in distances
    )
    confidences = tuple(
        _clamp(separation / (empty_distance + occupied_distance + 1e-6))
        for (empty_distance, occupied_distance), separation in zip(
            distances,
            separations,
            strict=True,
        )
    )
    return (
        OccupancyEvidence(occupied, confidences, _KNOWN_POSITION_ALGORITHM),
        separations,
    )
