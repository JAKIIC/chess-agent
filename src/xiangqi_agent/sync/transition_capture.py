from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.vision.geometry import BoardGeometry
from xiangqi_agent.vision.occupancy import OccupancyEvidence, OccupancyObserver

_CROP_SIZE = 48


@dataclass(frozen=True, slots=True)
class TransitionPointEvidence:
    point_index: int
    before: NDArray[np.uint8]
    after: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if (
            isinstance(self.point_index, bool)
            or not isinstance(self.point_index, int)
            or not 0 <= self.point_index < 90
        ):
            raise ValueError("point_index must be an integer from 0 through 89")
        object.__setattr__(self, "before", _owned_crop(self.before))
        object.__setattr__(self, "after", _owned_crop(self.after))


@dataclass(frozen=True, slots=True)
class TransitionCaptureEvidence:
    changed_points: tuple[int, ...]
    local_differences: tuple[float, ...]
    crops: tuple[TransitionPointEvidence, ...]
    decision_latency_ms: float
    before_occupancy: OccupancyEvidence | None = None
    after_occupancy: OccupancyEvidence | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.changed_points, tuple)
            or not 1 <= len(self.changed_points) <= 4
            or tuple(sorted(set(self.changed_points))) != self.changed_points
            or any(
                isinstance(point, bool)
                or not isinstance(point, int)
                or not 0 <= point < 90
                for point in self.changed_points
            )
        ):
            raise ValueError("changed_points must contain one through four stable board indices")
        if (
            not isinstance(self.local_differences, tuple)
            or len(self.local_differences) != 90
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
                for value in self.local_differences
            )
        ):
            raise ValueError("local_differences must contain 90 finite non-negative values")
        if not isinstance(self.crops, tuple) or any(
            not isinstance(crop, TransitionPointEvidence) for crop in self.crops
        ):
            raise TypeError("crops must be TransitionPointEvidence values")
        if tuple(crop.point_index for crop in self.crops) != self.changed_points:
            raise ValueError("transition crops must match changed_points in stable order")
        if (
            isinstance(self.decision_latency_ms, bool)
            or not isinstance(self.decision_latency_ms, (int, float))
            or not isfinite(self.decision_latency_ms)
            or self.decision_latency_ms < 0
        ):
            raise ValueError("decision_latency_ms must be finite and non-negative")
        if (self.before_occupancy is None) != (self.after_occupancy is None):
            raise ValueError("before_occupancy and after_occupancy must both be present or absent")
        if self.before_occupancy is not None and (
            not isinstance(self.before_occupancy, OccupancyEvidence)
            or not isinstance(self.after_occupancy, OccupancyEvidence)
        ):
            raise TypeError("occupancy snapshots must be OccupancyEvidence values")


def build_transition_capture_evidence(
    before: NDArray[np.uint8],
    after: NDArray[np.uint8],
    geometry: BoardGeometry,
    local_differences: tuple[float, ...],
    *,
    decision_latency_ms: float,
    occupancy_observer: OccupancyObserver | None = None,
) -> TransitionCaptureEvidence:
    differences = _validated_differences(local_differences)
    positive = tuple(index for index, value in enumerate(differences) if value > 0)
    ranked = sorted(
        positive or tuple(range(90)),
        key=lambda index: (-differences[index], index),
    )
    selected = tuple(sorted(ranked[:4]))
    before_patches = geometry.crop_selected_intersections(
        before,
        selected,
        size=_CROP_SIZE,
    )
    after_patches = geometry.crop_selected_intersections(
        after,
        selected,
        size=_CROP_SIZE,
    )
    crops = tuple(
        TransitionPointEvidence(
            point_index=index,
            before=before_patch,
            after=after_patch,
        )
        for index, before_patch, after_patch in zip(
            selected,
            before_patches,
            after_patches,
            strict=True,
        )
    )
    before_occupancy = (
        occupancy_observer.observe(before, geometry)
        if occupancy_observer is not None
        else None
    )
    after_occupancy = (
        occupancy_observer.observe(after, geometry)
        if occupancy_observer is not None
        else None
    )
    return TransitionCaptureEvidence(
        changed_points=selected,
        local_differences=differences,
        crops=crops,
        decision_latency_ms=decision_latency_ms,
        before_occupancy=before_occupancy,
        after_occupancy=after_occupancy,
    )


def _validated_differences(values: tuple[float, ...]) -> tuple[float, ...]:
    if (
        not isinstance(values, tuple)
        or len(values) != 90
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            for value in values
        )
    ):
        raise ValueError("local_differences must contain 90 finite non-negative values")
    return tuple(float(value) for value in values)


def _owned_crop(value: NDArray[np.generic]) -> NDArray[np.uint8]:
    pixels = np.asarray(value)
    if pixels.dtype != np.uint8 or pixels.shape != (_CROP_SIZE, _CROP_SIZE, 4):
        raise ValueError("transition crop must be a 48x48 BGRA uint8 image")
    owned = np.array(pixels, dtype=np.uint8, copy=True, order="C")
    owned.setflags(write=False)
    return owned
