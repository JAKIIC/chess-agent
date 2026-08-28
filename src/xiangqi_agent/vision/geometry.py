from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import Orientation


class GeometryError(ValueError):
    """Manual board geometry is invalid for the supplied frame."""


type Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class NormalizedQuad:
    points: tuple[Point, Point, Point, Point]

    def __post_init__(self) -> None:
        if not isinstance(self.points, tuple) or len(self.points) != 4:
            raise GeometryError("quad must contain four points")
        converted: list[Point] = []
        for point in self.points:
            if not isinstance(point, tuple) or len(point) != 2:
                raise GeometryError("quad points must be x,y pairs")
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool) or not isfinite(x) or not isfinite(y):
                raise GeometryError("quad coordinates must be finite numbers")
            if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                raise GeometryError("normalized quad coordinates must be between zero and one")
            converted.append((float(x), float(y)))
        object.__setattr__(self, "points", tuple(converted))
        if not _is_ordered_convex(tuple(converted)) or _polygon_area(tuple(converted)) < 0.01:
            raise GeometryError("quad must be a non-trivial convex TL,TR,BR,BL shape")

    @classmethod
    def from_pixels(
        cls,
        points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]],
        frame_size: tuple[int, int],
    ) -> NormalizedQuad:
        width, height = _validate_frame_size(frame_size)
        normalized = tuple((x / (width - 1), y / (height - 1)) for x, y in points)
        return cls(cast(tuple[Point, Point, Point, Point], normalized))


def parse_normalized_quad(text: str) -> NormalizedQuad:
    try:
        pairs = tuple(
            tuple(float(value.strip()) for value in pair.split(",", maxsplit=1))
            for pair in text.split(";")
        )
    except ValueError as exc:
        raise GeometryError("quad must contain four x,y pairs") from exc
    if len(pairs) != 4 or any(len(pair) != 2 for pair in pairs):
        raise GeometryError("quad must contain four x,y pairs")
    points = tuple((pair[0], pair[1]) for pair in pairs)
    return NormalizedQuad(cast(tuple[Point, Point, Point, Point], points))


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    quad: NormalizedQuad
    frame_size: tuple[int, int]
    orientation: Orientation = Orientation.RED_BOTTOM

    def __post_init__(self) -> None:
        _validate_frame_size(self.frame_size)
        if not isinstance(self.orientation, Orientation):
            raise GeometryError("orientation must be an Orientation")

    @classmethod
    def from_quad(
        cls,
        quad: NormalizedQuad,
        frame_size: tuple[int, int],
        orientation: Orientation = Orientation.RED_BOTTOM,
    ) -> BoardGeometry:
        return cls(quad=quad, frame_size=frame_size, orientation=orientation)

    def grid_points(self) -> tuple[Point, ...]:
        source = np.array(((0, 0), (8, 0), (8, 9), (0, 9)), dtype=np.float32)
        target = np.array(self._pixel_corners(), dtype=np.float32)
        transform = cv2.getPerspectiveTransform(source, target)
        logical = []
        for row in range(10):
            for column in range(9):
                if self.orientation is Orientation.RED_BOTTOM:
                    logical.append((column, row))
                else:
                    logical.append((8 - column, 9 - row))
        mapped = cv2.perspectiveTransform(np.array([logical], dtype=np.float32), transform)[0]
        return tuple((float(point[0]), float(point[1])) for point in mapped)

    def crop_intersections(
        self,
        frame: NDArray[np.generic],
        size: int = 48,
    ) -> tuple[NDArray[np.uint8], ...]:
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise GeometryError("crop size must be a positive integer")
        pixels = np.asarray(frame)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise GeometryError("frame must be a BGRA uint8 image")
        if (int(pixels.shape[1]), int(pixels.shape[0])) != self.frame_size:
            raise GeometryError("frame size changed after calibration")
        half = size / 2.0
        destination = np.array(
            (
                (half, half),
                (half + 8 * size, half),
                (half + 8 * size, half + 9 * size),
                (half, half + 9 * size),
            ),
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(np.array(self._pixel_corners(), dtype=np.float32), destination)
        warped: NDArray[np.uint8] = np.asarray(
            cv2.warpPerspective(
                pixels,
                transform,
                (9 * size, 10 * size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ),
            dtype=np.uint8,
        )
        patches: tuple[NDArray[np.uint8], ...] = tuple(
            warped[row * size : (row + 1) * size, column * size : (column + 1) * size].copy()
            for row in range(10)
            for column in range(9)
        )
        return patches if self.orientation is Orientation.RED_BOTTOM else tuple(reversed(patches))

    def _pixel_corners(self) -> tuple[Point, Point, Point, Point]:
        width, height = self.frame_size
        corners = tuple((x * (width - 1), y * (height - 1)) for x, y in self.quad.points)
        return cast(tuple[Point, Point, Point, Point], corners)


def _validate_frame_size(frame_size: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(frame_size, tuple)
        or len(frame_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 1 for value in frame_size)
    ):
        raise GeometryError("frame size must contain width and height greater than one")
    return frame_size


def _polygon_area(points: tuple[Point, ...]) -> float:
    return abs(
        sum(
            left[0] * right[1] - right[0] * left[1]
            for left, right in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2.0


def _is_ordered_convex(points: tuple[Point, ...]) -> bool:
    top_left, top_right, bottom_right, bottom_left = points
    if not (
        top_left[0] < top_right[0]
        and bottom_left[0] < bottom_right[0]
        and top_left[1] < bottom_left[1]
        and top_right[1] < bottom_right[1]
    ):
        return False
    crosses = []
    for first, second, third in zip(points, points[1:] + points[:1], points[2:] + points[:2], strict=True):
        crosses.append(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )
    return all(value > 0 for value in crosses) or all(value < 0 for value in crosses)
