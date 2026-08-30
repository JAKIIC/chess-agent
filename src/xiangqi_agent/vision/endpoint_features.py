from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops

_SIZE = 48


@dataclass(frozen=True, slots=True)
class EndpointFeatures:
    feature_version: str
    instance_distance: float
    instance_evidence_score: float
    color_distance: float
    gradient_distance: float
    source_change_distance: float
    target_change_distance: float
    best_shift: tuple[int, int]


class EndpointFeatureExtractor(Protocol):
    version: str

    def extract(self, crops: EndpointCrops) -> EndpointFeatures: ...


class RgbBaselineExtractor:
    version = "rgb-v1"

    def extract(self, crops: EndpointCrops) -> EndpointFeatures:
        _validate_crops(crops)
        instance = _rgb_distance(crops.source_before, crops.target_after)
        return _features(
            version=self.version,
            instance=instance,
            color=instance,
            gradient=0.0,
            source_change=_rgb_distance(crops.source_before, crops.source_after),
            target_change=_rgb_distance(crops.target_before, crops.target_after),
            best_shift=(0, 0),
        )


class MaskedLabExtractor:
    version = "masked-lab-v1"

    def extract(self, crops: EndpointCrops) -> EndpointFeatures:
        _validate_crops(crops)
        mask = _soft_circle_mask()
        source = _lab_features(crops.source_before, mask)
        target = _lab_features(crops.target_after, mask)
        color = _weighted_distance(source, target, mask)
        return _features(
            version=self.version,
            instance=color,
            color=color,
            gradient=0.0,
            source_change=_weighted_distance(
                source,
                _lab_features(crops.source_after, mask),
                mask,
            ),
            target_change=_weighted_distance(
                _lab_features(crops.target_before, mask),
                target,
                mask,
            ),
            best_shift=(0, 0),
        )


class AlignedGradientExtractor:
    version = "aligned-gradient-v1"

    def __init__(self, *, max_shift: int = 3) -> None:
        _validate_max_shift(max_shift)
        self._max_shift = max_shift

    def extract(self, crops: EndpointCrops) -> EndpointFeatures:
        _validate_crops(crops)
        return _aligned_features(crops, version=self.version, max_shift=self._max_shift)


class InstanceTransferExtractor:
    version = "instance-transfer-v1"

    def __init__(self, *, max_shift: int = 3) -> None:
        _validate_max_shift(max_shift)
        self._max_shift = max_shift

    def extract(self, crops: EndpointCrops) -> EndpointFeatures:
        _validate_crops(crops)
        return _aligned_features(crops, version=self.version, max_shift=self._max_shift)


def _aligned_features(
    crops: EndpointCrops,
    *,
    version: str,
    max_shift: int,
) -> EndpointFeatures:
    mask = _soft_circle_mask()
    source_lab = _lab_features(crops.source_before, mask)
    source_gradient = _gradient_features(crops.source_before)
    best: tuple[float, int, int, int, float, float] | None = None
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted_target = _shift(crops.target_after, dx, dy)
            color = _weighted_distance(source_lab, _lab_features(shifted_target, mask), mask)
            gradient = _weighted_distance(
                source_gradient,
                _gradient_features(shifted_target),
                mask,
            )
            distance = 0.55 * color + 0.45 * gradient
            candidate = (distance, abs(dx) + abs(dy), dy, dx, color, gradient)
            if best is None or candidate[:4] < best[:4]:
                best = candidate
    if best is None:
        raise RuntimeError("alignment search produced no candidate")
    distance, _, dy, dx, color, gradient = best
    return _features(
        version=version,
        instance=distance,
        color=color,
        gradient=gradient,
        source_change=_weighted_distance(
            source_lab,
            _lab_features(crops.source_after, mask),
            mask,
        ),
        target_change=_weighted_distance(
            _lab_features(crops.target_before, mask),
            _lab_features(crops.target_after, mask),
            mask,
        ),
        best_shift=(dx, dy),
    )


def _features(
    *,
    version: str,
    instance: float,
    color: float,
    gradient: float,
    source_change: float,
    target_change: float,
    best_shift: tuple[int, int],
) -> EndpointFeatures:
    stable_instance = _stable_float(instance)
    evidence_score = 1.0 / (1.0 + 4.0 * max(0.0, stable_instance))
    return EndpointFeatures(
        feature_version=version,
        instance_distance=stable_instance,
        instance_evidence_score=_stable_float(evidence_score),
        color_distance=_stable_float(color),
        gradient_distance=_stable_float(gradient),
        source_change_distance=_stable_float(source_change),
        target_change_distance=_stable_float(target_change),
        best_shift=best_shift,
    )


def _stable_float(value: float) -> float:
    """Remove sub-nanoscopic OpenCV SIMD reduction jitter from replay output."""
    return round(float(value), 8)


def _validate_crops(crops: EndpointCrops) -> None:
    if not isinstance(crops, EndpointCrops):
        raise TypeError("crops must be EndpointCrops")
    for name in ("source_before", "source_after", "target_before", "target_after"):
        pixels = np.asarray(getattr(crops, name))
        if pixels.dtype != np.uint8 or pixels.shape != (_SIZE, _SIZE, 4):
            raise ValueError("endpoint feature crops must be 48x48 BGRA uint8 images")


def _validate_max_shift(max_shift: int) -> None:
    if isinstance(max_shift, bool) or not isinstance(max_shift, int) or not 0 <= max_shift <= 8:
        raise ValueError("max_shift must be an integer from zero through eight")


def _rgb_distance(left: NDArray[np.uint8], right: NDArray[np.uint8]) -> float:
    left_rgb = np.asarray(left[..., :3], dtype=np.float32) / np.float32(255.0)
    right_rgb = np.asarray(right[..., :3], dtype=np.float32) / np.float32(255.0)
    return float(np.abs(left_rgb - right_rgb).mean())


def _soft_circle_mask() -> NDArray[np.float32]:
    axis = np.arange(_SIZE, dtype=np.float32) - np.float32((_SIZE - 1) / 2)
    x, y = np.meshgrid(axis, axis)
    radius = np.sqrt(x * x + y * y)
    return np.asarray(np.clip((18.0 - radius) / 3.0, 0.0, 1.0), dtype=np.float32)


def _lab_features(
    patch: NDArray[np.uint8],
    mask: NDArray[np.float32],
) -> NDArray[np.float32]:
    lab = cv2.cvtColor(patch[..., :3], cv2.COLOR_BGR2LAB).astype(np.float32)
    luminance = lab[..., 0] / np.float32(255.0)
    weight = float(mask.sum())
    mean = float((luminance * mask).sum() / weight)
    variance = float((((luminance - mean) ** 2) * mask).sum() / weight)
    scale = max(variance**0.5, 0.08)
    normalized_luminance = np.clip((luminance - mean) / scale, -3.0, 3.0) / 6.0
    chroma = (lab[..., 1:] - np.float32(128.0)) / np.float32(127.0)
    return np.concatenate((normalized_luminance[..., None], chroma), axis=2)


def _gradient_features(patch: NDArray[np.uint8]) -> NDArray[np.float32]:
    gray = cv2.cvtColor(patch[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(x, y)
    return np.asarray(np.clip(magnitude / 4.0, 0.0, 1.0)[..., None], dtype=np.float32)


def _weighted_distance(
    left: NDArray[np.float32],
    right: NDArray[np.float32],
    mask: NDArray[np.float32],
) -> float:
    difference = np.abs(left - right).mean(axis=2)
    return float((difference * mask).sum() / mask.sum())


def _shift(patch: NDArray[np.uint8], dx: int, dy: int) -> NDArray[np.uint8]:
    matrix = np.asarray(((1.0, 0.0, float(dx)), (0.0, 1.0, float(dy))), dtype=np.float32)
    shifted = cv2.warpAffine(
        np.asarray(patch, dtype=np.uint8),
        matrix,
        (_SIZE, _SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.asarray(shifted, dtype=np.uint8)
