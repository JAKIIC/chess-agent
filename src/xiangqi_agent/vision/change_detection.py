from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.vision.geometry import BoardGeometry


@dataclass(frozen=True, slots=True)
class FrameChange:
    global_difference: float
    local_differences: tuple[float, ...]
    most_changed_indices: tuple[int, ...]
    changed_indices: tuple[int, ...]
    stable: bool


def analyze_frame_change(
    before: NDArray[np.generic],
    after: NDArray[np.generic],
    geometry: BoardGeometry,
    *,
    global_threshold: float = 1.5,
    local_threshold: float = 3.0,
    top_k: int = 6,
    patch_size: int = 32,
) -> FrameChange:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    before_pixels = np.asarray(before)
    after_pixels = np.asarray(after)
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("consecutive frame sizes differ")
    before_patches = geometry.crop_intersections(before_pixels, size=patch_size)
    after_patches = geometry.crop_intersections(after_pixels, size=patch_size)
    global_difference = float(
        np.abs(before_pixels[..., :3].astype(np.int16) - after_pixels[..., :3].astype(np.int16)).mean()
    )
    local = tuple(
        float(np.abs(left[..., :3].astype(np.int16) - right[..., :3].astype(np.int16)).mean())
        for left, right in zip(before_patches, after_patches, strict=True)
    )
    ranked = tuple(sorted(range(len(local)), key=lambda index: (-local[index], index))[: min(top_k, len(local))])
    changed = tuple(index for index, difference in enumerate(local) if difference > local_threshold)
    pair_stable = global_difference <= global_threshold and not changed
    return FrameChange(
        global_difference=global_difference,
        local_differences=local,
        most_changed_indices=ranked,
        changed_indices=changed,
        stable=pair_stable,
    )


class FrameStabilityDetector:
    def __init__(
        self,
        geometry: BoardGeometry,
        *,
        required_stable_pairs: int = 2,
        global_threshold: float = 1.5,
        local_threshold: float = 3.0,
        top_k: int = 6,
    ) -> None:
        if required_stable_pairs <= 0:
            raise ValueError("required_stable_pairs must be positive")
        self._geometry = geometry
        self._required = required_stable_pairs
        self._global_threshold = global_threshold
        self._local_threshold = local_threshold
        self._top_k = top_k
        self._previous: NDArray[np.uint8] | None = None
        self._stable_pairs = 0

    def update(self, frame: NDArray[np.generic]) -> FrameChange | None:
        current = np.array(frame, dtype=np.uint8, copy=True, order="C")
        if self._previous is None:
            self._previous = current
            return None
        result = analyze_frame_change(
            self._previous,
            current,
            self._geometry,
            global_threshold=self._global_threshold,
            local_threshold=self._local_threshold,
            top_k=self._top_k,
        )
        self._previous = current
        self._stable_pairs = self._stable_pairs + 1 if result.stable else 0
        return replace(result, stable=self._stable_pairs >= self._required)
