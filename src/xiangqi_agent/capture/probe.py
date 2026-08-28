from dataclasses import dataclass
from itertools import pairwise

from xiangqi_agent.capture.protocol import CaptureFrame
from xiangqi_agent.vision.change_detection import FrameChange, analyze_frame_change
from xiangqi_agent.vision.geometry import BoardGeometry


class CaptureProbeError(RuntimeError):
    """The short capture probe could not produce useful metrics."""


@dataclass(frozen=True, slots=True)
class CaptureSummary:
    frame_count: int
    first_size: tuple[int, int]
    last_size: tuple[int, int]
    size_change_count: int
    timestamps_monotonic: bool
    effective_fps: float
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class ChangeSequenceSummary:
    comparison_count: int
    stable_comparison_count: int
    trailing_stable_comparisons: int
    peak_global_difference: float
    most_changed: tuple[tuple[int, float], ...]


def summarize_capture(frames: tuple[CaptureFrame, ...]) -> CaptureSummary:
    if not frames:
        raise CaptureProbeError("capture produced no frames")
    timestamps = tuple(frame.timestamp_ns for frame in frames)
    monotonic = all(right > left for left, right in pairwise(timestamps))
    sizes = tuple(frame.size for frame in frames)
    changes = sum(right != left for left, right in pairwise(sizes))
    duration = max(0.0, (timestamps[-1] - timestamps[0]) / 1_000_000_000)
    effective_fps = (len(frames) - 1) / duration if monotonic and duration > 0 else 0.0
    return CaptureSummary(
        frame_count=len(frames),
        first_size=sizes[0],
        last_size=sizes[-1],
        size_change_count=changes,
        timestamps_monotonic=monotonic,
        effective_fps=effective_fps,
        duration_seconds=duration,
    )


def analyze_change_sequence(
    frames: tuple[CaptureFrame, ...],
    geometry: BoardGeometry,
    *,
    global_threshold: float = 1.5,
    local_threshold: float = 3.0,
    top_k: int = 6,
) -> ChangeSequenceSummary:
    if len(frames) < 2:
        raise CaptureProbeError("change analysis needs at least two frames")
    if any(frame.size != geometry.frame_size for frame in frames):
        raise CaptureProbeError("frame size changed after calibration")
    comparisons = tuple(
        analyze_frame_change(
            before.bgra,
            after.bgra,
            geometry,
            global_threshold=global_threshold,
            local_threshold=local_threshold,
            top_k=top_k,
        )
        for before, after in pairwise(frames)
    )
    peak = max(comparisons, key=lambda item: item.global_difference)
    stable_count = sum(item.stable for item in comparisons)
    trailing = _trailing_stable_count(comparisons)
    return ChangeSequenceSummary(
        comparison_count=len(comparisons),
        stable_comparison_count=stable_count,
        trailing_stable_comparisons=trailing,
        peak_global_difference=peak.global_difference,
        most_changed=tuple((index, peak.local_differences[index]) for index in peak.most_changed_indices),
    )


def _trailing_stable_count(comparisons: tuple[FrameChange, ...]) -> int:
    count = 0
    for item in reversed(comparisons):
        if not item.stable:
            break
        count += 1
    return count
