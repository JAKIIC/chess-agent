import numpy as np
import pytest

from xiangqi_agent.capture.probe import (
    CaptureProbeError,
    analyze_change_sequence,
    summarize_capture,
)
from xiangqi_agent.capture.protocol import CaptureFrame
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad


def _frame(timestamp_ns: int, width: int = 30, height: int = 20) -> CaptureFrame:
    return CaptureFrame(
        timestamp_ns=timestamp_ns,
        hwnd=7,
        bgra=np.zeros((height, width, 4), dtype=np.uint8),
    )


def test_capture_summary_reports_rate_monotonicity_and_size_changes() -> None:
    frames = (_frame(1_000_000_000), _frame(1_500_000_000), _frame(2_000_000_000, 40, 25))

    summary = summarize_capture(frames)

    assert summary.frame_count == 3
    assert summary.first_size == (30, 20)
    assert summary.last_size == (40, 25)
    assert summary.size_change_count == 1
    assert summary.timestamps_monotonic is True
    assert summary.effective_fps == pytest.approx(2.0)


def test_capture_summary_rejects_empty_and_flags_non_monotonic_frames() -> None:
    with pytest.raises(CaptureProbeError, match="no frames"):
        summarize_capture(())

    summary = summarize_capture((_frame(200), _frame(100)))
    assert summary.timestamps_monotonic is False
    assert summary.effective_fps == 0.0


def test_change_sequence_reports_stability_and_peak_local_intersection() -> None:
    geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(((4, 4), (68, 4), (68, 76), (4, 76)), (73, 81)),
        (73, 81),
    )
    first = _frame(1, 73, 81)
    changed_pixels = first.bgra.copy()
    changed_pixels.setflags(write=True)
    x, y = geometry.grid_points()[21]
    changed_pixels[round(y) - 2 : round(y) + 3, round(x) - 2 : round(x) + 3, :3] = 255
    changed = CaptureFrame(timestamp_ns=2, hwnd=7, bgra=changed_pixels)
    stable = CaptureFrame(timestamp_ns=3, hwnd=7, bgra=changed.bgra)

    summary = analyze_change_sequence((first, changed, stable), geometry, top_k=3)

    assert summary.comparison_count == 2
    assert summary.stable_comparison_count == 1
    assert summary.trailing_stable_comparisons == 1
    assert summary.peak_global_difference > 0
    assert summary.most_changed[0][0] == 21


def test_change_sequence_rejects_too_few_frames_and_resize() -> None:
    geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(((4, 4), (68, 4), (68, 76), (4, 76)), (73, 81)),
        (73, 81),
    )
    with pytest.raises(CaptureProbeError, match="at least two"):
        analyze_change_sequence((_frame(1, 73, 81),), geometry)
    with pytest.raises(CaptureProbeError, match="size changed"):
        analyze_change_sequence((_frame(1, 73, 81), _frame(2, 74, 81)), geometry)
