from argparse import Namespace
from queue import Queue
from time import monotonic

import numpy as np
import pytest

import scripts.probe_move_observer as probe
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.domain.board import Orientation


def _frame(timestamp_ns: int = 1) -> CaptureFrame:
    return CaptureFrame(timestamp_ns, 1, np.zeros((2, 2, 4), dtype=np.uint8))


def test_quiet_capture_reuses_the_latest_frame_as_a_stability_tick() -> None:
    latest = _frame()

    result = probe._next_tick(Queue(), latest, tick_seconds=0.001, deadline=monotonic() + 1.0)

    assert result is latest


def test_observation_deadline_returns_none_instead_of_a_capture_error() -> None:
    latest = _frame()

    assert probe._next_tick(Queue(), latest, tick_seconds=0.001, deadline=monotonic()) is None


def test_capture_close_is_not_mistaken_for_a_quiet_tick() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(CaptureClosedError("closed"))

    with pytest.raises(CaptureClosedError, match="closed"):
        probe._next_tick(events, _frame(), tick_seconds=0.001, deadline=monotonic() + 1.0)


def test_tick_collapses_a_queued_burst_to_the_newest_frame() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(2))
    events.put(_frame(3))

    result = probe._next_tick(
        events,
        _frame(1),
        tick_seconds=0.001,
        deadline=monotonic() + 1.0,
    )

    assert result is not None
    assert result.timestamp_ns == 3


def test_tick_prioritizes_a_close_event_behind_queued_frames() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(2))
    events.put(CaptureClosedError("closed after frame"))

    with pytest.raises(CaptureClosedError, match="closed after frame"):
        probe._next_tick(
            events,
            _frame(1),
            tick_seconds=0.001,
            deadline=monotonic() + 1.0,
        )


def test_static_single_callback_can_establish_a_clocked_baseline() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    frame = CaptureFrame(1, 1, np.zeros((240, 216, 4), dtype=np.uint8))
    events.put(frame)
    args = Namespace(
        quad="0.05,0.05;0.95,0.05;0.95,0.95;0.05,0.95",
        stable_pairs=2,
        baseline_timeout=0.2,
        fps=100,
        orientation="red_bottom",
    )

    baseline, geometry = probe._stable_baseline(events, args)

    assert baseline is frame
    assert len(geometry.grid_points()) == 90


def test_black_bottom_orientation_is_shared_by_board_and_geometry() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    pixels = np.zeros((240, 216, 4), dtype=np.uint8)
    for timestamp in (1, 2, 3):
        events.put(CaptureFrame(timestamp, 1, pixels))
    args = Namespace(
        quad="0.05,0.05;0.95,0.05;0.95,0.95;0.05,0.95",
        stable_pairs=2,
        baseline_timeout=0.2,
        fps=100,
        orientation="black_bottom",
    )

    _, geometry = probe._stable_baseline(events, args)
    board = probe._parse_board(probe.START_FEN, geometry.orientation)

    assert geometry.orientation is Orientation.BLACK_BOTTOM
    assert board.orientation is geometry.orientation
