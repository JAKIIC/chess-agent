from __future__ import annotations

from threading import Event, Lock, current_thread
from time import monotonic, perf_counter_ns, sleep
from typing import Self

import numpy as np
import pytest

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.live_session import (
    LiveSyncSession,
    LiveSyncStatus,
    _CoalescingEventQueue,
)
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.vision.geometry import parse_normalized_quad
from xiangqi_agent.vision.occupancy import OccupancyEvidence

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
QUAD = parse_normalized_quad("0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540")


def _render(board: BoardState) -> np.ndarray:
    frame = np.zeros((240, 216, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        value = PALETTE[symbol]
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = value
    return frame


def _move(board: BoardState, uci: str):
    return next(move for move in legal_moves(board) if move.uci == uci)


def _occupancy_for(board: BoardState, confidence: float = 0.95) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (confidence,) * 90,
        "literal-v1",
    )


def _wait_until(predicate: object, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        sleep(0.01)
    raise AssertionError("timed out waiting for live sync update")


class _ThrowingCloseSource(FakeFrameSource):
    def __init__(self, hwnd: int) -> None:
        super().__init__(hwnd)
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        super().close()
        raise RuntimeError("close failed")


class _BurstAwareFakeSource(FakeFrameSource):
    def __init__(self, hwnd: int) -> None:
        super().__init__(hwnd)
        self.burst_modes: list[bool] = []

    def set_bursting(self, active: bool) -> None:
        self.burst_modes.append(active)


class _RecoveryFirstLock:
    def __init__(self) -> None:
        self.worker_waiting = Event()
        self.allow_worker = Event()
        self._lock = Lock()

    def __enter__(self) -> Self:
        if current_thread().name == "xiangqi-live-sync":
            self.worker_waiting.set()
            assert self.allow_worker.wait(2.0)
        self._lock.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self._lock.release()


class _SequenceOccupancyObserver:
    def __init__(self, values: tuple[OccupancyEvidence, ...]) -> None:
        self._values = list(values)

    def observe(self, _frame: np.ndarray, _geometry: object) -> OccupancyEvidence:
        if not self._values:
            raise AssertionError("occupancy observer received an unexpected call")
        return self._values.pop(0)


def test_live_event_queue_coalesces_frames_and_preserves_terminal_events() -> None:
    events = _CoalescingEventQueue(max_frames=3)
    pixels = np.zeros((2, 2, 4), dtype=np.uint8)

    for timestamp in range(10):
        events.put_frame(CaptureFrame(timestamp, 42, pixels))

    assert events.pending_frame_count == 3
    assert [events.get(timeout=0).timestamp_ns for _ in range(3)] == [7, 8, 9]

    error = CaptureClosedError("target closed")
    events.put_terminal(error)
    assert events.get(timeout=0) is error


def test_live_session_tracks_multiple_unique_moves_without_restarting() -> None:
    board = parse_fen(START)
    red_move = _move(board, "h2e2")
    after_red = apply_move(board, red_move)
    black_move = _move(after_red, "h7e7")
    after_black = apply_move(after_red, black_move)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )

    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    red_frame = _render(after_red)
    source.push(red_frame, 150_000_000)
    source.push(red_frame.copy(), 260_000_000)
    _wait_until(lambda: sum(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates) == 1)

    black_frame = _render(after_black)
    source.push(black_frame, 310_000_000)
    source.push(black_frame.copy(), 420_000_000)
    _wait_until(lambda: sum(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates) == 2)

    accepted = [update for update in updates if update.status is LiveSyncStatus.MOVE_ACCEPTED]
    assert [update.move.uci for update in accepted if update.move is not None] == [
        "h2e2",
        "h7e7",
    ]
    assert session.board == after_black

    session.close()
    session.close()


def test_live_session_requires_an_observer_for_matching_baselines() -> None:
    with pytest.raises(ValueError, match="occupancy observer"):
        LiveSyncSession(
            FakeFrameSource(hwnd=42),
            parse_fen(START),
            QUAD,
            require_matching_baseline=True,
        )


def test_live_session_rejects_mismatched_baseline_then_recovers_when_visible() -> None:
    board = parse_fen(START)
    mismatched = list(_occupancy_for(board).occupied)
    mismatched[0] = not mismatched[0]
    observer = _SequenceOccupancyObserver(
        (
            OccupancyEvidence(tuple(mismatched), (0.95,) * 90, "literal-v1"),
            _occupancy_for(board),
        )
    )
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        occupancy_observer=observer,
        require_matching_baseline=True,
        patch_size=CELL,
        stable_pairs=2,
    )
    session.start()
    baseline = _render(board)

    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.CONTEXT_INVALID for update in updates))
    assert all(update.status is not LiveSyncStatus.BASELINE_READY for update in updates)

    source.push(baseline.copy(), 150_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    assert session.board == board
    session.close()


def test_live_session_rejects_low_confidence_baseline() -> None:
    board = parse_fen(START)
    observer = _SequenceOccupancyObserver((_occupancy_for(board, confidence=0.64),))
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        occupancy_observer=observer,
        require_matching_baseline=True,
        baseline_minimum_confidence=0.65,
        patch_size=CELL,
        stable_pairs=2,
    )
    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)

    _wait_until(lambda: any(update.status is LiveSyncStatus.CONTEXT_INVALID for update in updates))

    assert all(update.status is not LiveSyncStatus.BASELINE_READY for update in updates)
    session.close()


def test_live_session_emits_atomic_two_ply_event_only_in_human_ai_mode() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        sync_mode=SyncMode.HUMAN_VS_AI,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )

    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    merged = _render(final)
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates))

    accepted = next(update for update in updates if update.status is LiveSyncStatus.MOVE_ACCEPTED)
    assert accepted.moves == (first, second)
    assert accepted.move is None
    assert accepted.before_position_id == board.position_id
    assert accepted.after_position_id == final.position_id
    assert accepted.sync_mode is SyncMode.HUMAN_VS_AI
    assert accepted.transition_evidence is None
    assert session.board == final
    session.close()


def test_live_session_forwards_explicit_transition_capture_evidence() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        sync_mode=SyncMode.HUMAN_VS_AI,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
        capture_transition_evidence=True,
    )

    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    merged = _render(final)
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates))

    accepted = next(update for update in updates if update.status is LiveSyncStatus.MOVE_ACCEPTED)
    assert accepted.transition_evidence is not None
    assert accepted.transition_evidence.changed_points == (22, 25, 67, 70)
    assert len(accepted.transition_evidence.crops) == 4
    session.close()


def test_live_session_defaults_to_strict_and_rejects_a_merged_two_ply_frame() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )

    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    merged = _render(final)
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)
    _wait_until(
        lambda: any(update.status is LiveSyncStatus.PAUSED_AMBIGUOUS for update in updates)
    )

    paused = next(update for update in updates if update.status is LiveSyncStatus.PAUSED_AMBIGUOUS)
    assert paused.sync_mode is SyncMode.STRICT_SINGLE
    assert paused.moves == ()
    assert session.board == board
    session.close()


def test_live_session_requests_burst_capture_only_while_a_move_is_settling() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = apply_move(board, move)
    source = _BurstAwareFakeSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )

    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    moved = _render(after)
    source.push(moved, 150_000_000)
    _wait_until(lambda: True in source.burst_modes)
    source.push(moved.copy(), 260_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates))

    assert source.burst_modes[-1] is False
    session.close()


def test_live_session_rebinds_proportional_resize_and_keeps_tracking() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = apply_move(board, move)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )
    session.start()

    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    resized = np.repeat(np.repeat(baseline, 2, axis=0), 2, axis=1)
    source.push(resized, 200_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.GEOMETRY_REBOUND for update in updates))

    moved = np.repeat(np.repeat(_render(after), 2, axis=0), 2, axis=1)
    source.push(moved, 250_000_000)
    source.push(moved.copy(), 360_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates))

    accepted = next(update for update in updates if update.status is LiveSyncStatus.MOVE_ACCEPTED)
    assert accepted.move == move
    assert accepted.board == after
    assert session.board == after
    session.close()


def test_live_session_requires_explicit_recovery_after_unsafe_resize() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = apply_move(board, move)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        settle_ms=100,
        stable_pairs=2,
    )
    session.start()

    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    stretched = np.repeat(baseline, 2, axis=1)
    source.push(stretched, 200_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.CONTEXT_INVALID for update in updates))
    assert session.board == board

    session.recover(after, quad=QUAD)
    _wait_until(lambda: any(update.status is LiveSyncStatus.RECOVERY_PENDING for update in updates))
    assert session.board == board

    moved = np.repeat(_render(after), 2, axis=1)
    animated = moved.copy()
    animated[:CELL, : CELL * 2, :3] = 255
    source.push(animated, 250_000_000)
    source.push(moved, 300_000_000)
    source.push(moved.copy(), 360_000_000)
    sleep(0.05)
    assert session.board == board

    source.push(moved.copy(), 420_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.RECOVERY_ACCEPTED for update in updates))

    assert session.board == after
    session.close()


def test_live_session_closes_owned_source_after_processing_error() -> None:
    board = parse_fen(START)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        stable_pairs=2,
    )
    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    source.push(baseline.copy(), 99_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.ERROR for update in updates))
    _wait_until(lambda: any(update.status is LiveSyncStatus.CLOSED for update in updates))

    with pytest.raises(CaptureClosedError, match="closed"):
        source.push(baseline.copy(), 110_000_000)
    session.close()


def test_live_session_close_finishes_when_owned_source_close_raises() -> None:
    source = _ThrowingCloseSource(hwnd=42)
    updates = []
    session = LiveSyncSession(source, parse_fen(START), QUAD, on_update=updates.append)
    session.start()

    session.close()
    session.close()

    assert source.close_calls >= 1
    assert [update.status for update in updates].count(LiveSyncStatus.CLOSED) == 1


def test_live_session_close_releases_full_frame_state() -> None:
    board = parse_fen(START)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        stable_pairs=2,
    )
    session.start()
    baseline = _render(board)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    session.close()
    session.close()

    assert session._latest_frame is None
    assert session._tracker is None
    assert session._sampler is None


def test_recovery_wins_over_a_frame_dequeued_before_the_processing_lock() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = apply_move(board, move)
    source = FakeFrameSource(hwnd=42)
    updates = []
    session = LiveSyncSession(
        source,
        board,
        QUAD,
        on_update=updates.append,
        patch_size=CELL,
        steady_fps=1,
        settle_ms=100,
        stable_pairs=2,
    )
    session.start()
    baseline = _render(board)
    timestamp = perf_counter_ns()
    source.push(baseline, timestamp)
    source.push(baseline.copy(), timestamp + 50_000_000)
    source.push(baseline.copy(), timestamp + 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))

    ordered_lock = _RecoveryFirstLock()
    session._processing_lock = ordered_lock
    moved = _render(after)
    source.push(moved, timestamp + 150_000_000)
    assert ordered_lock.worker_waiting.wait(2.0)

    session.recover(after)
    pending_index = next(
        index
        for index, update in enumerate(updates)
        if update.status is LiveSyncStatus.RECOVERY_PENDING
    )
    ordered_lock.allow_worker.set()
    sleep(0.05)

    assert all(
        update.status
        not in (LiveSyncStatus.MOVE_ACCEPTED, LiveSyncStatus.WAITING_FOR_STABLE)
        for update in updates[pending_index + 1 :]
    )

    source.push(moved.copy(), timestamp + 210_000_000)
    source.push(moved.copy(), timestamp + 270_000_000)
    source.push(moved.copy(), timestamp + 330_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.RECOVERY_ACCEPTED for update in updates))
    assert session.board == after
    session.close()
