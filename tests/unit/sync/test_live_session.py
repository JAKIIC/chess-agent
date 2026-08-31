from __future__ import annotations

from time import monotonic, sleep

import numpy as np

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.live_session import (
    LiveSyncSession,
    LiveSyncStatus,
)
from xiangqi_agent.vision.geometry import parse_normalized_quad

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


def _wait_until(predicate: object, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        sleep(0.01)
    raise AssertionError("timed out waiting for live sync update")


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

    session.recover(board, quad=QUAD)
    _wait_until(
        lambda: any(
            update.status is LiveSyncStatus.WATCHING
            for update in updates
            if update.frame_size == (432, 240)
        )
    )

    moved = np.repeat(_render(after), 2, axis=1)
    source.push(moved, 250_000_000)
    source.push(moved.copy(), 360_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.MOVE_ACCEPTED for update in updates))

    assert session.board == after
    session.close()
