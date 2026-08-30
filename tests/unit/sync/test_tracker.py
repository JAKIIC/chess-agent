import numpy as np

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.evidence import MoveEvidence, MoveProposal, ObservationStatus
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.tracker import StableMoveTracker, TrackingStatus
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}


def _geometry() -> BoardGeometry:
    return BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(((12, 12), (204, 12), (204, 228), (12, 228)), (216, 240)),
        (216, 240),
    )


def _render(board: BoardState) -> np.ndarray:
    frame = np.zeros((240, 216, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        value = PALETTE[symbol]
        frame[row * CELL : (row + 1) * CELL, column * CELL : (column + 1) * CELL, :3] = value
    return frame


def _move(board: BoardState, uci: str):
    return next(move for move in legal_moves(board) if move.uci == uci)


def _tracker(board: BoardState) -> StableMoveTracker:
    return StableMoveTracker(
        board,
        _geometry(),
        LegalMoveDiffObserver(patch_size=CELL),
        required_stable_pairs=2,
        patch_size=CELL,
    )


def test_tracker_waits_for_animation_to_end_before_accepting_move() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after_board = apply_move(board, move)
    tracker = _tracker(board)
    tracker.initialize(_render(board))

    first = tracker.push(_render(after_board))
    second = tracker.push(_render(after_board))
    third = tracker.push(_render(after_board))

    assert first.status is TrackingStatus.WAITING_FOR_STABLE
    assert second.status is TrackingStatus.WAITING_FOR_STABLE
    assert third.status is TrackingStatus.ACCEPTED
    assert third.move == move
    assert third.board == after_board


def test_tracker_keeps_last_confirmed_board_when_change_is_ambiguous() -> None:
    board = parse_fen(START)
    after_pieces = list(board.pieces)
    for uci in ("b2b3", "h2h3"):
        move = _move(board, uci)
        after_pieces[move.to_index] = after_pieces[move.from_index]
        after_pieces[move.from_index] = "."
    ambiguous = _render(BoardState(tuple(after_pieces), side_to_move="w"))
    tracker = _tracker(board)
    tracker.initialize(_render(board))

    tracker.push(ambiguous)
    tracker.push(ambiguous.copy())
    paused = tracker.push(ambiguous.copy())
    still_paused = tracker.push(ambiguous.copy())

    assert paused.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert paused.board == board
    assert paused.move is None
    assert still_paused.status is TrackingStatus.MANUAL_RECOVERY_REQUIRED
    assert still_paused.board == board


def test_tracker_returns_to_watching_when_transient_change_restores_baseline() -> None:
    board = parse_fen(START)
    baseline = _render(board)
    transient = baseline.copy()
    transient[20:40, 20:40, :3] = 255
    tracker = _tracker(board)
    tracker.initialize(baseline)

    assert tracker.push(transient).status is TrackingStatus.WAITING_FOR_STABLE
    assert tracker.push(baseline.copy()).status is TrackingStatus.WAITING_FOR_STABLE
    assert tracker.push(baseline.copy()).status is TrackingStatus.WAITING_FOR_STABLE
    restored = tracker.push(baseline.copy())

    assert restored.status is TrackingStatus.WATCHING
    assert restored.board == board
    assert restored.move is None


def test_tracker_rejects_an_observer_supplied_illegal_move_without_changing_board() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    illegal_after_first_commit = _move(apply_move(board, move), "h7e7")

    class IllegalObserver:
        def observe(self, *_args: object) -> MoveProposal:
            return MoveProposal(
                status=ObservationStatus.ACCEPTED,
                move=illegal_after_first_commit,
                evidence_score=1.0,
                evidence=MoveEvidence((), (), ()),
            )

    tracker = StableMoveTracker(
        board,
        _geometry(),
        IllegalObserver(),
        required_stable_pairs=1,
        patch_size=CELL,
    )
    after = _render(apply_move(board, move))
    tracker.initialize(_render(board))
    tracker.push(after)

    result = tracker.push(after.copy())

    assert result.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert result.board == board
    assert result.move is None


def test_tracker_context_invalidation_blocks_frames_until_explicit_recovery() -> None:
    board = parse_fen(START)
    tracker = _tracker(board)
    frame = _render(board)
    tracker.initialize(frame)

    invalidated = tracker.invalidate_context()
    blocked = tracker.push(frame.copy())

    assert invalidated.status is TrackingStatus.CONTEXT_INVALID
    assert blocked.status is TrackingStatus.CONTEXT_INVALID
    assert blocked.board == board


def test_tracker_desynchronization_requires_manual_recovery() -> None:
    board = parse_fen(START)
    tracker = _tracker(board)
    frame = _render(board)
    tracker.initialize(frame)

    desynchronized = tracker.mark_desynchronized()
    blocked = tracker.push(frame.copy())

    assert desynchronized.status is TrackingStatus.DESYNCHRONIZED
    assert blocked.status is TrackingStatus.MANUAL_RECOVERY_REQUIRED
    assert blocked.board == board


def test_manual_recovery_replaces_board_and_confirmed_frame() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    recovered_board = apply_move(board, move)
    recovered_frame = _render(recovered_board)
    tracker = _tracker(board)
    tracker.initialize(_render(board))
    tracker.mark_desynchronized()

    update = tracker.recover(recovered_board, recovered_frame)
    accepted_move = _move(recovered_board, "h7e7")
    after_recovery = apply_move(recovered_board, accepted_move)
    tracker.push(_render(after_recovery))
    tracker.push(_render(after_recovery))
    accepted = tracker.push(_render(after_recovery))

    assert update.status is TrackingStatus.WATCHING
    assert update.board == recovered_board
    assert tracker.board == after_recovery
    assert accepted.status is TrackingStatus.ACCEPTED
