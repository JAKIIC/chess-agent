from time import perf_counter

import numpy as np
import pytest

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter
from xiangqi_agent.sync.evidence import (
    MoveEvidence,
    MoveProposal,
    MoveSequenceEvidence,
    MoveSequenceProposal,
    ObservationStatus,
    SequenceCandidateEvidence,
)
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.sequence_observer import LegalTwoPlyDiffObserver
from xiangqi_agent.sync.tracker import StableMoveTracker, TrackingStatus
from xiangqi_agent.sync.transition_capture import TransitionCaptureEvidence
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.occupancy import OccupancyEvidence

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


def _scale_two_times(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1)


def _move(board: BoardState, uci: str):
    return next(move for move in legal_moves(board) if move.uci == uci)


def _tracker(
    board: BoardState,
    *,
    mode: SyncMode = SyncMode.STRICT_SINGLE,
    sequence_observer: object | None = None,
    committer: object | None = None,
    capture_transition_evidence: bool = False,
    occupancy_observer: object | None = None,
    require_atomic_two_ply: bool = False,
) -> StableMoveTracker:
    return StableMoveTracker(
        board,
        _geometry(),
        LegalMoveDiffObserver(patch_size=CELL),
        committer=committer,
        mode=mode,
        sequence_observer=sequence_observer,
        required_stable_pairs=2,
        patch_size=CELL,
        capture_transition_evidence=capture_transition_evidence,
        occupancy_observer=occupancy_observer,
        require_atomic_two_ply=require_atomic_two_ply,
    )


def _settle(tracker: StableMoveTracker, frame: np.ndarray):
    tracker.push(frame)
    tracker.push(frame.copy())
    return tracker.push(frame.copy())


def test_sequence_proposal_requires_exactly_two_moves_when_accepted() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        MoveSequenceProposal(
            status=ObservationStatus.ACCEPTED,
            moves=(),
            evidence_score=1.0,
            evidence=MoveSequenceEvidence((), (), (), "two-ply-v1"),
        )


def test_sequence_proposal_rejects_moves_on_ambiguous_result() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    reply = _move(apply_move(board, move), "h7e7")

    with pytest.raises(ValueError, match="must not expose"):
        MoveSequenceProposal(
            status=ObservationStatus.AMBIGUOUS,
            moves=(move, reply),
            evidence_score=0.0,
            evidence=MoveSequenceEvidence(
                (),
                (),
                ("candidate_margin",),
                "two-ply-v1",
            ),
        )


def test_strict_tracker_never_invokes_two_ply_observer() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    class RecordingSequenceObserver:
        def __init__(self) -> None:
            self.calls = 0

        def observe(self, *_args: object) -> MoveSequenceProposal:
            self.calls += 1
            return MoveSequenceProposal(
                status=ObservationStatus.ACCEPTED,
                moves=(first, second),
                evidence_score=1.0,
                evidence=MoveSequenceEvidence((), (), (), "two-ply-v1"),
            )

    sequence = RecordingSequenceObserver()
    tracker = _tracker(board, sequence_observer=sequence)
    tracker.initialize(_render(board))

    result = _settle(tracker, _render(final))

    assert result.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert sequence.calls == 0
    assert result.board == board


def test_human_ai_tracker_atomically_accepts_unique_two_ply_fallback() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
    )
    tracker.initialize(_render(board))

    result = _settle(tracker, _render(final))

    assert result.status is TrackingStatus.ACCEPTED
    assert result.moves == (first, second)
    assert result.move is None
    assert result.board == final


def test_transition_capture_is_disabled_by_default_without_calling_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    tracker = _tracker(board)
    tracker.initialize(_render(board))

    def fail_if_called(*_args: object, **_kwargs: object) -> TransitionCaptureEvidence:
        raise AssertionError("disabled transition capture must not build crop evidence")

    monkeypatch.setattr(
        "xiangqi_agent.sync.tracker.build_transition_capture_evidence",
        fail_if_called,
    )

    result = _settle(tracker, _render(apply_move(board, move)))

    assert result.status is TrackingStatus.ACCEPTED
    assert result.transition_evidence is None


def test_enabled_transition_capture_keeps_only_owned_top_four_point_crops() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    baseline = _render(board)
    final_frame = _render(final)
    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        capture_transition_evidence=True,
    )
    tracker.initialize(baseline)

    result = _settle(tracker, final_frame)
    evidence = result.transition_evidence

    assert result.status is TrackingStatus.ACCEPTED
    assert evidence is not None
    assert evidence.changed_points == (22, 25, 67, 70)
    assert len(evidence.local_differences) == 90
    assert evidence.decision_latency_ms >= 0.0
    assert tuple(crop.point_index for crop in evidence.crops) == evidence.changed_points
    before_copy = tuple(crop.before.copy() for crop in evidence.crops)
    after_copy = tuple(crop.after.copy() for crop in evidence.crops)
    assert all(crop.before.shape == (48, 48, 4) for crop in evidence.crops)
    assert all(crop.after.shape == (48, 48, 4) for crop in evidence.crops)
    assert all(not crop.before.flags.writeable for crop in evidence.crops)
    assert all(not crop.after.flags.writeable for crop in evidence.crops)

    baseline[...] = 255
    final_frame[...] = 0

    assert all(
        np.array_equal(crop.before, expected)
        for crop, expected in zip(evidence.crops, before_copy, strict=True)
    )
    assert all(
        np.array_equal(crop.after, expected)
        for crop, expected in zip(evidence.crops, after_copy, strict=True)
    )


def test_enabled_transition_capture_attaches_before_and_after_occupancy() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = apply_move(board, move)
    observer = _SequenceOccupancyObserver(
        (_occupancy_for(board), _occupancy_for(after))
    )
    tracker = _tracker(
        board,
        capture_transition_evidence=True,
        occupancy_observer=observer,
    )
    tracker.initialize(_render(board))

    result = _settle(tracker, _render(after))

    assert result.transition_evidence is not None
    assert result.transition_evidence.before_occupancy == _occupancy_for(board)
    assert result.transition_evidence.after_occupancy == _occupancy_for(after)
    assert observer.remaining == 0


def test_transition_latency_uses_the_final_capture_timestamp_when_clocks_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final_frame = _render(apply_move(middle, second))
    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        capture_transition_evidence=True,
    )
    tracker.initialize(_render(board))
    tracker.push(final_frame)
    tracker.push(final_frame.copy())
    monkeypatch.setattr("xiangqi_agent.sync.tracker.perf_counter_ns", lambda: 10_000_000_000)

    result = tracker.push(
        final_frame.copy(),
        capture_timestamp_ns=9_500_000_000,
    )

    assert result.transition_evidence is not None
    assert result.transition_evidence.decision_latency_ms == 500.0


def test_human_ai_tracker_rejects_sequence_whose_evidence_final_id_is_wrong() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    candidate = SequenceCandidateEvidence(
        moves=(first, second),
        changed_points=(19, 22, 64, 67),
        expected_change_floor=90.0,
        unexpected_difference=0.0,
        maximum_template_distance=0.0,
        minimum_template_margin=1.0,
        minimum_template_confidence=1.0,
        score=90.0,
        final_position_id="0" * 32,
    )

    class WrongEvidenceObserver:
        def observe(self, *_args: object) -> MoveSequenceProposal:
            return MoveSequenceProposal(
                status=ObservationStatus.ACCEPTED,
                moves=(first, second),
                evidence_score=1.0,
                evidence=MoveSequenceEvidence(
                    (candidate,),
                    (),
                    (),
                    "two-ply-v1",
                ),
            )

    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=WrongEvidenceObserver(),
    )
    tracker.initialize(_render(board))

    result = _settle(tracker, _render(final))

    assert result.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert result.moves == ()
    assert result.board == board
    assert isinstance(result.observation, MoveSequenceProposal)
    assert result.observation.status is ObservationStatus.AMBIGUOUS
    assert result.observation.moves == ()
    assert "final_position_mismatch" in result.observation.evidence.rejection_reasons


def test_human_ai_tracker_does_not_publish_first_move_when_atomic_commit_fails() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    class FailingSequenceCommitter(RuleStateCommitter):
        def commit_sequence(self, *_args: object) -> BoardState:
            raise ValueError("second move failed")

    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        committer=FailingSequenceCommitter(),
    )
    tracker.initialize(_render(board))

    result = _settle(tracker, _render(final))

    assert result.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert result.moves == ()
    assert result.board == board
    assert tracker.board == board
    assert isinstance(result.observation, MoveSequenceProposal)
    assert result.observation.status is ObservationStatus.AMBIGUOUS
    assert result.observation.moves == ()
    assert "sequence_commit_failed" in result.observation.evidence.rejection_reasons


def test_human_ai_tracker_keeps_two_single_events_when_a_stable_platform_exists() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    tracker = _tracker(board, mode=SyncMode.HUMAN_VS_AI)
    tracker.initialize(_render(board))

    first_update = _settle(tracker, _render(middle))
    second_update = _settle(tracker, _render(final))

    assert first_update.moves == (first,)
    assert second_update.moves == (second,)
    assert second_update.board == final


def test_atomic_human_ai_tracker_waits_across_stable_first_ply_then_accepts_pair() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        require_atomic_two_ply=True,
    )
    tracker.initialize(_render(board))

    waiting = _settle(tracker, _render(middle))
    still_waiting = tracker.push(_render(middle))
    accepted = _settle(tracker, _render(final))

    assert waiting.status is TrackingStatus.WAITING_FOR_REPLY
    assert still_waiting.status is TrackingStatus.WAITING_FOR_REPLY
    assert waiting.moves == ()
    assert waiting.board == board
    assert tracker.board == final
    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.moves == (first, second)
    assert accepted.board == final


def test_atomic_pending_reply_skips_repeating_the_single_move_observer() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    class SingleMoveObserver:
        def __init__(self) -> None:
            self.calls = 0
            self._delegate = LegalMoveDiffObserver(patch_size=CELL)

        def observe(self, *args: object) -> MoveProposal:
            self.calls += 1
            if self.calls > 1:
                raise AssertionError("pending reply repeated single-move enumeration")
            return self._delegate.observe(*args)  # type: ignore[arg-type]

    observer = SingleMoveObserver()
    tracker = StableMoveTracker(
        board,
        _geometry(),
        observer,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        required_stable_pairs=2,
        patch_size=CELL,
        require_atomic_two_ply=True,
    )
    tracker.initialize(_render(board))

    waiting = _settle(tracker, _render(middle))
    started = perf_counter()
    accepted = _settle(tracker, _render(final))
    elapsed_ms = (perf_counter() - started) * 1000

    assert waiting.status is TrackingStatus.WAITING_FOR_REPLY
    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.moves == (first, second)
    assert isinstance(accepted.observation, MoveSequenceProposal)
    assert (
        accepted.observation.evidence.feature_version
        == "two-ply-template-transfer-v5"
    )
    assert observer.calls == 1
    assert elapsed_ms < 500


def test_atomic_human_ai_tracker_uses_unique_occupancy_for_ambiguous_intermediate() -> None:
    board = parse_fen(START)
    first = _move(board, "h0g2")
    middle = apply_move(board, first)
    second = _move(middle, "h9g7")
    final = apply_move(middle, second)

    class AmbiguousMoveObserver:
        def observe(self, *_args: object) -> MoveProposal:
            return MoveProposal(
                status=ObservationStatus.AMBIGUOUS,
                move=None,
                evidence_score=0.0,
                evidence=MoveEvidence(
                    (),
                    (10.0,) * 90,
                    ("candidate_score",),
                ),
            )

    intermediate_occupancy = _occupancy_for(middle)
    confidences = list(intermediate_occupancy.confidences)
    confidences[first.from_index] = 0.14
    confidences[first.to_index] = 0.14
    occupancy = _SequenceOccupancyObserver(
        (
            OccupancyEvidence(
                intermediate_occupancy.occupied,
                tuple(confidences),
                "literal-v1",
            ),
        )
    )
    tracker = StableMoveTracker(
        board,
        _geometry(),
        AmbiguousMoveObserver(),
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        required_stable_pairs=2,
        patch_size=CELL,
        occupancy_observer=occupancy,
        require_atomic_two_ply=True,
    )
    tracker.initialize(_render(board))

    waiting = _settle(tracker, _render(middle))
    accepted = _settle(tracker, _render(final))

    assert waiting.status is TrackingStatus.WAITING_FOR_REPLY
    assert waiting.board == board
    assert occupancy.remaining == 0
    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.moves == (first, second)
    assert accepted.board == final


def test_atomic_human_ai_tracker_rejects_final_sequence_that_disagrees_with_intermediate() -> None:
    board = parse_fen(START)
    pending_first = _move(board, "h2e2")
    pending_board = apply_move(board, pending_first)
    different_first = _move(board, "b2b3")
    different_middle = apply_move(board, different_first)
    different_second = _move(different_middle, "h7e7")
    different_final = apply_move(different_middle, different_second)
    tracker = _tracker(
        board,
        mode=SyncMode.HUMAN_VS_AI,
        sequence_observer=LegalTwoPlyDiffObserver(patch_size=CELL),
        require_atomic_two_ply=True,
    )
    tracker.initialize(_render(board))

    waiting = _settle(tracker, _render(pending_board))
    rejected = _settle(tracker, _render(different_final))

    assert waiting.status is TrackingStatus.WAITING_FOR_REPLY
    assert rejected.status is TrackingStatus.PAUSED_AMBIGUOUS
    assert rejected.board == board
    assert isinstance(rejected.observation, MoveSequenceProposal)
    assert "intermediate_move_mismatch" in rejected.observation.evidence.rejection_reasons


def test_atomic_two_ply_requires_human_ai_mode() -> None:
    board = parse_fen(START)

    with pytest.raises(ValueError, match="human-vs-AI"):
        _tracker(board, require_atomic_two_ply=True)


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


def test_tracker_keeps_watching_after_selection_highlight_then_accepts_move() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    baseline = _render(board)
    selected = baseline.copy()
    source_row, source_column = divmod(move.from_index, 9)
    selected[
        source_row * CELL : (source_row + 1) * CELL,
        source_column * CELL : (source_column + 1) * CELL,
        :3,
    ] = np.clip(
        selected[
            source_row * CELL : (source_row + 1) * CELL,
            source_column * CELL : (source_column + 1) * CELL,
            :3,
        ].astype(np.int16)
        + 60,
        0,
        255,
    ).astype(np.uint8)
    tracker = _tracker(board)
    tracker.initialize(baseline)

    tracker.push(selected)
    tracker.push(selected.copy())
    selection = tracker.push(selected.copy())

    assert selection.status.value == "waiting_for_endpoint"
    assert selection.board == board

    after = _render(apply_move(board, move))
    tracker.push(after)
    tracker.push(after.copy())
    accepted = tracker.push(after.copy())

    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.move == move


def test_tracker_waits_when_selection_cosmetically_changes_both_move_endpoints() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    baseline = _render(board)
    checker = (np.indices((CELL, CELL)).sum(axis=0) % 2) * 40 - 20
    for index in (move.from_index, move.to_index):
        row, column = divmod(index, 9)
        patch = baseline[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ]
        patch[:] = np.clip(
            patch.astype(np.int16) + checker[..., None],
            0,
            255,
        ).astype(np.uint8)

    selected = baseline.copy()
    for index in (move.from_index, move.to_index):
        row, column = divmod(index, 9)
        patch = selected[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ]
        patch[:] = np.roll(patch, shift=1, axis=1)

    tracker = _tracker(board)
    tracker.initialize(baseline)
    tracker.push(selected)
    tracker.push(selected.copy())

    selection = tracker.push(selected.copy())

    assert selection.status is TrackingStatus.WAITING_FOR_ENDPOINT
    assert selection.board == board
    assert selection.move is None


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
    assert isinstance(result.observation, MoveProposal)
    assert result.observation.status is ObservationStatus.AMBIGUOUS
    assert result.observation.move is None
    assert "rule_commit_failed" in result.observation.evidence.rejection_reasons


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


def test_tracker_context_invalidation_releases_full_frame_state() -> None:
    board = parse_fen(START)
    tracker = _tracker(board)
    frame = _render(board)
    tracker.initialize(frame)

    invalid = tracker.invalidate_context()

    assert invalid.status is TrackingStatus.CONTEXT_INVALID
    assert tracker._confirmed_frame is None
    assert tracker._previous_frame is None
    assert tracker.push(frame).status is TrackingStatus.CONTEXT_INVALID


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


def test_tracker_rebinds_a_proportionally_resized_confirmed_position() -> None:
    board = parse_fen(START)
    tracker = _tracker(board)
    tracker.initialize(_render(board))

    rebound = tracker.rebind_frame_size(_scale_two_times(_render(board)))

    assert rebound.status is TrackingStatus.WATCHING
    assert rebound.board == board
    assert tracker.geometry.frame_size == (432, 480)

    move = _move(board, "h2e2")
    resized_after = _scale_two_times(_render(apply_move(board, move)))
    tracker.push(resized_after)
    tracker.push(resized_after.copy())
    accepted = tracker.push(resized_after.copy())

    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.move == move


def test_tracker_rejects_resize_when_the_position_changed_at_the_same_time() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    tracker = _tracker(board)
    tracker.initialize(_render(board))

    invalid = tracker.rebind_frame_size(
        _scale_two_times(_render(apply_move(board, move)))
    )

    assert invalid.status is TrackingStatus.CONTEXT_INVALID
    assert invalid.board == board
    assert tracker.board == board
    assert tracker.geometry.frame_size == (216, 240)


def test_manual_recovery_after_rejected_resize_rebinds_geometry_and_resumes() -> None:
    board = parse_fen(START)
    first_move = _move(board, "h2e2")
    recovered_board = apply_move(board, first_move)
    resized_recovery_frame = _scale_two_times(_render(recovered_board))
    tracker = _tracker(board)
    tracker.initialize(_render(board))
    tracker.rebind_frame_size(resized_recovery_frame)
    recovered_geometry = _geometry().rebind((432, 480))

    recovered = tracker.recover(
        recovered_board,
        resized_recovery_frame,
        geometry=recovered_geometry,
    )

    assert recovered.status is TrackingStatus.WATCHING
    assert tracker.geometry == recovered_geometry
    second_move = _move(recovered_board, "h7e7")
    resized_after = _scale_two_times(_render(apply_move(recovered_board, second_move)))
    tracker.push(resized_after)
    tracker.push(resized_after.copy())
    accepted = tracker.push(resized_after.copy())

    assert accepted.status is TrackingStatus.ACCEPTED
    assert accepted.move == second_move


def test_manual_recovery_rejects_geometry_that_does_not_match_the_frame() -> None:
    board = parse_fen(START)
    tracker = _tracker(board)
    tracker.initialize(_render(board))
    tracker.invalidate_context()

    with pytest.raises(ValueError, match="frame size"):
        tracker.recover(
            board,
            _scale_two_times(_render(board)),
            geometry=_geometry(),
        )

    assert tracker.push(_render(board)).status is TrackingStatus.CONTEXT_INVALID


def _occupancy_for(board: BoardState, confidence: float = 0.95) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (confidence,) * 90,
        "literal-v1",
    )


class _SequenceOccupancyObserver:
    def __init__(self, values: tuple[OccupancyEvidence, ...]) -> None:
        self._values = list(values)

    @property
    def remaining(self) -> int:
        return len(self._values)

    def observe(
        self,
        _frame: np.ndarray,
        _geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        if not self._values:
            raise AssertionError("occupancy observer received an unexpected call")
        return self._values.pop(0)
