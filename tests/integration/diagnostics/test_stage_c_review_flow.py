from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

import numpy as np
import pytest
from numpy.typing import NDArray

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.diagnostics.stage_c_gate import (
    DEFAULT_STAGE_C_FEATURE_VERSION,
    DEFAULT_STAGE_C_THRESHOLD_PROFILE,
)
from xiangqi_agent.diagnostics.stage_c_live_capture import StageCTerminalEventWriter
from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService
from xiangqi_agent.diagnostics.stage_c_replay import (
    HumanAiStageCReplayer,
    HumanAiStageCSampleLoader,
    StageCSampleIntegrityError,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewOutcome,
    StageCReviewService,
    StageCReviewStore,
)
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import ReviewedStageCSampleV2
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.live_session import LiveSyncSession, LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.sequence_gate import SequenceDecisionGate
from xiangqi_agent.vision.geometry import BoardGeometry, parse_normalized_quad
from xiangqi_agent.vision.occupancy import OccupancyEvidence

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
RECAPTURE_FEN = "r3k4/9/9/9/r3p4/9/9/9/9/R3K4 w"
CELL = 24
FRAME_SIZE = (216, 240)
QUAD = parse_normalized_quad(
    "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540"
)
PALETTE = {
    symbol: index * 15
    for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)
}


@dataclass(frozen=True, slots=True)
class _PromotedFlow:
    update: LiveSyncUpdate
    final_board: BoardState
    quarantine_root: Path
    review_root: Path
    sample_dir: Path


class _DeterministicOccupancyObserver:
    def __init__(self, evidence: tuple[OccupancyEvidence, ...]) -> None:
        self._evidence = list(evidence)

    def observe(
        self,
        _frame: NDArray[np.uint8],
        _geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        if not self._evidence:
            raise AssertionError("unexpected occupancy observation")
        return self._evidence.pop(0)


def test_confirmed_observer_candidate_survives_source_cleanup_and_replays(
    tmp_path: Path,
) -> None:
    board = parse_fen(START_FEN)
    moves = ("h2e2", "h7e7")

    flow = _promote_flow(
        tmp_path,
        board,
        visual_moves=moves,
        actual_moves=moves,
        event_id="accepted-event",
        expected_status=LiveSyncStatus.MOVE_ACCEPTED,
        review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
    )

    assert tuple(move.uci for move in flow.update.moves) == moves
    loaded = HumanAiStageCSampleLoader().load(flow.sample_dir)
    assert isinstance(loaded.metadata, ReviewedStageCSampleV2)
    result = _replayer().replay(flow.sample_dir)
    assert result.correct_accept
    assert not result.false_accept
    assert result.replayed_moves_uci == moves

    _delete_source_evidence(flow)

    assert not flow.quarantine_root.exists()
    assert not flow.review_root.exists()
    assert isinstance(
        HumanAiStageCSampleLoader().load(flow.sample_dir).metadata,
        ReviewedStageCSampleV2,
    )
    assert _replayer().replay(flow.sample_dir).correct_accept


def test_rejected_visual_candidate_can_be_corrected_and_replays_as_missed_valid(
    tmp_path: Path,
) -> None:
    board = parse_fen(RECAPTURE_FEN)
    actual_moves = ("a0a5", "a9a5")

    flow = _promote_flow(
        tmp_path,
        board,
        visual_moves=("a0a1", "a9a6"),
        actual_moves=actual_moves,
        event_id="corrected-event",
        expected_status=LiveSyncStatus.PAUSED_AMBIGUOUS,
        review_outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
        unrelated_point=40,
    )

    assert flow.update.moves == ()
    result = _replayer().replay(flow.sample_dir)
    assert result.ground_truth_moves_uci == actual_moves
    assert result.review_outcome is StageCReviewOutcome.LEGAL_MOVE_CORRECTION
    assert not result.accepted
    assert result.missed_valid
    assert not result.false_accept

    _delete_source_evidence(flow)

    loaded = HumanAiStageCSampleLoader().load(flow.sample_dir)
    assert isinstance(loaded.metadata, ReviewedStageCSampleV2)
    assert _replayer().replay(flow.sample_dir).missed_valid


@pytest.mark.parametrize(
    "sidecar",
    ("source-event-manifest.json", "review-manifest.json"),
)
def test_each_v2_provenance_sidecar_is_required_after_source_cleanup(
    tmp_path: Path,
    sidecar: str,
) -> None:
    board = parse_fen(START_FEN)
    moves = ("h2e2", "h7e7")
    flow = _promote_flow(
        tmp_path,
        board,
        visual_moves=moves,
        actual_moves=moves,
        event_id=f"tamper-{sidecar.split('-', maxsplit=1)[0]}",
        expected_status=LiveSyncStatus.MOVE_ACCEPTED,
        review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
    )
    _delete_source_evidence(flow)
    path = flow.sample_dir / sidecar
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(StageCSampleIntegrityError, match="provenance"):
        _replayer().replay(flow.sample_dir)


def _promote_flow(
    tmp_path: Path,
    board: BoardState,
    *,
    visual_moves: tuple[str, str],
    actual_moves: tuple[str, str],
    event_id: str,
    expected_status: LiveSyncStatus,
    review_outcome: StageCReviewOutcome,
    unrelated_point: int | None = None,
) -> _PromotedFlow:
    visual_final = _project(board, visual_moves)
    actual_final = _project(board, actual_moves)
    frame = _render(visual_final)
    if unrelated_point is not None:
        row, column = divmod(unrelated_point, 9)
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = 255
    update = _capture_terminal_update(
        board,
        frame,
        actual_final,
        expected_status,
    )

    local_root = tmp_path / ".local"
    quarantine_root = local_root / "stage-c-quarantine"
    review_root = local_root / "stage-c-reviews"
    reviewed_root = local_root / "stage-c-reviewed"
    event_dir = StageCTerminalEventWriter(local_root).record(
        update,
        board=board,
        quad=QUAD,
        session_id="integration-session",
        event_id=event_id,
        client_size=FRAME_SIZE,
        generation_id=1,
    )
    reviewed_moves = (
        tuple(move.uci for move in update.moves)
        if review_outcome is StageCReviewOutcome.CANDIDATE_CONFIRMED
        else actual_moves
    )
    review_path = StageCReviewService(
        StageCReviewStore(review_root, enabled=True)
    ).submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.VALID_TWO_PLY,
            moves_uci=reviewed_moves,
            scenario=None,
            review_outcome=review_outcome,
        ),
    )
    sample_dir = StageCPromotionService().promote(
        event_dir,
        review_path,
        reviewed_root,
    )
    return _PromotedFlow(
        update,
        actual_final,
        quarantine_root,
        review_root,
        sample_dir,
    )


def _capture_terminal_update(
    board: BoardState,
    final_frame: NDArray[np.uint8],
    occupancy_final: BoardState,
    expected_status: LiveSyncStatus,
) -> LiveSyncUpdate:
    source = FakeFrameSource(hwnd=42)
    updates: list[LiveSyncUpdate] = []
    observer = _DeterministicOccupancyObserver(
        (
            _occupancy(board),
            _occupancy(board),
            _occupancy(occupancy_final),
        )
    )
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
        occupancy_observer=observer,
        require_matching_baseline=True,
    )
    try:
        session.start()
        baseline = _render(board)
        source.push(baseline, 0)
        source.push(baseline.copy(), 50_000_000)
        source.push(baseline.copy(), 100_000_000)
        _wait_until(
            lambda: any(
                update.status is LiveSyncStatus.BASELINE_READY for update in updates
            )
        )

        source.push(final_frame, 150_000_000)
        source.push(final_frame.copy(), 260_000_000)
        _wait_until(
            lambda: any(update.status is expected_status for update in updates)
        )
        terminal = next(
            update for update in updates if update.status is expected_status
        )
        assert terminal.transition_evidence is not None
        return terminal
    finally:
        session.close()


def _project(board: BoardState, moves: tuple[str, ...]) -> BoardState:
    projected = board
    for uci in moves:
        move = _move(projected, uci)
        projected = apply_move(projected, move)
    return projected


def _move(board: BoardState, uci: str) -> Move:
    return next(move for move in legal_moves(board) if move.uci == uci)


def _render(board: BoardState) -> NDArray[np.uint8]:
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = PALETTE[symbol]
    return frame


def _occupancy(board: BoardState) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (0.95,) * 90,
        "synthetic-occupancy-v1",
    )


def _replayer() -> HumanAiStageCReplayer:
    return HumanAiStageCReplayer(
        SequenceDecisionGate(DEFAULT_STAGE_C_THRESHOLD_PROFILE),
        feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
    )


def _delete_source_evidence(flow: _PromotedFlow) -> None:
    shutil.rmtree(flow.quarantine_root)
    shutil.rmtree(flow.review_root)


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise AssertionError("timed out waiting for the terminal Stage C update")
