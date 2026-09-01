from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.unit.diagnostics.test_stage_c_quarantine import (
    START,
    _crops,
    _event,
    _two_ply_final,
)
from xiangqi_agent.diagnostics.stage_c_live_capture import (
    StageCTerminalEventError,
    StageCTerminalEventWriter,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import QuarantineEventLoader
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.evidence import (
    MoveSequenceEvidence,
    MoveSequenceProposal,
    ObservationStatus,
    SequenceCandidateEvidence,
)
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.transition_capture import (
    TransitionCaptureEvidence,
    TransitionPointEvidence,
)
from xiangqi_agent.vision.geometry import parse_normalized_quad

QUAD = parse_normalized_quad("0.05,0.05;0.95,0.05;0.95,0.95;0.05,0.95")


def _terminal_update() -> LiveSyncUpdate:
    event = _event()
    first = next(move for move in legal_moves(START) if move.uci == "h2e2")
    middle = apply_move(START, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    final = _two_ply_final(START)
    recorded = event.candidates[0]
    candidate = SequenceCandidateEvidence(
        moves=(first, second),
        changed_points=recorded.changed_points,
        expected_change_floor=recorded.expected_change_floor,
        unexpected_difference=recorded.unexpected_difference,
        maximum_template_distance=recorded.maximum_template_distance,
        minimum_template_margin=recorded.minimum_template_margin,
        minimum_template_confidence=recorded.minimum_template_confidence,
        score=recorded.score,
        final_position_id=recorded.final_position_id,
    )
    observation = MoveSequenceProposal(
        ObservationStatus.ACCEPTED,
        (first, second),
        candidate.score,
        MoveSequenceEvidence(
            (candidate,),
            event.local_differences,
            (),
            event.feature_version,
        ),
    )
    point_crops = _crops(event.changed_points)
    transition = TransitionCaptureEvidence(
        event.changed_points,
        event.local_differences,
        tuple(
            TransitionPointEvidence(crop.point_index, crop.before, crop.after)
            for crop in point_crops
        ),
        event.decision_latency_ms,
        event.before_occupancy,
        event.after_occupancy,
    )
    return LiveSyncUpdate(
        status=LiveSyncStatus.MOVE_ACCEPTED,
        board=final,
        message="accepted",
        moves=(first, second),
        observation=observation,
        before_position_id=START.position_id,
        after_position_id=final.position_id,
        sync_mode=SyncMode.HUMAN_VS_AI,
        frame_size=(216, 240),
        point_count=90,
        transition_evidence=transition,
    )


def test_terminal_writer_persists_only_quarantine_evidence(tmp_path: Path) -> None:
    writer = StageCTerminalEventWriter(tmp_path / ".local")

    event_dir = writer.record(
        _terminal_update(),
        board=START,
        quad=QUAD,
        session_id="session-ui",
        event_id="event-ui",
        client_size=(216, 240),
        generation_id=3,
    )
    loaded = QuarantineEventLoader().load(event_dir)

    assert loaded.metadata.event_id == "event-ui"
    assert loaded.metadata.session_id == "session-ui"
    assert loaded.metadata.capture_context.generation_id == 3
    assert loaded.metadata.observed_moves_uci == ("h2e2", "h7e7")
    assert all(crop.before.shape == (48, 48, 4) for crop in loaded.crops)
    assert not any(path.name.startswith("frame") for path in event_dir.iterdir())


def test_terminal_writer_rejects_missing_occupancy_without_writing(tmp_path: Path) -> None:
    writer = StageCTerminalEventWriter(tmp_path / ".local")
    update = _terminal_update()
    assert update.transition_evidence is not None
    update = replace(
        update,
        transition_evidence=replace(
            update.transition_evidence,
            before_occupancy=None,
            after_occupancy=None,
        ),
    )

    with pytest.raises(StageCTerminalEventError, match="occupancy"):
        writer.record(
            update,
            board=START,
            quad=QUAD,
            session_id="session-ui",
            event_id="event-ui",
            client_size=(216, 240),
            generation_id=3,
        )

    assert not (tmp_path / ".local" / "stage-c-quarantine").exists()
