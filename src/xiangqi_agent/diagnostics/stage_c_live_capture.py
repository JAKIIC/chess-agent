from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.stage_c_gate import DEFAULT_STAGE_C_THRESHOLD_PROFILE
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    QuarantinedStageCEventV1,
    QuarantineEventRecorder,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCObservedStatus,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.sync.evidence import MoveSequenceProposal, SequenceCandidateEvidence
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.transition_capture import TransitionCaptureEvidence
from xiangqi_agent.vision.geometry import NormalizedQuad

_ALLOWED_REJECTION_REASONS = frozenset(
    {
        "candidate_margin",
        "candidate_score",
        "expected_change",
        "no_legal_candidates",
        "outside_change",
        "template_confidence",
        "template_distance",
        "template_margin",
        "template_unavailable",
    }
)


class StageCTerminalEventError(RuntimeError):
    """A terminal sync update cannot be persisted as privacy-safe evidence."""


class StageCTerminalEventWriter:
    def __init__(self, local_root: Path) -> None:
        if not isinstance(local_root, Path):
            raise TypeError("local_root must be a Path")
        if local_root.name != ".local":
            raise ValueError("Stage C runtime root must be named .local")
        self._local_root = local_root
        self._output_root = local_root / "stage-c-quarantine"

    def record(
        self,
        update: LiveSyncUpdate,
        *,
        board: BoardState,
        quad: NormalizedQuad,
        session_id: str,
        event_id: str,
        client_size: tuple[int, int],
        generation_id: int,
    ) -> Path:
        if not isinstance(update, LiveSyncUpdate):
            raise TypeError("update must be a LiveSyncUpdate")
        if not isinstance(board, BoardState):
            raise TypeError("board must be a BoardState")
        if not isinstance(quad, NormalizedQuad):
            raise TypeError("quad must be a NormalizedQuad")
        evidence = update.transition_evidence
        observation = update.observation
        if evidence is None:
            raise StageCTerminalEventError(
                "terminal event did not contain transition evidence"
            )
        if evidence.before_occupancy is None or evidence.after_occupancy is None:
            raise StageCTerminalEventError(
                "terminal event did not contain both occupancy snapshots"
            )
        if not isinstance(observation, MoveSequenceProposal):
            raise StageCTerminalEventError(
                "terminal event did not exercise the frozen two-ply decision gate"
            )

        if update.status is LiveSyncStatus.MOVE_ACCEPTED:
            if len(update.moves) != 2 or update.after_position_id is None:
                raise StageCTerminalEventError(
                    "accepted Stage C evidence must contain one atomic two-ply event"
                )
            observed_status = StageCObservedStatus.ACCEPTED
            observed_moves = tuple(move.uci for move in update.moves)
            observed_final = update.after_position_id
        elif update.status is LiveSyncStatus.PAUSED_AMBIGUOUS:
            observed_status = StageCObservedStatus.REJECTED
            observed_moves = ()
            observed_final = board.position_id
        else:
            raise StageCTerminalEventError("update is not a Stage C terminal event")

        rejection_reasons = observation.evidence.rejection_reasons
        if (
            update.status is LiveSyncStatus.PAUSED_AMBIGUOUS
            and not set(rejection_reasons) <= _ALLOWED_REJECTION_REASONS
        ):
            raise StageCTerminalEventError(
                "internal tracker failure cannot become quarantine evidence"
            )
        context = _capture_context(
            evidence,
            quad=quad,
            board=board,
            frame_size=update.frame_size,
            client_size=client_size,
            generation_id=generation_id,
        )
        event = QuarantinedStageCEventV1(
            event_id=event_id,
            session_id=session_id,
            created_at_utc=_timestamp_from_update(update),
            confirmed_fen=board.fen,
            confirmed_position_id=board.position_id,
            observed_status=observed_status,
            observed_moves_uci=observed_moves,
            observed_final_position_id=observed_final,
            side_to_move=board.side_to_move,
            orientation=board.orientation,
            changed_points=evidence.changed_points,
            local_differences=evidence.local_differences,
            candidates=tuple(
                _candidate_record(candidate)
                for candidate in observation.evidence.candidates[:2]
            ),
            rejection_reasons=rejection_reasons,
            capture_context=context,
            feature_version=observation.evidence.feature_version,
            threshold_profile_version=DEFAULT_STAGE_C_THRESHOLD_PROFILE.profile_version,
            decision_latency_ms=evidence.decision_latency_ms,
            before_occupancy=evidence.before_occupancy,
            after_occupancy=evidence.after_occupancy,
        )
        crops = tuple(
            TransitionPointCrops(crop.point_index, crop.before, crop.after)
            for crop in evidence.crops
        )
        return QuarantineEventRecorder(self._output_root, enabled=True).record(
            event,
            crops,
        )


def _candidate_record(candidate: SequenceCandidateEvidence) -> StageCCandidateRecord:
    return StageCCandidateRecord(
        moves_uci=(candidate.moves[0].uci, candidate.moves[1].uci),
        changed_points=candidate.changed_points,
        expected_change_floor=candidate.expected_change_floor,
        unexpected_difference=candidate.unexpected_difference,
        maximum_template_distance=candidate.maximum_template_distance,
        minimum_template_margin=candidate.minimum_template_margin,
        minimum_template_confidence=candidate.minimum_template_confidence,
        score=candidate.score,
        final_position_id=candidate.final_position_id,
    )


def _capture_context(
    evidence: TransitionCaptureEvidence,
    *,
    quad: NormalizedQuad,
    board: BoardState,
    frame_size: tuple[int, int] | None,
    client_size: tuple[int, int],
    generation_id: int,
) -> CaptureContext:
    if frame_size is None:
        raise StageCTerminalEventError("terminal event did not expose a frame size")
    if (
        len(client_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in client_size
        )
    ):
        raise StageCTerminalEventError("client size is invalid")
    geometry_revision = sha256(
        repr((quad.points, frame_size, board.orientation.value)).encode("ascii")
    ).hexdigest()[:16]
    theme_summary = np.asarray(
        [crop.before[..., :3].mean(axis=(0, 1)) for crop in evidence.crops],
        dtype=np.float32,
    )
    theme_fingerprint = sha256(theme_summary.tobytes()).hexdigest()[:16]
    return CaptureContext(
        wgc_size=frame_size,
        client_size=client_size,
        dpi_scale=frame_size[0] / client_size[0],
        geometry_revision=geometry_revision,
        theme_fingerprint=theme_fingerprint,
        generation_id=generation_id,
    )


def _timestamp_from_update(update: LiveSyncUpdate) -> str:
    evidence = update.transition_evidence
    if evidence is None:
        raise StageCTerminalEventError("terminal event has no timing evidence")
    # The transition evidence exposes elapsed time rather than wall-clock identity.
    # A local UTC timestamp is added only to support retention and immutable ordering.
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
