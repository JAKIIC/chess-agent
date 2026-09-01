from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter, StateCommitter
from xiangqi_agent.sync.evidence import (
    MoveProposal,
    MoveSequenceProposal,
    ObservationStatus,
)
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.move_observer import MoveObserver
from xiangqi_agent.sync.sequence_observer import (
    LegalTwoPlyDiffObserver,
    MoveSequenceObserver,
    ReplyConstrainedMoveSequenceObserver,
)
from xiangqi_agent.sync.transition_capture import (
    TransitionCaptureEvidence,
    build_transition_capture_evidence,
)
from xiangqi_agent.vision.change_detection import analyze_frame_change
from xiangqi_agent.vision.geometry import BoardGeometry, GeometryError
from xiangqi_agent.vision.occupancy import OccupancyObserver
from xiangqi_agent.vision.position_validation import validate_fixed_theme_position


class TrackingStatus(StrEnum):
    WATCHING = "watching"
    WAITING_FOR_STABLE = "waiting_for_stable"
    WAITING_FOR_ENDPOINT = "waiting_for_endpoint"
    WAITING_FOR_REPLY = "waiting_for_reply"
    ACCEPTED = "accepted"
    PAUSED_AMBIGUOUS = "paused_ambiguous"
    CONTEXT_INVALID = "context_invalid"
    DESYNCHRONIZED = "desynchronized"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    status: TrackingStatus
    board: BoardState
    moves: tuple[Move, ...] = ()
    observation: MoveProposal | MoveSequenceProposal | None = None
    transition_evidence: TransitionCaptureEvidence | None = None

    @property
    def move(self) -> Move | None:
        return self.moves[0] if len(self.moves) == 1 else None


class StableMoveTracker:
    """Gate visual move inference behind animation settling and ambiguity safety."""

    def __init__(
        self,
        board: BoardState,
        geometry: BoardGeometry,
        observer: MoveObserver,
        committer: StateCommitter | None = None,
        *,
        mode: SyncMode = SyncMode.STRICT_SINGLE,
        sequence_observer: MoveSequenceObserver | None = None,
        required_stable_pairs: int = 2,
        global_threshold: float = 1.5,
        local_threshold: float = 3.0,
        patch_size: int = 32,
        capture_transition_evidence: bool = False,
        occupancy_observer: OccupancyObserver | None = None,
        require_atomic_two_ply: bool = False,
    ) -> None:
        if required_stable_pairs <= 0:
            raise ValueError("required_stable_pairs must be positive")
        if not isinstance(mode, SyncMode):
            raise TypeError("mode must be a SyncMode")
        if not isinstance(capture_transition_evidence, bool):
            raise TypeError("capture_transition_evidence must be a boolean")
        if not isinstance(require_atomic_two_ply, bool):
            raise TypeError("require_atomic_two_ply must be a boolean")
        if require_atomic_two_ply and mode is not SyncMode.HUMAN_VS_AI:
            raise ValueError("atomic two-ply tracking requires human-vs-AI mode")
        self._board = board
        self._geometry = geometry
        self._observer = observer
        self._committer = committer or RuleStateCommitter()
        self._required_stable_pairs = required_stable_pairs
        self._global_threshold = global_threshold
        self._local_threshold = local_threshold
        self._patch_size = patch_size
        self._capture_transition_evidence = capture_transition_evidence
        self._occupancy_observer = occupancy_observer
        self._require_atomic_two_ply = require_atomic_two_ply
        self._mode = mode
        self._sequence_observer = sequence_observer
        if self._mode is SyncMode.HUMAN_VS_AI and self._sequence_observer is None:
            self._sequence_observer = LegalTwoPlyDiffObserver(
                patch_size=patch_size,
                committer=self._committer,
            )
        self._confirmed_frame: NDArray[np.uint8] | None = None
        self._previous_frame: NDArray[np.uint8] | None = None
        self._motion_seen = False
        self._stable_pairs = 0
        self._blocked_status: TrackingStatus | None = None
        self._pending_first_move: Move | None = None

    @property
    def board(self) -> BoardState:
        return self._board

    @property
    def geometry(self) -> BoardGeometry:
        return self._geometry

    def initialize(self, frame: NDArray[np.generic]) -> TrackingUpdate:
        current = _owned_frame(frame)
        self._confirmed_frame = current.copy()
        self._previous_frame = current
        self._motion_seen = False
        self._stable_pairs = 0
        self._blocked_status = None
        self._pending_first_move = None
        return TrackingUpdate(TrackingStatus.WATCHING, self._board)

    def push(
        self,
        frame: NDArray[np.generic],
        *,
        capture_timestamp_ns: int | None = None,
    ) -> TrackingUpdate:
        if capture_timestamp_ns is not None and (
            isinstance(capture_timestamp_ns, bool)
            or not isinstance(capture_timestamp_ns, int)
            or capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be a non-negative integer")
        if self._blocked_status is not None:
            if self._blocked_status in (
                TrackingStatus.PAUSED_AMBIGUOUS,
                TrackingStatus.DESYNCHRONIZED,
                TrackingStatus.MANUAL_RECOVERY_REQUIRED,
            ):
                self._blocked_status = TrackingStatus.MANUAL_RECOVERY_REQUIRED
            return TrackingUpdate(self._blocked_status, self._board)
        if self._confirmed_frame is None or self._previous_frame is None:
            raise RuntimeError("tracker must be initialized with a confirmed frame")

        current = _owned_frame(frame)
        change = analyze_frame_change(
            self._previous_frame,
            current,
            self._geometry,
            global_threshold=self._global_threshold,
            local_threshold=self._local_threshold,
            patch_size=self._patch_size,
        )
        self._previous_frame = current

        if not change.stable:
            self._motion_seen = True
            self._stable_pairs = 0
            return TrackingUpdate(TrackingStatus.WAITING_FOR_STABLE, self._board)
        if not self._motion_seen:
            if self._pending_first_move is not None:
                return TrackingUpdate(TrackingStatus.WAITING_FOR_REPLY, self._board)
            return TrackingUpdate(TrackingStatus.WATCHING, self._board)

        self._stable_pairs += 1
        if self._stable_pairs < self._required_stable_pairs:
            return TrackingUpdate(TrackingStatus.WAITING_FOR_STABLE, self._board)

        decision_started_ns = perf_counter_ns()
        if self._require_atomic_two_ply and self._pending_first_move is not None:
            return self._resolve_pending_reply(
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )

        observation = self._observer.observe(
            self._board,
            self._confirmed_frame,
            current,
            self._geometry,
        )
        if observation.status is ObservationStatus.ACCEPTED:
            if observation.move is None:
                return self._pause(
                    _failed_observation(observation, "proposal_missing_move"),
                    current,
                    decision_started_ns,
                    capture_timestamp_ns,
                )
            if self._require_atomic_two_ply:
                if self._pending_first_move is None:
                    try:
                        self._committer.commit(self._board, observation.move)
                    except ValueError:
                        return self._pause(
                            _failed_observation(observation, "rule_commit_failed"),
                            current,
                            decision_started_ns,
                            capture_timestamp_ns,
                        )
                    return self._wait_for_reply(
                        observation.move,
                        observation,
                        current,
                    )
            else:
                try:
                    verified_after = self._committer.commit(self._board, observation.move)
                except ValueError:
                    return self._pause(
                        _failed_observation(observation, "rule_commit_failed"),
                        current,
                        decision_started_ns,
                        capture_timestamp_ns,
                    )
                transition_evidence = self._build_transition_evidence(
                    observation,
                    current,
                    decision_started_ns,
                    capture_timestamp_ns,
                )
                self._board = verified_after
                self._confirmed_frame = current.copy()
                self._motion_seen = False
                self._stable_pairs = 0
                return TrackingUpdate(
                    TrackingStatus.ACCEPTED,
                    self._board,
                    (observation.move,),
                    observation,
                    transition_evidence,
                )
        if observation.status is ObservationStatus.NO_CHANGE:
            self._confirmed_frame = current.copy()
            self._motion_seen = False
            self._stable_pairs = 0
            self._pending_first_move = None
            return TrackingUpdate(TrackingStatus.WATCHING, self._board)
        if _is_incomplete_endpoint_transition(observation):
            self._motion_seen = True
            self._stable_pairs = 0
            return TrackingUpdate(
                TrackingStatus.WAITING_FOR_ENDPOINT,
                self._board,
                observation=observation,
            )

        if self._can_try_sequence(observation):
            sequence_observer = self._sequence_observer
            if sequence_observer is None:
                return self._pause(
                    observation,
                    current,
                    decision_started_ns,
                    capture_timestamp_ns,
                )
            sequence = sequence_observer.observe(
                self._board,
                self._confirmed_frame,
                current,
                self._geometry,
            )
            resolved = self._commit_sequence(
                sequence,
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )
            if resolved is not None:
                return resolved
            if self._require_atomic_two_ply and self._pending_first_move is None:
                intermediate = self._unique_intermediate_move(current)
                if intermediate is not None:
                    return self._wait_for_reply(intermediate, observation, current)
            return self._pause(
                sequence,
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )

        if self._require_atomic_two_ply and self._pending_first_move is None:
            intermediate = self._unique_intermediate_move(current)
            if intermediate is not None:
                return self._wait_for_reply(intermediate, observation, current)

        return self._pause(
            observation,
            current,
            decision_started_ns,
            capture_timestamp_ns,
        )

    def _resolve_pending_reply(
        self,
        current: NDArray[np.uint8],
        decision_started_ns: int,
        capture_timestamp_ns: int | None,
    ) -> TrackingUpdate:
        pending_first = self._pending_first_move
        sequence_observer = self._sequence_observer
        confirmed = self._confirmed_frame
        if pending_first is None or sequence_observer is None or confirmed is None:
            raise RuntimeError("pending reply requires a sequence observer and baseline")
        if isinstance(sequence_observer, ReplyConstrainedMoveSequenceObserver):
            sequence = sequence_observer.observe_after_first(
                self._board,
                pending_first,
                confirmed,
                current,
                self._geometry,
            )
        else:
            sequence = sequence_observer.observe(
                self._board,
                confirmed,
                current,
                self._geometry,
            )
        resolved = self._commit_sequence(
            sequence,
            current,
            decision_started_ns,
            capture_timestamp_ns,
        )
        if resolved is not None:
            return resolved
        return self._pause(
            _failed_observation(sequence, "intermediate_move_mismatch"),
            current,
            decision_started_ns,
            capture_timestamp_ns,
        )

    def _commit_sequence(
        self,
        sequence: MoveSequenceProposal,
        current: NDArray[np.uint8],
        decision_started_ns: int,
        capture_timestamp_ns: int | None,
    ) -> TrackingUpdate | None:
        if sequence.status is not ObservationStatus.ACCEPTED:
            return None
        sequence_moves = (sequence.moves[0], sequence.moves[1])
        if (
            self._pending_first_move is not None
            and sequence_moves[0] != self._pending_first_move
        ):
            return self._pause(
                _failed_observation(sequence, "intermediate_move_mismatch"),
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )
        if (
            not sequence.evidence.candidates
            or sequence.evidence.candidates[0].moves != sequence_moves
        ):
            return self._pause(
                _failed_observation(sequence, "candidate_evidence_mismatch"),
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )
        try:
            verified_after = self._committer.commit_sequence(
                self._board,
                sequence_moves,
            )
        except (IndexError, ValueError):
            return self._pause(
                _failed_observation(sequence, "sequence_commit_failed"),
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )
        if (
            verified_after.position_id
            != sequence.evidence.candidates[0].final_position_id
        ):
            return self._pause(
                _failed_observation(sequence, "final_position_mismatch"),
                current,
                decision_started_ns,
                capture_timestamp_ns,
            )
        transition_evidence = self._build_transition_evidence(
            sequence,
            current,
            decision_started_ns,
            capture_timestamp_ns,
        )
        self._board = verified_after
        self._confirmed_frame = current.copy()
        self._motion_seen = False
        self._stable_pairs = 0
        self._pending_first_move = None
        return TrackingUpdate(
            TrackingStatus.ACCEPTED,
            self._board,
            sequence_moves,
            sequence,
            transition_evidence,
        )

    def _can_try_sequence(self, observation: MoveProposal) -> bool:
        if self._mode is not SyncMode.HUMAN_VS_AI:
            return False
        if self._pending_first_move is not None:
            return True
        reasons = frozenset(observation.evidence.rejection_reasons)
        return bool(reasons & {"outside_change", "candidate_margin", "candidate_score"})

    def _unique_intermediate_move(
        self,
        current: NDArray[np.uint8],
    ) -> Move | None:
        observer = self._occupancy_observer
        if observer is None:
            return None
        try:
            observed = observer.observe(current, self._geometry).occupied
        except (RuntimeError, TypeError, ValueError):
            return None
        confirmed = tuple(piece != "." for piece in self._board.pieces)
        if observed == confirmed:
            return None
        candidates: list[Move] = []
        for move in legal_moves(self._board):
            try:
                projected = self._committer.commit(self._board, move)
            except ValueError:
                continue
            if observed == tuple(piece != "." for piece in projected.pieces):
                candidates.append(move)
                if len(candidates) > 1:
                    return None
        return candidates[0] if candidates else None

    def _wait_for_reply(
        self,
        move: Move,
        observation: MoveProposal,
        current: NDArray[np.uint8],
    ) -> TrackingUpdate:
        self._pending_first_move = move
        self._previous_frame = current
        self._motion_seen = False
        self._stable_pairs = 0
        return TrackingUpdate(
            TrackingStatus.WAITING_FOR_REPLY,
            self._board,
            observation=observation,
        )

    def _pause(
        self,
        observation: MoveProposal | MoveSequenceProposal,
        current: NDArray[np.uint8],
        decision_started_ns: int,
        capture_timestamp_ns: int | None,
    ) -> TrackingUpdate:
        self._blocked_status = TrackingStatus.PAUSED_AMBIGUOUS
        return TrackingUpdate(
            TrackingStatus.PAUSED_AMBIGUOUS,
            self._board,
            observation=observation,
            transition_evidence=self._build_transition_evidence(
                observation,
                current,
                decision_started_ns,
                capture_timestamp_ns,
            ),
        )

    def _build_transition_evidence(
        self,
        observation: MoveProposal | MoveSequenceProposal,
        current: NDArray[np.uint8],
        decision_started_ns: int,
        capture_timestamp_ns: int | None,
    ) -> TransitionCaptureEvidence | None:
        if not self._capture_transition_evidence:
            return None
        confirmed = self._confirmed_frame
        if confirmed is None:
            raise RuntimeError("transition capture requires a confirmed frame")
        completed_ns = perf_counter_ns()
        latency_origin_ns = (
            capture_timestamp_ns
            if capture_timestamp_ns is not None and capture_timestamp_ns <= completed_ns
            else decision_started_ns
        )
        return build_transition_capture_evidence(
            confirmed,
            current,
            self._geometry,
            observation.evidence.local_differences,
            decision_latency_ms=(completed_ns - latency_origin_ns) / 1_000_000,
            occupancy_observer=self._occupancy_observer,
        )

    def invalidate_context(self) -> TrackingUpdate:
        self._blocked_status = TrackingStatus.CONTEXT_INVALID
        self._confirmed_frame = None
        self._previous_frame = None
        self._motion_seen = False
        self._stable_pairs = 0
        self._pending_first_move = None
        return TrackingUpdate(TrackingStatus.CONTEXT_INVALID, self._board)

    def rebind_frame_size(self, frame: NDArray[np.generic]) -> TrackingUpdate:
        """Adopt a proportional resize only when the confirmed position is unchanged."""
        if self._confirmed_frame is None or self._previous_frame is None:
            raise RuntimeError("tracker must be initialized with a confirmed frame")
        if self._blocked_status is not None:
            return TrackingUpdate(self._blocked_status, self._board)
        current = _owned_frame(frame)
        frame_size = (int(current.shape[1]), int(current.shape[0]))
        try:
            rebound = self._geometry.rebind(frame_size)
        except GeometryError:
            return self.invalidate_context()
        validation = validate_fixed_theme_position(
            self._board,
            self._confirmed_frame,
            self._geometry,
            current,
            rebound,
            patch_size=self._patch_size,
        )
        if not validation.accepted:
            return self.invalidate_context()
        self._geometry = rebound
        self._confirmed_frame = current.copy()
        self._previous_frame = current
        self._motion_seen = False
        self._stable_pairs = 0
        self._pending_first_move = None
        return TrackingUpdate(TrackingStatus.WATCHING, self._board)

    def mark_desynchronized(self) -> TrackingUpdate:
        self._blocked_status = TrackingStatus.DESYNCHRONIZED
        return TrackingUpdate(TrackingStatus.DESYNCHRONIZED, self._board)

    def recover(
        self,
        board: BoardState,
        frame: NDArray[np.generic],
        *,
        geometry: BoardGeometry | None = None,
    ) -> TrackingUpdate:
        if not isinstance(board, BoardState):
            raise TypeError("recovery board must be a BoardState")
        current = _owned_frame(frame)
        recovered_geometry = geometry or self._geometry
        if not isinstance(recovered_geometry, BoardGeometry):
            raise TypeError("recovery geometry must be a BoardGeometry")
        frame_size = (int(current.shape[1]), int(current.shape[0]))
        if recovered_geometry.frame_size != frame_size:
            raise ValueError("recovery geometry frame size must match the recovery frame")
        if recovered_geometry.orientation is not board.orientation:
            raise ValueError("recovery geometry orientation must match the recovery board")
        self._board = board
        self._geometry = recovered_geometry
        self._confirmed_frame = current.copy()
        self._previous_frame = current
        self._motion_seen = False
        self._stable_pairs = 0
        self._blocked_status = None
        self._pending_first_move = None
        return TrackingUpdate(TrackingStatus.WATCHING, self._board)


def _owned_frame(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    pixels = np.asarray(frame)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("frame must be a BGRA uint8 image")
    return np.array(pixels, dtype=np.uint8, copy=True, order="C")


def _is_incomplete_endpoint_transition(observation: MoveProposal) -> bool:
    reasons = frozenset(observation.evidence.rejection_reasons)
    if "semantic_noop" in reasons:
        return True
    source_missing = "source_change" in reasons
    destination_missing = "destination_change" in reasons
    return source_missing != destination_missing


def _failed_observation(
    observation: MoveProposal | MoveSequenceProposal,
    reason: str,
) -> MoveProposal | MoveSequenceProposal:
    reasons = observation.evidence.rejection_reasons
    updated_reasons = reasons if reason in reasons else (*reasons, reason)
    if isinstance(observation, MoveProposal):
        return MoveProposal(
            status=ObservationStatus.AMBIGUOUS,
            move=None,
            evidence_score=0.0,
            evidence=replace(
                observation.evidence,
                rejection_reasons=updated_reasons,
            ),
        )
    return MoveSequenceProposal(
        status=ObservationStatus.AMBIGUOUS,
        moves=(),
        evidence_score=0.0,
        evidence=replace(
            observation.evidence,
            rejection_reasons=updated_reasons,
        ),
    )
