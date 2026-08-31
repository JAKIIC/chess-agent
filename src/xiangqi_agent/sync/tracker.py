from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.sync.committer import RuleStateCommitter, StateCommitter
from xiangqi_agent.sync.evidence import MoveProposal, ObservationStatus
from xiangqi_agent.sync.move_observer import MoveObserver
from xiangqi_agent.vision.change_detection import analyze_frame_change
from xiangqi_agent.vision.geometry import BoardGeometry, GeometryError
from xiangqi_agent.vision.position_validation import validate_fixed_theme_position


class TrackingStatus(StrEnum):
    WATCHING = "watching"
    WAITING_FOR_STABLE = "waiting_for_stable"
    WAITING_FOR_ENDPOINT = "waiting_for_endpoint"
    ACCEPTED = "accepted"
    PAUSED_AMBIGUOUS = "paused_ambiguous"
    CONTEXT_INVALID = "context_invalid"
    DESYNCHRONIZED = "desynchronized"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    status: TrackingStatus
    board: BoardState
    move: Move | None = None
    observation: MoveProposal | None = None


class StableMoveTracker:
    """Gate visual move inference behind animation settling and ambiguity safety."""

    def __init__(
        self,
        board: BoardState,
        geometry: BoardGeometry,
        observer: MoveObserver,
        committer: StateCommitter | None = None,
        *,
        required_stable_pairs: int = 2,
        global_threshold: float = 1.5,
        local_threshold: float = 3.0,
        patch_size: int = 32,
    ) -> None:
        if required_stable_pairs <= 0:
            raise ValueError("required_stable_pairs must be positive")
        self._board = board
        self._geometry = geometry
        self._observer = observer
        self._committer = committer or RuleStateCommitter()
        self._required_stable_pairs = required_stable_pairs
        self._global_threshold = global_threshold
        self._local_threshold = local_threshold
        self._patch_size = patch_size
        self._confirmed_frame: NDArray[np.uint8] | None = None
        self._previous_frame: NDArray[np.uint8] | None = None
        self._motion_seen = False
        self._stable_pairs = 0
        self._blocked_status: TrackingStatus | None = None

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
        return TrackingUpdate(TrackingStatus.WATCHING, self._board)

    def push(self, frame: NDArray[np.generic]) -> TrackingUpdate:
        if self._confirmed_frame is None or self._previous_frame is None:
            raise RuntimeError("tracker must be initialized with a confirmed frame")
        if self._blocked_status is not None:
            if self._blocked_status in (
                TrackingStatus.PAUSED_AMBIGUOUS,
                TrackingStatus.DESYNCHRONIZED,
                TrackingStatus.MANUAL_RECOVERY_REQUIRED,
            ):
                self._blocked_status = TrackingStatus.MANUAL_RECOVERY_REQUIRED
            return TrackingUpdate(self._blocked_status, self._board)

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
            return TrackingUpdate(TrackingStatus.WATCHING, self._board)

        self._stable_pairs += 1
        if self._stable_pairs < self._required_stable_pairs:
            return TrackingUpdate(TrackingStatus.WAITING_FOR_STABLE, self._board)

        observation = self._observer.observe(
            self._board,
            self._confirmed_frame,
            current,
            self._geometry,
        )
        if observation.status is ObservationStatus.ACCEPTED:
            if observation.move is None:
                return self._pause(observation)
            try:
                verified_after = self._committer.commit(self._board, observation.move)
            except ValueError:
                return self._pause(observation)
            self._board = verified_after
            self._confirmed_frame = current.copy()
            self._motion_seen = False
            self._stable_pairs = 0
            return TrackingUpdate(
                TrackingStatus.ACCEPTED,
                self._board,
                observation.move,
                observation,
            )
        if observation.status is ObservationStatus.NO_CHANGE:
            self._confirmed_frame = current.copy()
            self._motion_seen = False
            self._stable_pairs = 0
            return TrackingUpdate(TrackingStatus.WATCHING, self._board)
        if _is_incomplete_endpoint_transition(observation):
            self._motion_seen = True
            self._stable_pairs = 0
            return TrackingUpdate(
                TrackingStatus.WAITING_FOR_ENDPOINT,
                self._board,
                observation=observation,
            )

        return self._pause(observation)

    def _pause(self, observation: MoveProposal) -> TrackingUpdate:
        self._blocked_status = TrackingStatus.PAUSED_AMBIGUOUS
        return TrackingUpdate(
            TrackingStatus.PAUSED_AMBIGUOUS,
            self._board,
            observation=observation,
        )

    def invalidate_context(self) -> TrackingUpdate:
        self._blocked_status = TrackingStatus.CONTEXT_INVALID
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
