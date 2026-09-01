from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from queue import Empty
from threading import Condition, Lock, Thread, current_thread
from time import perf_counter_ns

from xiangqi_agent.capture.adaptive_sampling import (
    AdaptiveBurstSampler,
    FrameSizeChangedError,
)
from xiangqi_agent.capture.protocol import (
    BurstFrameSource,
    CaptureClosedError,
    CaptureFrame,
    FrameSource,
)
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.sync.evidence import MoveProposal, MoveSequenceProposal
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.sequence_observer import LegalTwoPlyDiffObserver
from xiangqi_agent.sync.tracker import StableMoveTracker, TrackingStatus, TrackingUpdate
from xiangqi_agent.sync.transition_capture import TransitionCaptureEvidence
from xiangqi_agent.vision.change_detection import FrameStabilityDetector
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.occupancy import OccupancyObserver, compare_occupancy


class LiveSyncStatus(StrEnum):
    CONNECTING = "connecting"
    BASELINE_READY = "baseline_ready"
    WATCHING = "watching"
    WAITING_FOR_STABLE = "waiting_for_stable"
    WAITING_FOR_ENDPOINT = "waiting_for_endpoint"
    WAITING_FOR_REPLY = "waiting_for_reply"
    MOVE_ACCEPTED = "move_accepted"
    GEOMETRY_REBOUND = "geometry_rebound"
    PAUSED_AMBIGUOUS = "paused_ambiguous"
    CONTEXT_INVALID = "context_invalid"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"
    RECOVERY_PENDING = "recovery_pending"
    RECOVERY_ACCEPTED = "recovery_accepted"
    CLOSED = "closed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LiveSyncUpdate:
    status: LiveSyncStatus
    board: BoardState
    message: str
    moves: tuple[Move, ...] = ()
    observation: MoveProposal | MoveSequenceProposal | None = None
    before_position_id: str | None = None
    after_position_id: str | None = None
    sync_mode: SyncMode = SyncMode.STRICT_SINGLE
    recovery_id: int | None = None
    frame_size: tuple[int, int] | None = None
    point_count: int = 0
    transition_evidence: TransitionCaptureEvidence | None = None

    @property
    def move(self) -> Move | None:
        return self.moves[0] if len(self.moves) == 1 else None


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


@dataclass(slots=True)
class _PendingRecovery:
    request_id: int
    board: BoardState
    quad: NormalizedQuad
    cutoff_timestamp_ns: int
    geometry: BoardGeometry
    detector: FrameStabilityDetector


class _CoalescingEventQueue:
    """Bound full-frame memory while keeping the newest capture evidence."""

    def __init__(self, *, max_frames: int = 3) -> None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
            raise ValueError("max_frames must be a positive integer")
        self._max_frames = max_frames
        self._frames: deque[CaptureFrame] = deque()
        self._terminal: CaptureClosedError | _Stop | None = None
        self._accepting = True
        self._condition = Condition()

    @property
    def pending_frame_count(self) -> int:
        with self._condition:
            return len(self._frames)

    def put_frame(self, frame: CaptureFrame) -> None:
        with self._condition:
            if not self._accepting:
                return
            if len(self._frames) == self._max_frames:
                self._frames.popleft()
            self._frames.append(frame)
            self._condition.notify()

    def put_terminal(self, event: CaptureClosedError | _Stop) -> None:
        with self._condition:
            if self._terminal is None:
                self._terminal = event
            self._accepting = False
            self._frames.clear()
            self._condition.notify_all()

    def clear_frames(self) -> None:
        with self._condition:
            self._frames.clear()

    def get(self, *, timeout: float) -> CaptureFrame | CaptureClosedError | _Stop:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._terminal is not None or bool(self._frames),
                timeout=timeout,
            )
            if not ready:
                raise Empty
            if self._terminal is not None:
                return self._terminal
            return self._frames.popleft()


class LiveSyncSession:
    """Continuously turn stable capture frames into rule-verified board updates."""

    def __init__(
        self,
        source: FrameSource,
        board: BoardState,
        quad: NormalizedQuad,
        *,
        on_update: Callable[[LiveSyncUpdate], None] | None = None,
        sync_mode: SyncMode = SyncMode.STRICT_SINGLE,
        steady_fps: int = 2,
        settle_ms: int = 400,
        stable_pairs: int = 2,
        patch_size: int = 48,
        capture_transition_evidence: bool = False,
        occupancy_observer: OccupancyObserver | None = None,
        require_matching_baseline: bool = False,
        baseline_minimum_confidence: float = 0.65,
        require_atomic_two_ply: bool = False,
    ) -> None:
        if stable_pairs <= 0:
            raise ValueError("stable_pairs must be positive")
        if not isinstance(sync_mode, SyncMode):
            raise TypeError("sync_mode must be a SyncMode")
        if not isinstance(capture_transition_evidence, bool):
            raise TypeError("capture_transition_evidence must be a boolean")
        if not isinstance(require_matching_baseline, bool):
            raise TypeError("require_matching_baseline must be a boolean")
        if not isinstance(require_atomic_two_ply, bool):
            raise TypeError("require_atomic_two_ply must be a boolean")
        if require_atomic_two_ply and sync_mode is not SyncMode.HUMAN_VS_AI:
            raise ValueError("atomic two-ply tracking requires human-vs-AI mode")
        if require_matching_baseline and occupancy_observer is None:
            raise ValueError("matching baseline requires an occupancy observer")
        if isinstance(baseline_minimum_confidence, bool) or not isinstance(
            baseline_minimum_confidence,
            (int, float),
        ):
            raise TypeError("baseline_minimum_confidence must be a number")
        if not isfinite(baseline_minimum_confidence) or not 0.0 <= baseline_minimum_confidence <= 1.0:
            raise ValueError("baseline_minimum_confidence must be between zero and one")
        self._source = source
        self._board = board
        self._quad = quad
        self._on_update = on_update or _ignore_update
        self._sync_mode = sync_mode
        self._steady_fps = steady_fps
        self._settle_ms = settle_ms
        self._stable_pairs = stable_pairs
        self._patch_size = patch_size
        self._capture_transition_evidence = capture_transition_evidence
        self._occupancy_observer = occupancy_observer
        self._require_matching_baseline = require_matching_baseline
        self._baseline_minimum_confidence = float(baseline_minimum_confidence)
        self._require_atomic_two_ply = require_atomic_two_ply
        self._events = _CoalescingEventQueue(max_frames=3)
        self._lock = Lock()
        self._processing_lock = Lock()
        self._started = False
        self._closed = False
        self._finalized = False
        self._closed_emitted = False
        self._paused_for_recovery = False
        self._last_status: LiveSyncStatus | None = None
        self._thread: Thread | None = None
        self._tracker: StableMoveTracker | None = None
        self._sampler: AdaptiveBurstSampler | None = None
        self._latest_frame: CaptureFrame | None = None
        self._pending_recovery: _PendingRecovery | None = None
        self._next_recovery_id = 0

    @property
    def board(self) -> BoardState:
        with self._lock:
            return self._board

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("live sync session has already started")
            if self._closed:
                raise RuntimeError("live sync session is closed")
            self._started = True
        self._emit(LiveSyncStatus.CONNECTING, "waiting for a stable board frame")
        try:
            self._source.start(self._events.put_frame, self._events.put_terminal)
        except (OSError, RuntimeError, ValueError) as exc:
            self._emit(LiveSyncStatus.ERROR, str(exc))
            self._finalize("live sync source failed to start")
            return
        worker = Thread(target=self._run, name="xiangqi-live-sync", daemon=True)
        self._thread = worker
        worker.start()

    def close(self) -> None:
        with self._lock:
            if self._finalized:
                return
            first_request = not self._closed
            self._closed = True
            thread = self._thread
        if first_request:
            self._events.put_terminal(_Stop())
        try:
            self._source.close()
        except (OSError, RuntimeError):
            pass
        finally:
            if thread is not None and thread is not current_thread():
                thread.join(timeout=2.0)
            with self._lock:
                if thread is None or not thread.is_alive():
                    self._finalized = True
                    self._release_frame_state_locked()
            self._emit_closed("live sync session closed")

    def recover(
        self,
        board: BoardState,
        *,
        quad: NormalizedQuad | None = None,
    ) -> int:
        with self._processing_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("live sync session is closed")
                if self._tracker is None or self._sampler is None or self._latest_frame is None:
                    raise RuntimeError("live sync baseline is not ready")
                latest = self._latest_frame
                recovered_quad = quad or self._quad
                geometry = BoardGeometry.from_quad(
                    recovered_quad,
                    latest.size,
                    board.orientation,
                )
                self._next_recovery_id += 1
                recovery_id = self._next_recovery_id
                self._pending_recovery = _PendingRecovery(
                    request_id=recovery_id,
                    board=board,
                    quad=recovered_quad,
                    cutoff_timestamp_ns=latest.timestamp_ns,
                    geometry=geometry,
                    detector=FrameStabilityDetector(
                        geometry,
                        required_stable_pairs=self._stable_pairs,
                    ),
                )
                self._paused_for_recovery = True
            self._events.clear_frames()
        self._emit(
            LiveSyncStatus.RECOVERY_PENDING,
            "manual recovery is waiting for a fresh stable frame",
            frame_size=latest.size,
            point_count=90,
            recovery_id=recovery_id,
        )
        return recovery_id

    def _run(self) -> None:
        geometry: BoardGeometry | None = None
        detector: FrameStabilityDetector | None = None
        tracker: StableMoveTracker | None = None
        sampler: AdaptiveBurstSampler | None = None
        try:
            while not self._is_closed():
                timeout = _queue_timeout(sampler)
                try:
                    event = self._events.get(timeout=timeout)
                except Empty:
                    if sampler is not None and tracker is not None:
                        with self._processing_lock:
                            if (
                                self._is_recovery_pending()
                                or self._is_paused_for_recovery()
                            ):
                                continue
                            self._process_samples(
                                sampler.on_clock(perf_counter_ns()),
                                sampler,
                                tracker,
                            )
                    continue
                if isinstance(event, _Stop):
                    return
                if isinstance(event, CaptureClosedError):
                    self._emit_closed(str(event))
                    return
                with self._lock:
                    self._latest_frame = event

                if tracker is None or sampler is None:
                    if geometry is None or geometry.frame_size != event.size:
                        geometry = BoardGeometry.from_quad(
                            self._quad,
                            event.size,
                            self.board.orientation,
                        )
                        detector = FrameStabilityDetector(
                            geometry,
                            required_stable_pairs=self._stable_pairs,
                        )
                    if detector is None:
                        raise RuntimeError("baseline detector was not initialized")
                    change = detector.update(event.bgra)
                    if change is None or not change.stable:
                        continue
                    if self._require_matching_baseline:
                        occupancy_observer = self._occupancy_observer
                        if occupancy_observer is None:
                            raise RuntimeError("matching baseline lost its occupancy observer")
                        comparison = compare_occupancy(
                            occupancy_observer.observe(event.bgra, geometry),
                            self.board,
                            minimum_confidence=self._baseline_minimum_confidence,
                        )
                        if not comparison.accepted:
                            self._emit(
                                LiveSyncStatus.CONTEXT_INVALID,
                                "target board is not visibly consistent with the confirmed position",
                                frame_size=event.size,
                                point_count=90,
                            )
                            continue
                    tracker = StableMoveTracker(
                        self.board,
                        geometry,
                        LegalMoveDiffObserver(patch_size=self._patch_size),
                        mode=self._sync_mode,
                        sequence_observer=(
                            LegalTwoPlyDiffObserver(patch_size=self._patch_size)
                            if self._sync_mode is SyncMode.HUMAN_VS_AI
                            else None
                        ),
                        required_stable_pairs=self._stable_pairs,
                        patch_size=self._patch_size,
                        capture_transition_evidence=self._capture_transition_evidence,
                        occupancy_observer=self._occupancy_observer,
                        require_atomic_two_ply=self._require_atomic_two_ply,
                    )
                    tracker.initialize(event.bgra)
                    sampler = AdaptiveBurstSampler(
                        steady_fps=self._steady_fps,
                        settle_ms=self._settle_ms,
                        stable_repeats=self._stable_pairs,
                    )
                    sampler.initialize(event)
                    with self._lock:
                        self._tracker = tracker
                        self._sampler = sampler
                    self._emit(
                        LiveSyncStatus.BASELINE_READY,
                        "stable board baseline is ready",
                        frame_size=event.size,
                        point_count=len(geometry.grid_points()),
                    )
                    continue

                with self._processing_lock:
                    if self._is_recovery_pending():
                        self._process_recovery_frame(event, tracker, sampler)
                        continue
                    if self._is_paused_for_recovery():
                        continue
                    try:
                        samples = sampler.on_frame(event)
                    except FrameSizeChangedError as exc:
                        resize_update = tracker.rebind_frame_size(exc.frame.bgra)
                        if resize_update.status is TrackingStatus.WATCHING:
                            sampler.initialize(exc.frame)
                            self._emit_tracking(
                                LiveSyncStatus.GEOMETRY_REBOUND,
                                "window resized proportionally; board geometry was rebound",
                                resize_update,
                                frame_size=exc.frame.size,
                            )
                            continue
                        with self._lock:
                            self._paused_for_recovery = True
                        self._emit_tracking(
                            LiveSyncStatus.CONTEXT_INVALID,
                            "resized frame did not preserve the confirmed position",
                            resize_update,
                            frame_size=exc.frame.size,
                        )
                        continue
                    self._process_samples(samples, sampler, tracker)
        except (OSError, RuntimeError, ValueError) as exc:
            if not self._is_closed():
                self._emit(LiveSyncStatus.ERROR, str(exc))
        finally:
            self._finalize("live sync worker stopped")

    def _process_samples(
        self,
        samples: tuple[CaptureFrame, ...],
        sampler: AdaptiveBurstSampler,
        tracker: StableMoveTracker,
    ) -> None:
        for sample in samples:
            before_position_id = tracker.board.position_id
            update = tracker.push(
                sample.bgra,
                capture_timestamp_ns=sample.timestamp_ns,
            )
            _set_sampling_mode(sampler, self._source, update.status)
            if update.status is TrackingStatus.ACCEPTED:
                with self._lock:
                    self._board = update.board
                self._emit_tracking(
                    LiveSyncStatus.MOVE_ACCEPTED,
                    (
                        "unique legal two-ply sequence accepted atomically"
                        if len(update.moves) == 2
                        else "unique legal move accepted"
                    ),
                    update,
                    frame_size=sample.size,
                    before_position_id=before_position_id,
                )
            elif update.status is TrackingStatus.WAITING_FOR_STABLE:
                self._emit_tracking(
                    LiveSyncStatus.WAITING_FOR_STABLE,
                    "waiting for board animation to settle",
                    update,
                    frame_size=sample.size,
                )
            elif update.status is TrackingStatus.WAITING_FOR_ENDPOINT:
                self._emit_tracking(
                    LiveSyncStatus.WAITING_FOR_ENDPOINT,
                    "selection is visible; waiting for the completed move",
                    update,
                    frame_size=sample.size,
                )
            elif update.status is TrackingStatus.WAITING_FOR_REPLY:
                self._emit_tracking(
                    LiveSyncStatus.WAITING_FOR_REPLY,
                    "first move is stable; waiting for the AI reply",
                    update,
                    frame_size=sample.size,
                )
            elif update.status is TrackingStatus.PAUSED_AMBIGUOUS:
                with self._lock:
                    self._paused_for_recovery = True
                self._emit_tracking(
                    LiveSyncStatus.PAUSED_AMBIGUOUS,
                    "visual change was ambiguous; manual recovery is required",
                    update,
                    frame_size=sample.size,
                )
            elif update.status is TrackingStatus.WATCHING:
                if self._keep_visible_status():
                    continue
                self._emit_tracking(
                    LiveSyncStatus.WATCHING,
                    "watching the confirmed position",
                    update,
                    frame_size=sample.size,
                )

    def _process_recovery_frame(
        self,
        frame: CaptureFrame,
        tracker: StableMoveTracker,
        sampler: AdaptiveBurstSampler,
    ) -> None:
        with self._lock:
            pending = self._pending_recovery
        if pending is None or frame.timestamp_ns <= pending.cutoff_timestamp_ns:
            return
        if frame.size != pending.geometry.frame_size:
            pending.geometry = BoardGeometry.from_quad(
                pending.quad,
                frame.size,
                pending.board.orientation,
            )
            pending.detector = FrameStabilityDetector(
                pending.geometry,
                required_stable_pairs=self._stable_pairs,
            )
        change = pending.detector.update(frame.bgra)
        if change is None or not change.stable:
            return
        before_position_id = tracker.board.position_id
        update = tracker.recover(
            pending.board,
            frame.bgra,
            geometry=pending.geometry,
        )
        sampler.initialize(frame)
        with self._lock:
            self._board = update.board
            self._quad = pending.quad
            self._pending_recovery = None
            self._paused_for_recovery = False
        self._emit_tracking(
            LiveSyncStatus.RECOVERY_ACCEPTED,
            "manual recovery accepted from a fresh stable frame",
            update,
            frame_size=frame.size,
            before_position_id=before_position_id,
            recovery_id=pending.request_id,
        )

    def _emit_tracking(
        self,
        status: LiveSyncStatus,
        message: str,
        update: TrackingUpdate,
        *,
        frame_size: tuple[int, int],
        before_position_id: str | None = None,
        recovery_id: int | None = None,
    ) -> None:
        with self._lock:
            self._last_status = status
        self._on_update(
            LiveSyncUpdate(
                status=status,
                board=update.board,
                message=message,
                moves=update.moves,
                observation=update.observation,
                before_position_id=before_position_id,
                after_position_id=(
                    update.board.position_id
                    if status in (LiveSyncStatus.MOVE_ACCEPTED, LiveSyncStatus.RECOVERY_ACCEPTED)
                    else None
                ),
                sync_mode=self._sync_mode,
                recovery_id=recovery_id,
                frame_size=frame_size,
                point_count=90,
                transition_evidence=update.transition_evidence,
            )
        )

    def _emit(
        self,
        status: LiveSyncStatus,
        message: str,
        *,
        frame_size: tuple[int, int] | None = None,
        point_count: int = 0,
        recovery_id: int | None = None,
    ) -> None:
        with self._lock:
            self._last_status = status
        self._on_update(
            LiveSyncUpdate(
                status=status,
                board=self.board,
                message=message,
                sync_mode=self._sync_mode,
                recovery_id=recovery_id,
                frame_size=frame_size,
                point_count=point_count,
            )
        )

    def _emit_closed(self, message: str) -> None:
        with self._lock:
            if self._closed_emitted:
                return
            self._closed_emitted = True
        self._emit(LiveSyncStatus.CLOSED, message)

    def _finalize(self, message: str) -> None:
        with self._lock:
            self._closed = True
        try:
            self._source.close()
        except (OSError, RuntimeError):
            pass
        with self._lock:
            self._finalized = True
            self._release_frame_state_locked()
        self._emit_closed(message)

    def _release_frame_state_locked(self) -> None:
        self._latest_frame = None
        self._tracker = None
        self._sampler = None
        self._pending_recovery = None

    def _is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _is_paused_for_recovery(self) -> bool:
        with self._lock:
            return self._paused_for_recovery

    def _is_recovery_pending(self) -> bool:
        with self._lock:
            return self._pending_recovery is not None

    def _keep_visible_status(self) -> bool:
        with self._lock:
            return self._last_status in (
                LiveSyncStatus.BASELINE_READY,
                LiveSyncStatus.WATCHING,
                LiveSyncStatus.MOVE_ACCEPTED,
            )


def _set_sampling_mode(
    sampler: AdaptiveBurstSampler,
    source: FrameSource,
    status: TrackingStatus,
) -> None:
    active = status in (
        TrackingStatus.WAITING_FOR_STABLE,
        TrackingStatus.WAITING_FOR_ENDPOINT,
        TrackingStatus.WAITING_FOR_REPLY,
    )
    sampler.set_bursting(active)
    if isinstance(source, BurstFrameSource):
        source.set_bursting(active)


def _queue_timeout(sampler: AdaptiveBurstSampler | None) -> float:
    if sampler is None:
        return 0.25
    due = sampler.next_due_ns()
    if due is None:
        return 0.25
    return max(0.0, min(0.25, (due - perf_counter_ns()) / 1_000_000_000))


def _ignore_update(_update: LiveSyncUpdate) -> None:
    pass
