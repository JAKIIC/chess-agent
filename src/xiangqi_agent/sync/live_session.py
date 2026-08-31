from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from queue import Empty, Queue
from threading import Lock, Thread, current_thread
from time import perf_counter_ns

from xiangqi_agent.capture.adaptive_sampling import (
    AdaptiveBurstSampler,
    FrameSizeChangedError,
)
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame, FrameSource
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.sync.evidence import MoveProposal
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.tracker import StableMoveTracker, TrackingStatus, TrackingUpdate
from xiangqi_agent.vision.change_detection import FrameStabilityDetector
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad


class LiveSyncStatus(StrEnum):
    CONNECTING = "connecting"
    BASELINE_READY = "baseline_ready"
    WATCHING = "watching"
    WAITING_FOR_STABLE = "waiting_for_stable"
    WAITING_FOR_ENDPOINT = "waiting_for_endpoint"
    MOVE_ACCEPTED = "move_accepted"
    GEOMETRY_REBOUND = "geometry_rebound"
    PAUSED_AMBIGUOUS = "paused_ambiguous"
    CONTEXT_INVALID = "context_invalid"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"
    CLOSED = "closed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LiveSyncUpdate:
    status: LiveSyncStatus
    board: BoardState
    message: str
    move: Move | None = None
    observation: MoveProposal | None = None
    frame_size: tuple[int, int] | None = None
    point_count: int = 0


@dataclass(frozen=True, slots=True)
class _Stop:
    pass


class LiveSyncSession:
    """Continuously turn stable capture frames into rule-verified board updates."""

    def __init__(
        self,
        source: FrameSource,
        board: BoardState,
        quad: NormalizedQuad,
        *,
        on_update: Callable[[LiveSyncUpdate], None] | None = None,
        steady_fps: int = 2,
        settle_ms: int = 100,
        stable_pairs: int = 2,
        patch_size: int = 48,
    ) -> None:
        if stable_pairs <= 0:
            raise ValueError("stable_pairs must be positive")
        self._source = source
        self._board = board
        self._quad = quad
        self._on_update = on_update or _ignore_update
        self._steady_fps = steady_fps
        self._settle_ms = settle_ms
        self._stable_pairs = stable_pairs
        self._patch_size = patch_size
        self._events: Queue[CaptureFrame | CaptureClosedError | _Stop] = Queue()
        self._lock = Lock()
        self._processing_lock = Lock()
        self._started = False
        self._closed = False
        self._closed_emitted = False
        self._paused_for_recovery = False
        self._last_status: LiveSyncStatus | None = None
        self._thread: Thread | None = None
        self._tracker: StableMoveTracker | None = None
        self._sampler: AdaptiveBurstSampler | None = None
        self._latest_frame: CaptureFrame | None = None

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
            self._source.start(self._events.put, self._events.put)
        except (OSError, RuntimeError, ValueError) as exc:
            self._emit(LiveSyncStatus.ERROR, str(exc))
            return
        worker = Thread(target=self._run, name="xiangqi-live-sync", daemon=True)
        self._thread = worker
        worker.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self._events.put(_Stop())
        self._source.close()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=2.0)
        self._emit_closed("live sync session closed")

    def recover(
        self,
        board: BoardState,
        *,
        quad: NormalizedQuad | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("live sync session is closed")
            tracker = self._tracker
            sampler = self._sampler
            latest = self._latest_frame
            recovered_quad = quad or self._quad
        if tracker is None or sampler is None or latest is None:
            raise RuntimeError("live sync baseline is not ready")
        geometry = BoardGeometry.from_quad(
            recovered_quad,
            latest.size,
            board.orientation,
        )
        with self._processing_lock:
            update = tracker.recover(board, latest.bgra, geometry=geometry)
            sampler.initialize(latest)
        with self._lock:
            self._board = update.board
            self._quad = recovered_quad
            self._paused_for_recovery = False
        self._emit_tracking(
            LiveSyncStatus.WATCHING,
            "manual board recovery accepted; watching resumed",
            update,
            frame_size=latest.size,
        )

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
                    if (
                        sampler is not None
                        and tracker is not None
                        and not self._is_paused_for_recovery()
                    ):
                        with self._processing_lock:
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
                    tracker = StableMoveTracker(
                        self.board,
                        geometry,
                        LegalMoveDiffObserver(patch_size=self._patch_size),
                        required_stable_pairs=self._stable_pairs,
                        patch_size=self._patch_size,
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

                if self._is_paused_for_recovery():
                    continue

                with self._processing_lock:
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

    def _process_samples(
        self,
        samples: tuple[CaptureFrame, ...],
        sampler: AdaptiveBurstSampler,
        tracker: StableMoveTracker,
    ) -> None:
        for sample in samples:
            update = tracker.push(sample.bgra)
            _set_sampling_mode(sampler, update.status)
            if update.status is TrackingStatus.ACCEPTED:
                with self._lock:
                    self._board = update.board
                self._emit_tracking(
                    LiveSyncStatus.MOVE_ACCEPTED,
                    "unique legal move accepted",
                    update,
                    frame_size=sample.size,
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

    def _emit_tracking(
        self,
        status: LiveSyncStatus,
        message: str,
        update: TrackingUpdate,
        *,
        frame_size: tuple[int, int],
    ) -> None:
        with self._lock:
            self._last_status = status
        self._on_update(
            LiveSyncUpdate(
                status=status,
                board=update.board,
                message=message,
                move=update.move,
                observation=update.observation,
                frame_size=frame_size,
                point_count=90,
            )
        )

    def _emit(
        self,
        status: LiveSyncStatus,
        message: str,
        *,
        frame_size: tuple[int, int] | None = None,
        point_count: int = 0,
    ) -> None:
        with self._lock:
            self._last_status = status
        self._on_update(
            LiveSyncUpdate(
                status=status,
                board=self.board,
                message=message,
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

    def _is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def _is_paused_for_recovery(self) -> bool:
        with self._lock:
            return self._paused_for_recovery

    def _keep_visible_status(self) -> bool:
        with self._lock:
            return self._last_status in (
                LiveSyncStatus.BASELINE_READY,
                LiveSyncStatus.WATCHING,
                LiveSyncStatus.MOVE_ACCEPTED,
            )


def _set_sampling_mode(sampler: AdaptiveBurstSampler, status: TrackingStatus) -> None:
    if status in (
        TrackingStatus.WAITING_FOR_STABLE,
        TrackingStatus.WAITING_FOR_ENDPOINT,
    ):
        sampler.set_bursting(True)
    elif status in (TrackingStatus.WATCHING, TrackingStatus.ACCEPTED):
        sampler.set_bursting(False)


def _queue_timeout(sampler: AdaptiveBurstSampler | None) -> float:
    if sampler is None:
        return 0.25
    due = sampler.next_due_ns()
    if due is None:
        return 0.25
    return max(0.0, min(0.25, (due - perf_counter_ns()) / 1_000_000_000))


def _ignore_update(_update: LiveSyncUpdate) -> None:
    pass
