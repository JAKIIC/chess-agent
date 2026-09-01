from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from time import monotonic
from typing import Protocol
from uuid import uuid4

import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.capture.protocol import (
    BurstFrameSource,
    CaptureClosedError,
    CaptureFrame,
    ClosedCallback,
    FrameCallback,
    FrameSource,
)
from xiangqi_agent.capture.windows_capture_source import WindowsCaptureSource
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
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.platform.windows import (
    WindowInfo,
    WindowsWindowCatalog,
    filter_target_windows,
    select_window,
)
from xiangqi_agent.sync.evidence import MoveSequenceProposal, SequenceCandidateEvidence
from xiangqi_agent.sync.live_session import LiveSyncSession, LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.transition_capture import TransitionCaptureEvidence
from xiangqi_agent.vision.geometry import NormalizedQuad, parse_normalized_quad
from xiangqi_agent.vision.occupancy import (
    KnownPositionOccupancyObserver,
    OccupancyObserver,
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / ".local"
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


class QuarantineCollectionError(RuntimeError):
    """One unlabelled Stage C event could not be isolated safely."""


class QuarantineCollectionTimeout(QuarantineCollectionError):
    """No terminal Stage C event arrived before the explicit deadline."""


class WindowCatalog(Protocol):
    def list_candidates(self) -> tuple[WindowInfo, ...]: ...


type SourceFactory = Callable[[WindowInfo], FrameSource]


class _FixedSizeFrameSource:
    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._first_size: tuple[int, int] | None = None
        self._closed_reported = False
        self._closed = False
        self._lock = Lock()

    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None:
        def receive(frame: CaptureFrame) -> None:
            with self._lock:
                if self._closed_reported:
                    return
                if self._first_size is None:
                    self._first_size = frame.size
                resized = frame.size != self._first_size
                if resized:
                    self._closed_reported = True
            if resized:
                self.close()
                if on_closed is not None:
                    on_closed(
                        CaptureClosedError(
                            "frame size changed during quarantine collection"
                        )
                    )
                return
            on_frame(frame)

        def closed(error: CaptureClosedError) -> None:
            with self._lock:
                if self._closed_reported:
                    return
                self._closed_reported = True
            if on_closed is not None:
                on_closed(error)

        self._source.start(receive, closed)

    def set_bursting(self, active: bool) -> None:
        if isinstance(self._source, BurstFrameSource):
            self._source.set_bursting(active)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._source.close()


def collect_human_ai_quarantine_event(
    *,
    source: FrameSource,
    board: BoardState,
    quad: NormalizedQuad,
    session_id: str,
    output_root: Path,
    timeout_seconds: float,
    on_update: Callable[[LiveSyncUpdate], None] | None = None,
    client_size: tuple[int, int],
    allowed_local_root: Path | None = None,
    event_id: str | None = None,
    occupancy_observer: OccupancyObserver | None = None,
    patch_size: int = 48,
    settle_ms: int = 100,
    stable_pairs: int = 2,
) -> Path:
    chosen_event_id = event_id or uuid4().hex
    _validate_collection_arguments(
        board=board,
        quad=quad,
        session_id=session_id,
        event_id=chosen_event_id,
        output_root=output_root,
        allowed_local_root=allowed_local_root or _DEFAULT_LOCAL_ROOT,
        client_size=client_size,
        timeout_seconds=timeout_seconds,
    )
    updates: Queue[LiveSyncUpdate] = Queue()

    def receive(update: LiveSyncUpdate) -> None:
        if on_update is not None:
            on_update(update)
        updates.put(update)

    session = LiveSyncSession(
        _FixedSizeFrameSource(source),
        board,
        quad,
        on_update=receive,
        sync_mode=SyncMode.HUMAN_VS_AI,
        steady_fps=2,
        settle_ms=settle_ms,
        stable_pairs=stable_pairs,
        patch_size=patch_size,
        capture_transition_evidence=True,
        occupancy_observer=occupancy_observer or KnownPositionOccupancyObserver(board),
        require_matching_baseline=True,
        require_atomic_two_ply=True,
    )
    deadline = monotonic() + timeout_seconds
    session.start()
    try:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise QuarantineCollectionTimeout(
                    "quarantine collection timed out without a terminal event"
                )
            try:
                update = updates.get(timeout=remaining)
            except Empty as exc:
                raise QuarantineCollectionTimeout(
                    "quarantine collection timed out without a terminal event"
                ) from exc
            if update.status in (
                LiveSyncStatus.MOVE_ACCEPTED,
                LiveSyncStatus.PAUSED_AMBIGUOUS,
            ):
                return _record_terminal_update(
                    update,
                    board=board,
                    quad=quad,
                    session_id=session_id,
                    event_id=chosen_event_id,
                    output_root=output_root,
                    client_size=client_size,
                )
            if update.status in (
                LiveSyncStatus.CONTEXT_INVALID,
                LiveSyncStatus.GEOMETRY_REBOUND,
                LiveSyncStatus.MANUAL_RECOVERY_REQUIRED,
                LiveSyncStatus.ERROR,
                LiveSyncStatus.CLOSED,
            ):
                raise QuarantineCollectionError(
                    f"quarantine collection stopped safely: {update.status.value}"
                )
    finally:
        session.close()


def _record_terminal_update(
    update: LiveSyncUpdate,
    *,
    board: BoardState,
    quad: NormalizedQuad,
    session_id: str,
    event_id: str,
    output_root: Path,
    client_size: tuple[int, int],
) -> Path:
    evidence = update.transition_evidence
    observation = update.observation
    if evidence is None:
        raise QuarantineCollectionError(
            "terminal event did not contain transition evidence"
        )
    if evidence.before_occupancy is None or evidence.after_occupancy is None:
        raise QuarantineCollectionError(
            "terminal event did not contain both occupancy snapshots"
        )
    if not isinstance(observation, MoveSequenceProposal):
        raise QuarantineCollectionError(
            "terminal event did not exercise the frozen two-ply decision gate"
        )

    if update.status is LiveSyncStatus.MOVE_ACCEPTED:
        if len(update.moves) != 2 or update.after_position_id is None:
            raise QuarantineCollectionError(
                "separate single-ply events are not atomic Stage C evidence"
            )
        observed_status = StageCObservedStatus.ACCEPTED
        observed_moves = tuple(move.uci for move in update.moves)
        observed_final = update.after_position_id
    else:
        observed_status = StageCObservedStatus.REJECTED
        observed_moves = ()
        observed_final = board.position_id

    rejection_reasons = observation.evidence.rejection_reasons
    if (
        update.status is LiveSyncStatus.PAUSED_AMBIGUOUS
        and not set(rejection_reasons) <= _ALLOWED_REJECTION_REASONS
    ):
        raise QuarantineCollectionError(
            "internal tracker failure cannot become quarantine evidence"
        )
    candidates = tuple(
        _candidate_record(candidate)
        for candidate in observation.evidence.candidates[:2]
    )
    context = _capture_context(
        evidence,
        quad=quad,
        orientation=board.orientation,
        frame_size=update.frame_size,
        client_size=client_size,
    )
    event = QuarantinedStageCEventV1(
        event_id=event_id,
        session_id=session_id,
        created_at_utc=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00",
            "Z",
        ),
        confirmed_fen=board.fen,
        confirmed_position_id=board.position_id,
        observed_status=observed_status,
        observed_moves_uci=observed_moves,
        observed_final_position_id=observed_final,
        side_to_move=board.side_to_move,
        orientation=board.orientation,
        changed_points=evidence.changed_points,
        local_differences=evidence.local_differences,
        candidates=candidates,
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
    return QuarantineEventRecorder(output_root, enabled=True).record(event, crops)


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
    orientation: Orientation,
    frame_size: tuple[int, int] | None,
    client_size: tuple[int, int],
) -> CaptureContext:
    if frame_size is None:
        raise QuarantineCollectionError("terminal event did not expose a frame size")
    geometry_revision = sha256(
        repr((quad.points, frame_size, orientation.value)).encode("ascii")
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
        generation_id=1,
    )


def _validate_collection_arguments(
    *,
    board: BoardState,
    quad: NormalizedQuad,
    session_id: str,
    event_id: str,
    output_root: Path,
    allowed_local_root: Path,
    client_size: tuple[int, int],
    timeout_seconds: float,
) -> None:
    if not isinstance(board, BoardState):
        raise TypeError("board must be a BoardState")
    if not isinstance(quad, NormalizedQuad):
        raise TypeError("quad must be a NormalizedQuad")
    if _IDENTIFIER.fullmatch(session_id) is None or _IDENTIFIER.fullmatch(event_id) is None:
        raise ValueError("session and event ids must be anonymous path-safe identifiers")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not isinstance(allowed_local_root, Path):
        raise TypeError("allowed_local_root must be a Path")
    if allowed_local_root.name != ".local":
        raise ValueError("approved diagnostic root must be named .local")
    if output_root.name != "stage-c-quarantine":
        raise ValueError("output_root must be the stage-c-quarantine directory")
    try:
        output_root.resolve().relative_to(allowed_local_root.resolve())
    except ValueError as exc:
        raise ValueError("output_root must stay under the approved .local root") from exc
    if (
        not isinstance(client_size, tuple)
        or len(client_size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in client_size
        )
    ):
        raise ValueError("client_size must contain two positive integers")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and positive")


def main(
    argv: Sequence[str] | None = None,
    *,
    catalog: WindowCatalog | None = None,
    source_factory: SourceFactory | None = None,
    allowed_local_root: Path | None = None,
    occupancy_observer: OccupancyObserver | None = None,
) -> int:
    args = _parse_args(argv)
    window_catalog = catalog or WindowsWindowCatalog()
    factory = source_factory or (
        lambda window: WindowsCaptureSource(window, fps=20)
    )
    try:
        candidates = filter_target_windows(window_catalog.list_candidates())
        if args.hwnd is None:
            if len(candidates) != 1:
                raise QuarantineCollectionError(
                    "collector requires one selected visible target window"
                )
            window = candidates[0]
        else:
            window = select_window(candidates, args.hwnd)
        board = replace(
            parse_fen(args.fen),
            orientation=Orientation(args.orientation),
        )

        def report_update(update: LiveSyncUpdate) -> None:
            if update.status is LiveSyncStatus.BASELINE_READY:
                print(
                    json.dumps(
                        {
                            "status": "baseline_ready",
                            "frame_size": update.frame_size,
                            "point_count": update.point_count,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        output = collect_human_ai_quarantine_event(
            source=factory(window),
            board=board,
            quad=parse_normalized_quad(args.quad),
            session_id=args.session_id,
            output_root=args.output_root,
            timeout_seconds=args.timeout_seconds,
            on_update=report_update,
            client_size=window.client_size,
            allowed_local_root=allowed_local_root or _DEFAULT_LOCAL_ROOT,
            occupancy_observer=occupancy_observer,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "collection_error", "code": _error_code(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "recorded",
                "session_id": output.parent.name,
                "event_id": output.name,
            },
            sort_keys=True,
        )
    )
    return 0


def _error_code(error: BaseException) -> str:
    if isinstance(error, QuarantineCollectionTimeout):
        return "timeout"
    if isinstance(error, CaptureClosedError):
        return "capture_closed"
    if isinstance(error, QuarantineCollectionError):
        return "collection_stopped"
    if isinstance(error, OSError):
        return "platform_error"
    return "invalid_configuration"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one unlabelled human-vs-AI event without controlling the game"
    )
    parser.add_argument("--fen", required=True)
    parser.add_argument("--quad", required=True)
    parser.add_argument(
        "--orientation",
        choices=tuple(item.value for item in Orientation),
        default=Orientation.RED_BOTTOM.value,
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hwnd", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0 or not isfinite(args.timeout_seconds):
        parser.error("--timeout-seconds must be finite and positive")
    if args.hwnd is not None and args.hwnd <= 0:
        parser.error("--hwnd must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
