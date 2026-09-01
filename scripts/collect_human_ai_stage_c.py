from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from time import monotonic
from typing import Protocol, cast
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
from xiangqi_agent.capture.visible_window_source import VisibleWindowCaptureSource
from xiangqi_agent.diagnostics.stage_c_gate import DEFAULT_STAGE_C_THRESHOLD_PROFILE
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleRecorder,
    HumanAiStageCSampleV1,
    StageCCandidateRecord,
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo, WindowsWindowCatalog
from xiangqi_agent.sync.evidence import MoveSequenceProposal, SequenceCandidateEvidence
from xiangqi_agent.sync.live_session import LiveSyncSession, LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.sync.transition_capture import TransitionCaptureEvidence
from xiangqi_agent.vision.geometry import NormalizedQuad, parse_normalized_quad

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / ".local"
_FROZEN_GATE_REJECTION_REASONS = frozenset(
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


class StageCCollectionError(RuntimeError):
    """One requested Stage C event could not be recorded safely."""


class StageCCollectionTimeout(StageCCollectionError):
    """No terminal event arrived before the explicit collection deadline."""


class WindowCatalog(Protocol):
    def list_candidates(self) -> tuple[WindowInfo, ...]: ...


type SourceFactory = Callable[[WindowInfo], FrameSource]


class _FixedSizeFrameSource:
    """Fail a collection generation closed after any post-first-frame resize."""

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._first_size: tuple[int, int] | None = None
        self._closed_reported = False
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
                self._source.close()
                if on_closed is not None:
                    on_closed(
                        CaptureClosedError(
                            "frame size changed during frozen Stage C collection"
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
        self._source.close()


def collect_human_ai_stage_c_event(
    source: FrameSource,
    board: BoardState,
    quad: NormalizedQuad,
    *,
    expected_moves_uci: tuple[str, str] | None,
    scenario: StageCScenario,
    session_id: str,
    output_root: Path,
    client_size: tuple[int, int],
    allowed_local_root: Path | None = None,
    timeout_seconds: float = 60.0,
    sample_id: str | None = None,
    patch_size: int = 48,
    settle_ms: int = 100,
    stable_pairs: int = 2,
    on_update: Callable[[LiveSyncUpdate], None] | None = None,
) -> Path:
    _validate_collection_arguments(
        board=board,
        quad=quad,
        expected_moves_uci=expected_moves_uci,
        scenario=scenario,
        session_id=session_id,
        output_root=output_root,
        allowed_local_root=allowed_local_root or _DEFAULT_LOCAL_ROOT,
        client_size=client_size,
        timeout_seconds=timeout_seconds,
        sample_id=sample_id,
    )
    expected_final = (
        _project_expected_sequence(board, expected_moves_uci)
        if expected_moves_uci is not None
        else None
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
    )
    deadline = monotonic() + timeout_seconds
    session.start()
    try:
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise StageCCollectionTimeout("Stage C collection timed out without a terminal event")
            try:
                update = updates.get(timeout=remaining)
            except Empty as exc:
                raise StageCCollectionTimeout(
                    "Stage C collection timed out without a terminal event"
                ) from exc
            if update.status in (
                LiveSyncStatus.MOVE_ACCEPTED,
                LiveSyncStatus.PAUSED_AMBIGUOUS,
            ):
                return _record_terminal_update(
                    update,
                    board=board,
                    quad=quad,
                    expected_moves_uci=expected_moves_uci,
                    expected_final=expected_final,
                    scenario=scenario,
                    session_id=session_id,
                    sample_id=sample_id or uuid4().hex,
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
                raise StageCCollectionError(
                    f"Stage C collection stopped safely: {update.status.value}"
                )
    finally:
        session.close()


def _record_terminal_update(
    update: LiveSyncUpdate,
    *,
    board: BoardState,
    quad: NormalizedQuad,
    expected_moves_uci: tuple[str, str] | None,
    expected_final: BoardState | None,
    scenario: StageCScenario,
    session_id: str,
    sample_id: str,
    output_root: Path,
    client_size: tuple[int, int],
) -> Path:
    evidence = update.transition_evidence
    observation = update.observation
    if evidence is None:
        raise StageCCollectionError("terminal event did not contain transition evidence")
    if not isinstance(observation, MoveSequenceProposal):
        raise StageCCollectionError(
            "terminal event did not exercise the frozen two-ply decision gate"
        )

    observed_moves: tuple[str, ...]
    if update.status is LiveSyncStatus.MOVE_ACCEPTED:
        if len(update.moves) != 2 or update.after_position_id is None:
            raise StageCCollectionError(
                "separate single-ply events are not Stage C atomic evidence"
            )
        observed_status = StageCObservedStatus.ACCEPTED
        observed_moves = tuple(move.uci for move in update.moves)
        observed_final = update.after_position_id
    else:
        observed_status = StageCObservedStatus.REJECTED
        observed_moves = ()
        observed_final = board.position_id

    candidates = tuple(
        _candidate_record(candidate)
        for candidate in observation.evidence.candidates[:2]
    )
    rejection_reasons = observation.evidence.rejection_reasons
    if (
        update.status is LiveSyncStatus.PAUSED_AMBIGUOUS
        and not set(rejection_reasons) <= _FROZEN_GATE_REJECTION_REASONS
    ):
        raise StageCCollectionError(
            "internal tracker failure cannot be recorded as frozen gate evidence"
        )
    context = _capture_context(
        evidence,
        quad=quad,
        orientation=board.orientation,
        frame_size=update.frame_size,
        client_size=client_size,
    )
    sample = HumanAiStageCSampleV1(
        sample_id=sample_id,
        session_id=session_id,
        created_at_utc=datetime.now(UTC).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        confirmed_fen=board.fen,
        confirmed_position_id=board.position_id,
        expected_outcome=(
            StageCExpectedOutcome.ACCEPT
            if expected_moves_uci is not None
            else StageCExpectedOutcome.REJECT
        ),
        scenario=scenario,
        ground_truth_moves_uci=expected_moves_uci or (),
        expected_final_position_id=(
            expected_final.position_id if expected_final is not None else None
        ),
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
    )
    crops = tuple(
        TransitionPointCrops(crop.point_index, crop.before, crop.after)
        for crop in evidence.crops
    )
    return HumanAiStageCSampleRecorder(output_root, enabled=True).record(sample, crops)


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
        raise StageCCollectionError("terminal event did not expose a frame size")
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


def _project_expected_sequence(
    board: BoardState,
    moves_uci: tuple[str, str],
) -> BoardState:
    projected = board
    for uci in moves_uci:
        move = next((candidate for candidate in legal_moves(projected) if candidate.uci == uci), None)
        if move is None:
            raise StageCCollectionError("expected Stage C moves are not a legal two-ply chain")
        projected = apply_move(projected, move)
    return projected


def _validate_collection_arguments(
    *,
    board: BoardState,
    quad: NormalizedQuad,
    expected_moves_uci: tuple[str, str] | None,
    scenario: StageCScenario,
    session_id: str,
    output_root: Path,
    allowed_local_root: Path,
    client_size: tuple[int, int],
    timeout_seconds: float,
    sample_id: str | None,
) -> None:
    if not isinstance(board, BoardState):
        raise TypeError("board must be a BoardState")
    if not isinstance(quad, NormalizedQuad):
        raise TypeError("quad must be a NormalizedQuad")
    if not isinstance(scenario, StageCScenario):
        raise TypeError("scenario must be a StageCScenario")
    if expected_moves_uci is not None and (
        not isinstance(expected_moves_uci, tuple) or len(expected_moves_uci) != 2
    ):
        raise ValueError("expected_moves_uci must contain exactly two moves")
    if expected_moves_uci is not None and scenario is not StageCScenario.VALID_TWO_PLY:
        raise ValueError("expected moves require the valid_two_ply scenario")
    if expected_moves_uci is None and scenario is StageCScenario.VALID_TWO_PLY:
        raise ValueError("a rejection label requires a rejection scenario")
    for identifier in (session_id, sample_id):
        if identifier is not None and _IDENTIFIER.fullmatch(identifier) is None:
            raise ValueError("session and sample ids must be anonymous path-safe identifiers")
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a Path")
    if not isinstance(allowed_local_root, Path):
        raise TypeError("allowed_local_root must be a Path")
    if allowed_local_root.name != ".local":
        raise ValueError("approved diagnostic root must be named .local")
    resolved_local_root = allowed_local_root.resolve()
    resolved_output = output_root.resolve()
    try:
        resolved_output.relative_to(resolved_local_root)
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
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive")


def main(
    argv: Sequence[str] | None = None,
    *,
    catalog: WindowCatalog | None = None,
    source_factory: SourceFactory | None = None,
    allowed_local_root: Path | None = None,
) -> int:
    args = _parse_args(argv)
    window_catalog = catalog or WindowsWindowCatalog()
    factory = source_factory or (
        lambda window: VisibleWindowCaptureSource(window, fps=2, burst_fps=20)
    )
    try:
        candidates = window_catalog.list_candidates()
        if len(candidates) != 1:
            raise StageCCollectionError(
                "collector requires exactly one visible, unobscured 天天象棋 window"
            )
        window = candidates[0]
        board = replace(
            parse_fen(args.fen),
            orientation=Orientation(args.orientation),
        )
        expected = (
            cast(tuple[str, str], tuple(args.expected_moves))
            if args.expected_moves is not None
            else None
        )

        def report_update(update: LiveSyncUpdate) -> None:
            if update.status is LiveSyncStatus.BASELINE_READY:
                print(
                    json.dumps(
                        {
                            "status": "BASELINE_READY",
                            "frame_size": update.frame_size,
                            "point_count": update.point_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        output = collect_human_ai_stage_c_event(
            factory(window),
            board,
            parse_normalized_quad(args.quad),
            expected_moves_uci=expected,
            scenario=StageCScenario(args.scenario),
            session_id=args.session_id,
            output_root=args.output_root,
            allowed_local_root=allowed_local_root or _DEFAULT_LOCAL_ROOT,
            client_size=window.client_size,
            timeout_seconds=args.timeout_seconds,
            on_update=report_update,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "collection_error", "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "RECORDED",
                "session_id": output.parent.name,
                "sample_id": output.name,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one labeled human-vs-AI Stage C event without controlling the game"
    )
    parser.add_argument("--fen", required=True)
    parser.add_argument("--quad", required=True)
    parser.add_argument(
        "--orientation",
        choices=tuple(item.value for item in Orientation),
        default=Orientation.RED_BOTTOM.value,
    )
    labels = parser.add_mutually_exclusive_group(required=True)
    labels.add_argument("--expected-moves", nargs=2, metavar=("USER_UCI", "AI_UCI"))
    labels.add_argument("--expect-reject", action="store_true")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=tuple(item.value for item in StageCScenario),
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.expected_moves is not None and args.scenario != StageCScenario.VALID_TWO_PLY.value:
        parser.error("--expected-moves requires --scenario valid_two_ply")
    if args.expect_reject and args.scenario == StageCScenario.VALID_TWO_PLY.value:
        parser.error("--expect-reject requires a rejection scenario")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
