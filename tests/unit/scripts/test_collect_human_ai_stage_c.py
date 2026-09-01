from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import cv2
import numpy as np
import pytest

from scripts.collect_human_ai_stage_c import (
    StageCCollectionError,
    StageCCollectionTimeout,
    _record_terminal_update,
    collect_human_ai_stage_c_event,
    main,
)
from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureClosedError, ClosedCallback, FrameCallback
from xiangqi_agent.diagnostics.stage_c_gate import DEFAULT_STAGE_C_FEATURE_VERSION
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.sync.evidence import (
    MoveSequenceEvidence,
    MoveSequenceProposal,
    ObservationStatus,
    SequenceCandidateEvidence,
)
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.transition_capture import (
    TransitionCaptureEvidence,
    TransitionPointEvidence,
)
from xiangqi_agent.vision.geometry import parse_normalized_quad

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
QUAD = parse_normalized_quad("0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540")


def _render(board: BoardState) -> np.ndarray:
    frame = np.zeros((240, 216, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = PALETTE[symbol]
    return frame


def _move(board: BoardState, uci: str):
    return next(move for move in legal_moves(board) if move.uci == uci)


def _merged_position() -> tuple[BoardState, tuple[str, str]]:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    return apply_move(middle, second), (first.uci, second.uci)


def _ambiguous_position() -> BoardState:
    board = parse_fen(START)
    pieces = list(board.pieces)
    for uci in ("b2b3", "h2h3"):
        move = _move(board, uci)
        pieces[move.to_index] = pieces[move.from_index]
        pieces[move.from_index] = "."
    return BoardState(tuple(pieces), side_to_move=board.side_to_move)


class _NotifyingFakeSource(FakeFrameSource):
    def __init__(self) -> None:
        super().__init__(hwnd=42)
        self.started = Event()

    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None:
        super().start(on_frame, on_closed)
        self.started.set()


class _Catalog:
    def __init__(self, candidates: tuple[WindowInfo, ...]) -> None:
        self._candidates = candidates

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return self._candidates


class _ScriptedSource(FakeFrameSource):
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        super().__init__(hwnd=42)
        self._frames = frames

    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None:
        super().start(on_frame, on_closed)

        def emit() -> None:
            timestamps = (0, 50_000_000, 100_000_000, 150_000_000, 260_000_000)
            try:
                for frame, timestamp in zip(self._frames, timestamps, strict=True):
                    sleep(0.03)
                    self.push(frame, timestamp)
            except CaptureClosedError:
                return

        Thread(target=emit, daemon=True).start()


def _wait_until(predicate: object, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        sleep(0.01)
    raise AssertionError("timed out waiting for collector state")


def _local_root(tmp_path: Path) -> Path:
    return tmp_path / ".local"


def _sample_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "samples"


def _launch_collection(
    tmp_path: Path,
    source: _NotifyingFakeSource,
    *,
    expected_moves: tuple[str, str] | None,
    scenario: StageCScenario,
    sample_id: str,
    timeout_seconds: float = 2.0,
    client_size: tuple[int, int] = (216, 240),
    patch_size: int = CELL,
):
    updates = []
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["path"] = collect_human_ai_stage_c_event(
                source,
                parse_fen(START),
                QUAD,
                expected_moves_uci=expected_moves,
                scenario=scenario,
                session_id="session-a",
                sample_id=sample_id,
                output_root=_sample_root(tmp_path),
                allowed_local_root=_local_root(tmp_path),
                client_size=client_size,
                timeout_seconds=timeout_seconds,
                patch_size=patch_size,
                settle_ms=100,
                stable_pairs=2,
                on_update=updates.append,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            outcome["error"] = exc

    worker = Thread(target=run, daemon=True)
    worker.start()
    assert source.started.wait(2.0)
    return worker, updates, outcome


def _start_collection(
    tmp_path: Path,
    source: _NotifyingFakeSource,
    *,
    expected_moves: tuple[str, str] | None,
    scenario: StageCScenario,
    sample_id: str,
    timeout_seconds: float = 2.0,
    client_size: tuple[int, int] = (216, 240),
    patch_size: int = CELL,
):
    worker, updates, outcome = _launch_collection(
        tmp_path,
        source,
        expected_moves=expected_moves,
        scenario=scenario,
        sample_id=sample_id,
        timeout_seconds=timeout_seconds,
        client_size=client_size,
        patch_size=patch_size,
    )
    baseline = _render(parse_fen(START))
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))
    return worker, updates, outcome


def _finish_merged(source: _NotifyingFakeSource) -> None:
    final, _moves = _merged_position()
    merged = _render(final)
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)


def test_collector_writes_only_after_an_atomic_terminal_event(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    final, moves = _merged_position()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=moves,
        scenario=StageCScenario.VALID_TWO_PLY,
        sample_id="accepted-event",
    )
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))

    _finish_merged(source)
    worker.join(3.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    sample_dir = outcome["path"]
    assert isinstance(sample_dir, Path)
    manifest = json.loads((sample_dir / "manifest.json").read_text("utf-8"))
    assert manifest["expected_outcome"] == "accept"
    assert manifest["ground_truth_moves_uci"] == list(moves)
    assert manifest["observed_status"] == "accepted"
    assert manifest["observed_moves_uci"] == list(moves)
    assert manifest["observed_final_position_id"] == final.position_id
    pngs = tuple(sample_dir.glob("*.png"))
    assert 2 <= len(pngs) <= 8
    assert all(cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (48, 48, 4) for path in pngs)


def test_collector_records_a_safe_rejection_without_exposing_moves(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.MULTIPLE_CANDIDATES,
        sample_id="rejected-event",
    )
    ambiguous = _render(_ambiguous_position())

    source.push(ambiguous, 150_000_000)
    source.push(ambiguous.copy(), 260_000_000)
    worker.join(3.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    sample_dir = outcome["path"]
    assert isinstance(sample_dir, Path)
    manifest = json.loads((sample_dir / "manifest.json").read_text("utf-8"))
    assert manifest["expected_outcome"] == "reject"
    assert manifest["observed_status"] == "rejected"
    assert manifest["observed_moves_uci"] == []
    assert manifest["observed_final_position_id"] == manifest["confirmed_position_id"]
    assert manifest["rejection_reasons"]


def test_collector_timeout_writes_no_sample(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.CONTINUOUS_ANIMATION,
        sample_id="timeout-event",
        timeout_seconds=0.15,
    )

    worker.join(2.0)

    assert isinstance(outcome.get("error"), StageCCollectionTimeout)
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))


def test_collector_rejects_output_outside_the_explicit_local_root_before_capture(
    tmp_path: Path,
) -> None:
    source = _NotifyingFakeSource()

    with pytest.raises(ValueError, match="approved .local"):
        collect_human_ai_stage_c_event(
            source,
            parse_fen(START),
            QUAD,
            expected_moves_uci=None,
            scenario=StageCScenario.OCCLUSION,
            session_id="session-a",
            sample_id="outside-event",
            output_root=tmp_path / "outside",
            allowed_local_root=_local_root(tmp_path),
            client_size=(216, 240),
            timeout_seconds=0.1,
        )

    assert not source.started.is_set()
    assert not (tmp_path / "outside").exists()


def test_collector_resize_failure_writes_no_sample(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.RESIZE,
        sample_id="resize-event",
    )
    stretched = np.repeat(_render(parse_fen(START)), 2, axis=1)

    source.push(stretched, 200_000_000)
    worker.join(3.0)

    assert isinstance(outcome.get("error"), StageCCollectionError)
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))


def test_collector_proportional_resize_also_aborts_without_a_sample(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.RESIZE,
        sample_id="proportional-resize-event",
    )
    resized = np.repeat(np.repeat(_render(parse_fen(START)), 2, axis=0), 2, axis=1)

    source.push(resized, 200_000_000)
    worker.join(3.0)

    assert isinstance(outcome.get("error"), StageCCollectionError)
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))


def test_collector_prebaseline_resize_aborts_without_a_sample(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _launch_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.RESIZE,
        sample_id="prebaseline-resize-event",
    )
    baseline = _render(parse_fen(START))
    resized = np.repeat(np.repeat(baseline, 2, axis=0), 2, axis=1)

    source.push(baseline, 0)
    source.push(resized, 50_000_000)
    worker.join(3.0)

    assert isinstance(outcome.get("error"), StageCCollectionError)
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))


def test_collector_accepts_a_dpi_scaled_first_frame_as_the_frozen_generation(
    tmp_path: Path,
) -> None:
    source = _NotifyingFakeSource()
    final, moves = _merged_position()
    worker, updates, outcome = _launch_collection(
        tmp_path,
        source,
        expected_moves=moves,
        scenario=StageCScenario.VALID_TWO_PLY,
        sample_id="dpi-scaled-event",
        client_size=(216, 240),
        patch_size=CELL * 2,
    )
    baseline = np.repeat(np.repeat(_render(parse_fen(START)), 2, axis=0), 2, axis=1)
    merged = np.repeat(np.repeat(_render(final), 2, axis=0), 2, axis=1)

    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(lambda: any(update.status is LiveSyncStatus.BASELINE_READY for update in updates))
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)
    worker.join(3.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    sample_dir = outcome["path"]
    assert isinstance(sample_dir, Path)
    manifest = json.loads((sample_dir / "manifest.json").read_text("utf-8"))
    assert manifest["capture_context"]["client_size"] == [216, 240]
    assert manifest["capture_context"]["wgc_size"] == [432, 480]
    assert manifest["capture_context"]["dpi_scale"] == 2.0


def test_collector_window_close_writes_no_sample(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, _updates, outcome = _start_collection(
        tmp_path,
        source,
        expected_moves=None,
        scenario=StageCScenario.OCCLUSION,
        sample_id="close-event",
    )

    source.simulate_target_close()
    worker.join(3.0)

    assert isinstance(outcome.get("error"), StageCCollectionError)
    assert not tuple(_sample_root(tmp_path).rglob("manifest.json"))


def test_collector_duplicate_sample_id_never_overwrites_first_event(tmp_path: Path) -> None:
    _final, moves = _merged_position()
    first_source = _NotifyingFakeSource()
    first_worker, _updates, first_outcome = _start_collection(
        tmp_path,
        first_source,
        expected_moves=moves,
        scenario=StageCScenario.VALID_TWO_PLY,
        sample_id="same-event",
    )
    _finish_merged(first_source)
    first_worker.join(3.0)
    assert "error" not in first_outcome
    sample_dir = first_outcome["path"]
    assert isinstance(sample_dir, Path)
    original_manifest = (sample_dir / "manifest.json").read_bytes()

    second_source = _NotifyingFakeSource()
    second_worker, _updates, second_outcome = _start_collection(
        tmp_path,
        second_source,
        expected_moves=moves,
        scenario=StageCScenario.VALID_TWO_PLY,
        sample_id="same-event",
    )
    _finish_merged(second_source)
    second_worker.join(3.0)

    assert isinstance(second_outcome.get("error"), FileExistsError)
    assert (sample_dir / "manifest.json").read_bytes() == original_manifest


def test_collector_refuses_internal_commit_failures_as_frozen_gate_samples(
    tmp_path: Path,
) -> None:
    board = parse_fen(START)
    final, moves_uci = _merged_position()
    first = _move(board, moves_uci[0])
    middle = apply_move(board, first)
    second = _move(middle, moves_uci[1])
    changed_points = tuple(
        index
        for index, (before, after) in enumerate(
            zip(board.pieces, final.pieces, strict=True)
        )
        if before != after
    )
    local_differences = tuple(
        20.0 if index in changed_points else 0.0 for index in range(90)
    )
    candidate = SequenceCandidateEvidence(
        moves=(first, second),
        changed_points=changed_points,
        expected_change_floor=20.0,
        unexpected_difference=0.0,
        maximum_template_distance=0.01,
        minimum_template_margin=0.2,
        minimum_template_confidence=0.99,
        score=20.0,
        final_position_id=final.position_id,
    )
    observation = MoveSequenceProposal(
        status=ObservationStatus.AMBIGUOUS,
        moves=(),
        evidence_score=0.0,
        evidence=MoveSequenceEvidence(
            candidates=(candidate,),
            local_differences=local_differences,
            rejection_reasons=("sequence_commit_failed",),
            feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
        ),
    )
    blank = np.zeros((48, 48, 4), dtype=np.uint8)
    evidence = TransitionCaptureEvidence(
        changed_points=changed_points,
        local_differences=local_differences,
        crops=tuple(
            TransitionPointEvidence(index, blank, blank)
            for index in changed_points
        ),
        decision_latency_ms=1.0,
    )
    update = LiveSyncUpdate(
        status=LiveSyncStatus.PAUSED_AMBIGUOUS,
        board=board,
        message="paused",
        observation=observation,
        before_position_id=board.position_id,
        frame_size=(216, 240),
        point_count=90,
        transition_evidence=evidence,
    )

    with pytest.raises(StageCCollectionError, match="internal tracker failure"):
        _record_terminal_update(
            update,
            board=board,
            quad=QUAD,
            expected_moves_uci=None,
            expected_final=None,
            scenario=StageCScenario.MULTIPLE_CANDIDATES,
            session_id="session-a",
            sample_id="internal-failure",
            output_root=_sample_root(tmp_path),
            client_size=(216, 240),
        )

    assert not _sample_root(tmp_path).exists()


def test_cli_runs_one_fake_atomic_event_end_to_end(tmp_path: Path) -> None:
    board = parse_fen(START)
    final, moves = _merged_position()
    baseline = _render(board)
    merged = _render(final)
    scripted = _ScriptedSource(
        (baseline, baseline.copy(), baseline.copy(), merged, merged.copy())
    )
    window = WindowInfo(
        hwnd=42,
        title="天天象棋",
        process_name="WeChatAppEx.exe",
        client_size=(216, 240),
    )

    exit_code = main(
        [
            "--fen",
            START,
            "--quad",
            "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540",
            "--expected-moves",
            *moves,
            "--scenario",
            "valid_two_ply",
            "--session-id",
            "cli-session",
            "--output-root",
            str(_sample_root(tmp_path)),
            "--timeout-seconds",
            "2",
        ],
        catalog=_Catalog((window,)),
        source_factory=lambda _window: scripted,
        allowed_local_root=_local_root(tmp_path),
    )

    assert exit_code == 0
    manifests = tuple(_sample_root(tmp_path).rglob("manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text("utf-8"))
    assert payload["observed_moves_uci"] == list(moves)
    pngs = tuple(manifests[0].parent.glob("*.png"))
    assert 2 <= len(pngs) <= 8
    assert all(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (48, 48, 4)
        for path in pngs
    )


def test_cli_refuses_zero_or_multiple_window_candidates(tmp_path: Path) -> None:
    window = WindowInfo(
        hwnd=42,
        title="天天象棋",
        process_name="WeChatAppEx.exe",
        client_size=(216, 240),
    )
    arguments = [
        "--fen",
        START,
        "--quad",
        "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540",
        "--expect-reject",
        "--scenario",
        "occlusion",
        "--session-id",
        "cli-session",
        "--output-root",
        str(_sample_root(tmp_path)),
    ]

    assert main(arguments, catalog=_Catalog(())) == 2
    assert main(arguments, catalog=_Catalog((window, window))) == 2
    assert not _sample_root(tmp_path).exists()
