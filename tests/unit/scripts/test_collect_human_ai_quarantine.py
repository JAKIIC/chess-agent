from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import cv2
import numpy as np
import pytest

from scripts.collect_human_ai_quarantine import (
    QuarantineCollectionError,
    QuarantineCollectionTimeout,
    _parse_args,
    collect_human_ai_quarantine_event,
    main,
)
from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureFrame, ClosedCallback, FrameCallback
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.platform.windows import WindowInfo
from xiangqi_agent.sync.live_session import LiveSyncStatus
from xiangqi_agent.vision.geometry import BoardGeometry, parse_normalized_quad
from xiangqi_agent.vision.occupancy import OccupancyEvidence

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
START = parse_fen(START_FEN)
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}
QUAD = parse_normalized_quad("0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540")


def test_unlabelled_collector_records_only_after_one_atomic_terminal_event(
    tmp_path: Path,
) -> None:
    final = _two_ply_final(START)
    source = _NotifyingFakeSource()
    observer = _SequenceOccupancyObserver(
        (_occupancy(START), _occupancy(START), _occupancy(final))
    )
    worker, updates, outcome = _launch(
        tmp_path,
        source,
        observer=observer,
        event_id="unknown-reply",
    )
    _establish_baseline(source, updates)
    assert not tuple(_quarantine_root(tmp_path).rglob("manifest.json"))

    merged = _render(final)
    source.push(merged, 150_000_000)
    source.push(merged.copy(), 260_000_000)
    worker.join(3.0)

    assert not worker.is_alive()
    assert "error" not in outcome
    event_dir = outcome["path"]
    assert isinstance(event_dir, Path)
    payload = json.loads((event_dir / "manifest.json").read_text("utf-8"))
    assert payload["observed_status"] == "accepted"
    assert payload["observed_moves_uci"] == ["h2e2", "h7e7"]
    assert {
        "expected_outcome",
        "ground_truth_moves_uci",
        "expected_final_position_id",
        "scenario",
    }.isdisjoint(payload)
    assert payload["before_occupancy"]["occupied"] == [
        piece != "." for piece in START.pieces
    ]
    assert payload["after_occupancy"]["occupied"] == [
        piece != "." for piece in final.pieces
    ]
    pngs = tuple(event_dir.glob("*.png"))
    assert 2 <= len(pngs) <= 8
    assert all(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED).shape == (48, 48, 4)
        for path in pngs
    )
    assert source.close_calls == 1


def test_unlabelled_collector_rejects_mismatched_baseline_without_output(
    tmp_path: Path,
) -> None:
    mismatched = list(_occupancy(START).occupied)
    mismatched[0] = not mismatched[0]
    source = _NotifyingFakeSource()
    observer = _SequenceOccupancyObserver(
        (OccupancyEvidence(tuple(mismatched), (0.95,) * 90, "literal-v1"),)
    )
    worker, _updates, outcome = _launch(
        tmp_path,
        source,
        observer=observer,
        event_id="bad-baseline",
    )
    baseline = _render(START)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    worker.join(3.0)

    assert isinstance(outcome.get("error"), QuarantineCollectionError)
    assert not tuple(_quarantine_root(tmp_path).rglob("manifest.json"))


def test_unlabelled_collector_timeout_and_window_close_write_nothing(
    tmp_path: Path,
) -> None:
    timeout_source = _NotifyingFakeSource()
    timeout_worker, timeout_updates, timeout_outcome = _launch(
        tmp_path,
        timeout_source,
        observer=_SequenceOccupancyObserver((_occupancy(START),)),
        event_id="timeout",
        timeout_seconds=0.15,
    )
    _establish_baseline(timeout_source, timeout_updates)
    timeout_worker.join(2.0)

    assert isinstance(timeout_outcome.get("error"), QuarantineCollectionTimeout)

    close_source = _NotifyingFakeSource()
    close_worker, close_updates, close_outcome = _launch(
        tmp_path,
        close_source,
        observer=_SequenceOccupancyObserver((_occupancy(START),)),
        event_id="closed",
    )
    _establish_baseline(close_source, close_updates)
    close_source.simulate_target_close()
    close_worker.join(3.0)

    assert isinstance(close_outcome.get("error"), QuarantineCollectionError)
    assert not tuple(_quarantine_root(tmp_path).rglob("manifest.json"))


def test_unlabelled_collector_aborts_on_any_frame_resize(tmp_path: Path) -> None:
    source = _NotifyingFakeSource()
    worker, updates, outcome = _launch(
        tmp_path,
        source,
        observer=_SequenceOccupancyObserver((_occupancy(START),)),
        event_id="resize",
    )
    _establish_baseline(source, updates)
    resized = np.repeat(np.repeat(_render(START), 2, axis=0), 2, axis=1)
    source.push(resized, 200_000_000)
    worker.join(3.0)

    assert isinstance(outcome.get("error"), QuarantineCollectionError)
    assert not tuple(_quarantine_root(tmp_path).rglob("manifest.json"))


@pytest.mark.parametrize(
    "forbidden",
    ("--expected-moves", "--expect-reject", "--scenario"),
)
def test_unlabelled_cli_has_no_truth_arguments(
    tmp_path: Path,
    forbidden: str,
) -> None:
    arguments = [
        "--fen",
        START_FEN,
        "--quad",
        "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540",
        "--session-id",
        "session-a",
        "--output-root",
        str(_quarantine_root(tmp_path)),
        forbidden,
        "value",
    ]

    with pytest.raises(SystemExit):
        _parse_args(arguments)


def test_cli_refuses_zero_or_multiple_visible_candidates(tmp_path: Path) -> None:
    window = WindowInfo(
        hwnd=42,
        title="天天象棋",
        process_name="WeChatAppEx.exe",
        client_size=(216, 240),
    )
    arguments = _cli_arguments(tmp_path)

    assert main(arguments, catalog=_Catalog(()), allowed_local_root=_local_root(tmp_path)) == 2
    assert (
        main(
            arguments,
            catalog=_Catalog((window, window)),
            allowed_local_root=_local_root(tmp_path),
        )
        == 2
    )
    assert not _quarantine_root(tmp_path).exists()


def test_cli_collects_one_event_without_printing_window_title_or_moves(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    final = _two_ply_final(START)
    frames = (
        _render(START),
        _render(START),
        _render(START),
        _render(final),
        _render(final),
    )
    source = _ScriptedSource(frames)
    window = WindowInfo(
        hwnd=42,
        title="天天象棋",
        process_name="WeChatAppEx.exe",
        client_size=(216, 240),
    )

    exit_code = main(
        _cli_arguments(tmp_path),
        catalog=_Catalog((window,)),
        source_factory=lambda _window: source,
        allowed_local_root=_local_root(tmp_path),
        occupancy_observer=_SequenceOccupancyObserver(
            (_occupancy(START), _occupancy(START), _occupancy(final))
        ),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "天天象棋" not in captured.out
    assert "h2e2" not in captured.out
    assert "h7e7" not in captured.out
    assert len(tuple(_quarantine_root(tmp_path).rglob("manifest.json"))) == 1


def _launch(
    tmp_path: Path,
    source: _NotifyingFakeSource,
    *,
    observer: _SequenceOccupancyObserver,
    event_id: str,
    timeout_seconds: float = 2.0,
):
    updates = []
    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["path"] = collect_human_ai_quarantine_event(
                source=source,
                board=START,
                quad=QUAD,
                session_id="session-a",
                output_root=_quarantine_root(tmp_path),
                timeout_seconds=timeout_seconds,
                on_update=updates.append,
                client_size=(216, 240),
                allowed_local_root=_local_root(tmp_path),
                event_id=event_id,
                occupancy_observer=observer,
                patch_size=CELL,
                settle_ms=100,
                stable_pairs=2,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            outcome["error"] = exc

    worker = Thread(target=run, daemon=True)
    worker.start()
    assert source.started.wait(2.0)
    return worker, updates, outcome


def _establish_baseline(
    source: _NotifyingFakeSource,
    updates: list[object],
) -> None:
    baseline = _render(START)
    source.push(baseline, 0)
    source.push(baseline.copy(), 50_000_000)
    source.push(baseline.copy(), 100_000_000)
    _wait_until(
        lambda: any(
            getattr(update, "status", None) is LiveSyncStatus.BASELINE_READY
            for update in updates
        )
    )


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


def _occupancy(board: BoardState) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (0.95,) * 90,
        "literal-v1",
    )


def _two_ply_final(board: BoardState) -> BoardState:
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    return apply_move(middle, second)


def _wait_until(predicate: object, *, timeout: float = 2.0) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if callable(predicate) and predicate():
            return
        sleep(0.01)
    raise AssertionError("timed out waiting for collection update")


def _local_root(tmp_path: Path) -> Path:
    return tmp_path / ".local"


def _quarantine_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-quarantine"


def _cli_arguments(tmp_path: Path) -> list[str]:
    return [
        "--fen",
        START_FEN,
        "--quad",
        "0.0558,0.0502;0.9488,0.0502;0.9488,0.9540;0.0558,0.9540",
        "--session-id",
        "cli-session",
        "--output-root",
        str(_quarantine_root(tmp_path)),
        "--timeout-seconds",
        "2",
    ]


class _NotifyingFakeSource(FakeFrameSource):
    def __init__(self) -> None:
        super().__init__(hwnd=42)
        self.started = Event()
        self.close_calls = 0

    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None:
        super().start(on_frame, on_closed)
        self.started.set()

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class _SequenceOccupancyObserver:
    def __init__(self, values: tuple[OccupancyEvidence, ...]) -> None:
        self._values = list(values)

    def observe(
        self,
        _frame: np.ndarray,
        _geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        if not self._values:
            raise AssertionError("occupancy observer received an unexpected call")
        return self._values.pop(0)


class _Catalog:
    def __init__(self, windows: tuple[WindowInfo, ...]) -> None:
        self._windows = windows

    def list_candidates(self) -> tuple[WindowInfo, ...]:
        return self._windows


class _ScriptedSource:
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        self._frames = frames
        self._closed = False

    def start(
        self,
        on_frame: FrameCallback,
        _on_closed: ClosedCallback | None = None,
    ) -> None:
        def emit() -> None:
            for index, frame in enumerate(self._frames):
                sleep(0.03)
                if self._closed:
                    return
                on_frame(CaptureFrame(index * 110_000_000, 42, frame))

        Thread(target=emit, daemon=True).start()

    def close(self) -> None:
        self._closed = True
