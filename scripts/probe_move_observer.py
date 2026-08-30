from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import replace
from queue import Empty, Queue
from time import monotonic, perf_counter_ns

from xiangqi_agent.capture.adaptive_sampling import AdaptiveBurstSampler
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.capture.windows_capture_source import WindowsCaptureSource
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.platform.windows import WindowsWindowCatalog, select_window
from xiangqi_agent.sync.evidence import MoveProposal
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.tracker import StableMoveTracker, TrackingStatus
from xiangqi_agent.vision.change_detection import FrameStabilityDetector
from xiangqi_agent.vision.geometry import BoardGeometry, parse_normalized_quad
from xiangqi_agent.vision.templates import PieceTemplateBank

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def main() -> int:
    args = _parse_args()
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    try:
        window = select_window(WindowsWindowCatalog().list_candidates(), args.hwnd)
        source = WindowsCaptureSource(window, fps=args.capture_fps)
        source.start(events.put, events.put)
        try:
            baseline, geometry = _stable_baseline(events, args)
            board = _parse_board(args.fen, geometry.orientation)
            templates = PieceTemplateBank.from_position(
                board,
                geometry,
                baseline.bgra,
                patch_size=args.patch_size,
            )
            tracker = StableMoveTracker(
                board,
                geometry,
                LegalMoveDiffObserver(patch_size=args.patch_size),
                required_stable_pairs=args.stable_pairs,
                patch_size=args.patch_size,
            )
            tracker.initialize(baseline.bgra)
            sampler = AdaptiveBurstSampler(
                steady_fps=args.fps,
                settle_ms=args.settle_ms,
                stable_repeats=args.stable_pairs,
            )
            sampler.initialize(baseline)
            print(
                json.dumps(
                    {
                        "status": "BASELINE_READY",
                        "frame_size": baseline.size,
                        "point_count": len(geometry.grid_points()),
                        "template_classes": len(templates.symbols),
                        "orientation": geometry.orientation,
                        "position_id": board.position_id,
                        "capture_fps": args.capture_fps,
                        "steady_fps": args.fps,
                        "settle_ms": args.settle_ms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            deadline_ns = perf_counter_ns() + round(args.seconds * 1_000_000_000)
            while True:
                samples = _next_adaptive_samples(
                    events,
                    sampler,
                    deadline_ns=deadline_ns,
                )
                if samples is None:
                    break
                for sample in samples:
                    update = tracker.push(sample.bgra)
                    if update.status is TrackingStatus.WAITING_FOR_STABLE:
                        sampler.set_bursting(True)
                    elif update.status is TrackingStatus.WATCHING:
                        sampler.set_bursting(False)
                    if update.status not in (
                        TrackingStatus.ACCEPTED,
                        TrackingStatus.PAUSED_AMBIGUOUS,
                    ):
                        continue
                    observation = update.observation
                    result: dict[str, object] = {
                        "status": update.status,
                        "confirmed_fen": update.board.fen,
                        "confirmed_position_id": update.board.position_id,
                    }
                    if observation is not None:
                        result.update(_proposal_details(observation))
                    if update.move is not None:
                        result["uci"] = update.move.uci
                        result["chinese"] = to_chinese(board, update.move)
                        result["before_fen"] = board.fen
                        result["before_position_id"] = board.position_id
                    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
                    return 0 if update.status is TrackingStatus.ACCEPTED else 3
            print(json.dumps({"status": "TIMEOUT_NO_MOVE"}, ensure_ascii=False), flush=True)
            return 4
        finally:
            source.close()
    except (CaptureClosedError, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
            flush=True,
        )
        return 2


def _stable_baseline(
    events: Queue[CaptureFrame | CaptureClosedError], args: argparse.Namespace
) -> tuple[CaptureFrame, BoardGeometry]:
    deadline = monotonic() + args.baseline_timeout
    first = _next_event(events, deadline)
    geometry = BoardGeometry.from_quad(
        parse_normalized_quad(args.quad),
        first.size,
        orientation=Orientation(args.orientation),
    )
    detector = FrameStabilityDetector(
        geometry,
        required_stable_pairs=args.stable_pairs,
    )
    detector.update(first.bgra)
    latest = first
    while True:
        sample = _next_tick(
            events,
            latest,
            tick_seconds=1.0 / args.fps,
            deadline=deadline,
        )
        if sample is None:
            break
        latest = sample
        change = detector.update(latest.bgra)
        if change is not None and change.stable:
            return latest, geometry
    raise RuntimeError("board did not become stable before the baseline timeout")


def _next_event(
    events: Queue[CaptureFrame | CaptureClosedError], deadline: float
) -> CaptureFrame:
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise RuntimeError("capture timed out")
    try:
        event = events.get(timeout=remaining)
    except Empty as exc:
        raise RuntimeError("capture timed out") from exc
    if isinstance(event, CaptureClosedError):
        raise event
    return event


def _next_tick(
    events: Queue[CaptureFrame | CaptureClosedError],
    latest: CaptureFrame,
    *,
    tick_seconds: float,
    deadline: float,
) -> CaptureFrame | None:
    """At one clock tick, collapse WGC callbacks to the newest frame."""
    tick_deadline = min(monotonic() + tick_seconds, deadline)
    newest = latest
    while True:
        remaining = tick_deadline - monotonic()
        if remaining <= 0:
            return None if monotonic() >= deadline else newest
        try:
            event = events.get(timeout=remaining)
        except Empty:
            return None if monotonic() >= deadline else newest
        if isinstance(event, CaptureClosedError):
            raise event
        newest = event


def _next_adaptive_samples(
    events: Queue[CaptureFrame | CaptureClosedError],
    sampler: AdaptiveBurstSampler,
    *,
    deadline_ns: int,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> tuple[CaptureFrame, ...] | None:
    while True:
        event: CaptureFrame | CaptureClosedError
        try:
            event = events.get_nowait()
        except Empty:
            now_ns = clock_ns()
            if now_ns >= deadline_ns:
                return None
            due_ns = sampler.next_due_ns()
            wake_ns = deadline_ns if due_ns is None else min(deadline_ns, due_ns)
            remaining_seconds = max(0.0, (wake_ns - now_ns) / 1_000_000_000)
            if remaining_seconds == 0.0:
                try:
                    event = events.get_nowait()
                except Empty:
                    clock_samples = sampler.on_clock(now_ns)
                    if clock_samples:
                        return clock_samples
                    continue
            else:
                try:
                    event = events.get(timeout=remaining_seconds)
                except Empty:
                    continue
        batch: list[CaptureFrame | CaptureClosedError] = [event]
        while True:
            try:
                batch.append(events.get_nowait())
            except Empty:
                break
        close_error = next(
            (item for item in batch if isinstance(item, CaptureClosedError)),
            None,
        )
        if close_error is not None:
            raise close_error
        emitted_samples: list[CaptureFrame] = []
        for frame in (item for item in batch if isinstance(item, CaptureFrame)):
            emitted_samples.extend(sampler.on_frame(frame))
        if emitted_samples:
            return tuple(emitted_samples)


def _parse_board(fen: str, orientation: Orientation) -> BoardState:
    return replace(parse_fen(fen), orientation=orientation)


def _proposal_details(proposal: MoveProposal) -> dict[str, object]:
    return {
        "evidence_score": round(proposal.evidence_score, 4),
        "rejection_reasons": list(proposal.evidence.rejection_reasons),
        "top_candidates": [
            {
                "uci": candidate.move.uci,
                "score": round(candidate.score, 4),
                "source_difference": round(candidate.source_difference, 4),
                "destination_difference": round(candidate.destination_difference, 4),
                "unexpected_difference": round(candidate.unexpected_difference, 4),
                "semantic_distance": round(
                    max(
                        candidate.source_expected_distance,
                        candidate.destination_expected_distance,
                    ),
                    4,
                ),
                "semantic_margin": round(candidate.semantic_margin, 4),
                "semantic_evidence_score": round(
                    min(
                        candidate.source_semantic_evidence_score,
                        candidate.destination_semantic_evidence_score,
                    ),
                    4,
                ),
            }
            for candidate in proposal.evidence.candidates[:5]
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Observe one unique legal move from a visible board")
    parser.add_argument("--hwnd", type=int, required=True)
    parser.add_argument("--quad", required=True)
    parser.add_argument("--fen", default=START_FEN)
    parser.add_argument(
        "--orientation",
        choices=tuple(item.value for item in Orientation),
        default=Orientation.RED_BOTTOM.value,
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--capture-fps", type=int, default=20)
    parser.add_argument("--settle-ms", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--baseline-timeout", type=float, default=10.0)
    parser.add_argument("--stable-pairs", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=48)
    args = parser.parse_args()
    if (
        args.fps <= 0
        or args.capture_fps <= 0
        or args.settle_ms <= 0
        or args.seconds <= 0
        or args.baseline_timeout <= 0
    ):
        parser.error("capture rates and timeouts must be positive")
    if args.stable_pairs <= 0 or args.patch_size <= 0:
        parser.error("stable-pairs and patch-size must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
