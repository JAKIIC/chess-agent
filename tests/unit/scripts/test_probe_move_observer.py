import sys
from argparse import Namespace
from pathlib import Path
from queue import Queue
from time import monotonic

import numpy as np
import pytest

import scripts.probe_move_observer as probe
from xiangqi_agent.capture.adaptive_sampling import AdaptiveBurstSampler
from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame
from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops
from xiangqi_agent.domain.board import Move, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.sync.evidence import (
    CandidateEvidence,
    MoveEvidence,
    MoveProposal,
    ObservationStatus,
)
from xiangqi_agent.vision.endpoint_features import EndpointFeatures


def _frame(timestamp_ns: int = 1, *, shape: tuple[int, int] = (2, 2)) -> CaptureFrame:
    return CaptureFrame(timestamp_ns, 1, np.zeros((*shape, 4), dtype=np.uint8))


def test_quiet_capture_reuses_the_latest_frame_as_a_stability_tick() -> None:
    latest = _frame()

    result = probe._next_tick(Queue(), latest, tick_seconds=0.001, deadline=monotonic() + 1.0)

    assert result is latest


def test_observation_deadline_returns_none_instead_of_a_capture_error() -> None:
    latest = _frame()

    assert probe._next_tick(Queue(), latest, tick_seconds=0.001, deadline=monotonic()) is None


def test_capture_close_is_not_mistaken_for_a_quiet_tick() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(CaptureClosedError("closed"))

    with pytest.raises(CaptureClosedError, match="closed"):
        probe._next_tick(events, _frame(), tick_seconds=0.001, deadline=monotonic() + 1.0)


def test_tick_collapses_a_queued_burst_to_the_newest_frame() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(2))
    events.put(_frame(3))

    result = probe._next_tick(
        events,
        _frame(1),
        tick_seconds=0.001,
        deadline=monotonic() + 1.0,
    )

    assert result is not None
    assert result.timestamp_ns == 3


def test_tick_prioritizes_a_close_event_behind_queued_frames() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(2))
    events.put(CaptureClosedError("closed after frame"))

    with pytest.raises(CaptureClosedError, match="closed after frame"):
        probe._next_tick(
            events,
            _frame(1),
            tick_seconds=0.001,
            deadline=monotonic() + 1.0,
        )


def test_static_single_callback_can_establish_a_clocked_baseline() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    frame = CaptureFrame(1, 1, np.zeros((240, 216, 4), dtype=np.uint8))
    events.put(frame)
    args = Namespace(
        quad="0.05,0.05;0.95,0.05;0.95,0.95;0.05,0.95",
        stable_pairs=2,
        baseline_timeout=0.2,
        fps=100,
        orientation="red_bottom",
    )

    baseline, geometry = probe._stable_baseline(events, args)

    assert baseline is frame
    assert len(geometry.grid_points()) == 90


def test_black_bottom_orientation_is_shared_by_board_and_geometry() -> None:
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    pixels = np.zeros((240, 216, 4), dtype=np.uint8)
    for timestamp in (1, 2, 3):
        events.put(CaptureFrame(timestamp, 1, pixels))
    args = Namespace(
        quad="0.05,0.05;0.95,0.05;0.95,0.95;0.05,0.95",
        stable_pairs=2,
        baseline_timeout=0.2,
        fps=100,
        orientation="black_bottom",
    )

    _, geometry = probe._stable_baseline(events, args)
    board = probe._parse_board(probe.START_FEN, geometry.orientation)

    assert geometry.orientation is Orientation.BLACK_BOTTOM
    assert board.orientation is geometry.orientation


def test_adaptive_event_preserves_a_quiet_endpoint_before_the_next_callback() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    changed = _frame(50_000_000)
    sampler.initialize(_frame(0))
    sampler.on_frame(changed)
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    next_change = _frame(170_000_000)
    events.put(next_change)

    samples = probe._next_adaptive_samples(
        events,
        sampler,
        deadline_ns=1_000_000_000,
        clock_ns=lambda: 60_000_000,
    )

    assert samples == (changed, changed, changed, next_change)


def test_adaptive_quiet_deadline_promotes_the_buffered_steady_callback() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    changed = _frame(50_000_000)
    sampler.initialize(_frame(0))
    sampler.on_frame(changed)

    samples = probe._next_adaptive_samples(
        Queue(),
        sampler,
        deadline_ns=200_000_000,
        clock_ns=lambda: 150_000_000,
    )

    assert samples == (changed, changed, changed)


def test_adaptive_event_prioritizes_a_queued_close_over_buffered_endpoint() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    changed = _frame(50_000_000)
    sampler.initialize(_frame(0))
    sampler.on_frame(changed)
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(170_000_000))
    events.put(CaptureClosedError("closed after endpoint"))

    with pytest.raises(CaptureClosedError, match="closed after endpoint"):
        probe._next_adaptive_samples(
            events,
            sampler,
            deadline_ns=1_000_000_000,
            clock_ns=lambda: 60_000_000,
        )


def test_adaptive_event_rejects_a_queued_resize_before_buffered_endpoint() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    changed = _frame(50_000_000)
    sampler.initialize(_frame(0))
    sampler.on_frame(changed)
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(170_000_000))
    events.put(_frame(180_000_000, shape=(3, 2)))

    with pytest.raises(ValueError, match="size changed"):
        probe._next_adaptive_samples(
            events,
            sampler,
            deadline_ns=1_000_000_000,
            clock_ns=lambda: 60_000_000,
        )


def test_adaptive_clock_prioritizes_an_already_queued_close_at_settle_due() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    sampler.initialize(_frame(0))
    sampler.on_frame(_frame(50_000_000))
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(CaptureClosedError("closed at settle due"))

    with pytest.raises(CaptureClosedError, match="closed at settle due"):
        probe._next_adaptive_samples(
            events,
            sampler,
            deadline_ns=200_000_000,
            clock_ns=lambda: 150_000_000,
        )


def test_adaptive_clock_rejects_an_already_queued_resize_at_settle_due() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, settle_ms=100, stable_repeats=2)
    sampler.initialize(_frame(0))
    sampler.on_frame(_frame(50_000_000))
    events: Queue[CaptureFrame | CaptureClosedError] = Queue()
    events.put(_frame(60_000_000, shape=(3, 2)))

    with pytest.raises(ValueError, match="size changed"):
        probe._next_adaptive_samples(
            events,
            sampler,
            deadline_ns=200_000_000,
            clock_ns=lambda: 150_000_000,
        )


def test_probe_defaults_to_high_rate_capture_with_two_fps_steady_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe_move_observer.py",
            "--hwnd",
            "1",
            "--quad",
            "0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9",
        ],
    )

    args = probe._parse_args()

    assert args.fps == 2
    assert args.capture_fps == 20
    assert args.settle_ms == 100
    assert not args.record_endpoints


def test_probe_serializes_evidence_without_probability_like_confidence_names() -> None:
    candidate = CandidateEvidence(
        move=Move("i0h0", 89, 88),
        source_difference=14.0,
        destination_difference=27.0,
        unexpected_difference=1.0,
        source_expected_distance=0.1,
        destination_expected_distance=0.12,
        semantic_margin=0.03,
        source_semantic_evidence_score=0.9,
        destination_semantic_evidence_score=0.8,
        score=13.0,
    )
    proposal = MoveProposal(
        status=ObservationStatus.AMBIGUOUS,
        move=None,
        evidence_score=0.8,
        evidence=MoveEvidence((candidate,), (0.0,) * 90, ("candidate_margin",)),
    )

    payload = probe._proposal_details(proposal)

    assert payload["evidence_score"] == 0.8
    assert payload["rejection_reasons"] == ["candidate_margin"]
    assert payload["top_candidates"][0]["semantic_evidence_score"] == 0.8
    assert "confidence" not in str(payload)


def _recording_proposal() -> MoveProposal:
    candidate = CandidateEvidence(
        move=Move("h2e2", 70, 67),
        source_difference=14.0,
        destination_difference=27.0,
        unexpected_difference=1.0,
        source_expected_distance=0.1,
        destination_expected_distance=0.1,
        semantic_margin=0.03,
        source_semantic_evidence_score=0.9,
        destination_semantic_evidence_score=0.9,
        score=13.0,
    )
    features = EndpointFeatures(
        feature_version="instance-transfer-v1",
        instance_distance=0.1,
        instance_evidence_score=0.8,
        color_distance=0.1,
        gradient_distance=0.1,
        source_change_distance=0.3,
        target_change_distance=0.3,
        best_shift=(0, 0),
    )
    return MoveProposal(
        status=ObservationStatus.ACCEPTED,
        move=candidate.move,
        evidence_score=0.8,
        evidence=MoveEvidence((candidate,), (0.0,) * 90, (), features),
    )


def _endpoint_crops() -> EndpointCrops:
    values = [np.full((48, 48, 4), value, dtype=np.uint8) for value in (20, 40, 60, 80)]
    for value in values:
        value[..., 3] = 255
    return EndpointCrops(*values)


def _capture_context() -> CaptureContext:
    return CaptureContext((200, 300), (200, 300), 1.0, "quad-v1", "theme-v1", 1)


def test_probe_does_not_write_endpoint_samples_without_explicit_opt_in(tmp_path: Path) -> None:
    args = Namespace(
        record_endpoints=False,
        sample_root=tmp_path,
        session_id=None,
        actual_uci="h2e2",
        sample_kind="move",
    )

    sample_id = probe._maybe_record_endpoint_sample(
        args,
        parse_fen(probe.START_FEN),
        _recording_proposal(),
        _endpoint_crops(),
        _capture_context(),
    )

    assert sample_id is None
    assert list(tmp_path.iterdir()) == []


def test_probe_opt_in_writes_only_one_four_crop_sample(tmp_path: Path) -> None:
    args = Namespace(
        record_endpoints=True,
        sample_root=tmp_path,
        session_id="session-1",
        actual_uci="h2e2",
        sample_kind="move",
    )

    sample_id = probe._maybe_record_endpoint_sample(
        args,
        parse_fen(probe.START_FEN),
        _recording_proposal(),
        _endpoint_crops(),
        _capture_context(),
    )

    assert sample_id is not None
    files = sorted(path.name for path in (tmp_path / "session-1" / sample_id).iterdir())
    assert files == [
        "manifest.json",
        "source_after.png",
        "source_before.png",
        "target_after.png",
        "target_before.png",
    ]
