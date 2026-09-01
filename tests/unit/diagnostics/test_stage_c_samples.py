from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleRecorder,
    HumanAiStageCSampleV1,
    StageCCandidateRecord,
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import Orientation

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
START_ID = "132bdaf223100c4bd42ae8b81f0fb96c"
FINAL_ID = "d4c03fbf35e2cedd0dff78f5897df228"


def _context() -> CaptureContext:
    return CaptureContext(
        wgc_size=(2309, 1383),
        client_size=(1539, 922),
        dpi_scale=1.5,
        geometry_revision="quad-v1",
        theme_fingerprint="theme-fixed-v1",
        generation_id=1,
    )


def _candidate(
    *,
    moves_uci: tuple[str, str] = ("h2e2", "h7e7"),
    score: float = 20.0,
    final_position_id: str = FINAL_ID,
) -> StageCCandidateRecord:
    return StageCCandidateRecord(
        moves_uci=moves_uci,
        changed_points=(19, 22, 64, 67),
        expected_change_floor=20.0,
        unexpected_difference=1.0,
        maximum_template_distance=0.05,
        minimum_template_margin=0.1,
        minimum_template_confidence=0.9,
        score=score,
        final_position_id=final_position_id,
    )


def _sample(
    *,
    sample_id: str = "stage-c-1",
    session_id: str = "session-1",
    created_at_utc: str = "2026-09-01T00:00:00Z",
    expected_outcome: StageCExpectedOutcome = StageCExpectedOutcome.ACCEPT,
    scenario: StageCScenario = StageCScenario.VALID_TWO_PLY,
    ground_truth_moves_uci: tuple[str, ...] = ("h2e2", "h7e7"),
    expected_final_position_id: str | None = FINAL_ID,
    observed_status: StageCObservedStatus = StageCObservedStatus.ACCEPTED,
    observed_moves_uci: tuple[str, ...] = ("h2e2", "h7e7"),
    observed_final_position_id: str | None = FINAL_ID,
    changed_points: tuple[int, ...] = (19, 22, 64, 67),
    candidates: tuple[StageCCandidateRecord, ...] = (_candidate(),),
    rejection_reasons: tuple[str, ...] = (),
) -> HumanAiStageCSampleV1:
    return HumanAiStageCSampleV1(
        sample_id=sample_id,
        session_id=session_id,
        created_at_utc=created_at_utc,
        confirmed_fen=START,
        confirmed_position_id=START_ID,
        expected_outcome=expected_outcome,
        scenario=scenario,
        ground_truth_moves_uci=ground_truth_moves_uci,
        expected_final_position_id=expected_final_position_id,
        observed_status=observed_status,
        observed_moves_uci=observed_moves_uci,
        observed_final_position_id=observed_final_position_id,
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        changed_points=changed_points,
        local_differences=tuple(float(index) / 10.0 for index in range(90)),
        candidates=candidates,
        rejection_reasons=rejection_reasons,
        capture_context=_context(),
        feature_version="two-ply-template-v1",
        threshold_profile_version="human-ai-two-ply-v1",
        decision_latency_ms=125.0,
    )


def _rejected_sample(**changes: object) -> HumanAiStageCSampleV1:
    values: dict[str, object] = {
        "expected_outcome": StageCExpectedOutcome.REJECT,
        "scenario": StageCScenario.SELECTION_HIGHLIGHT,
        "ground_truth_moves_uci": (),
        "expected_final_position_id": None,
        "observed_status": StageCObservedStatus.REJECTED,
        "observed_moves_uci": (),
        "observed_final_position_id": START_ID,
        "changed_points": (64,),
        "candidates": (),
        "rejection_reasons": ("outside_change",),
    }
    values.update(changes)
    return _sample(**values)  # type: ignore[arg-type]


def _crops(points: tuple[int, ...] = (19, 22, 64, 67)) -> tuple[TransitionPointCrops, ...]:
    records = []
    for offset, point in enumerate(points):
        before = np.full((48, 48, 4), 20 + offset, dtype=np.uint8)
        after = np.full((48, 48, 4), 60 + offset, dtype=np.uint8)
        before[..., 3] = 255
        after[..., 3] = 255
        records.append(TransitionPointCrops(point, before, after))
    return tuple(records)


def test_valid_event_requires_exactly_two_ground_truth_moves() -> None:
    with pytest.raises(ValueError, match="two ground-truth moves"):
        replace(_sample(), ground_truth_moves_uci=())


def test_rejection_event_never_claims_an_expected_final_position() -> None:
    with pytest.raises(ValueError, match="rejection event"):
        replace(_rejected_sample(), expected_final_position_id=FINAL_ID)


def test_rejected_observation_exposes_no_moves_and_keeps_confirmed_position() -> None:
    with pytest.raises(ValueError, match="rejected observation"):
        replace(_rejected_sample(), observed_moves_uci=("h2e2", "h7e7"))
    with pytest.raises(ValueError, match="confirmed position"):
        replace(_rejected_sample(), observed_final_position_id=FINAL_ID)


def test_event_requires_ninety_finite_non_negative_local_differences() -> None:
    with pytest.raises(ValueError, match="exactly 90"):
        replace(_sample(), local_differences=(1.0,) * 89)
    with pytest.raises(ValueError, match="finite non-negative"):
        replace(_sample(), local_differences=(1.0,) * 89 + (float("nan"),))


def test_event_keeps_only_two_deterministically_ranked_candidates() -> None:
    runner = _candidate(moves_uci=("b2b3", "b7b6"), score=10.0, final_position_id="2" * 32)
    third = _candidate(moves_uci=("h0g2", "h9g7"), score=5.0, final_position_id="3" * 32)

    with pytest.raises(ValueError, match="at most two"):
        replace(_sample(), candidates=(_candidate(), runner, third))
    with pytest.raises(ValueError, match="ranked"):
        replace(_sample(), candidates=(runner, _candidate()))


def test_event_changed_points_are_one_to_four_unique_sorted_indices() -> None:
    with pytest.raises(ValueError, match="one through four"):
        replace(_sample(), changed_points=())
    with pytest.raises(ValueError, match="stable ascending"):
        replace(_sample(), changed_points=(22, 19))


def test_stage_c_recorder_is_disabled_by_default_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsDisabledError, match="explicitly enabled"):
        HumanAiStageCSampleRecorder(tmp_path).record(_sample(), _crops())

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_stage_c_recorder_writes_only_declared_small_crops_and_manifest(
    tmp_path: Path,
) -> None:
    sample_dir = HumanAiStageCSampleRecorder(tmp_path, enabled=True).record(
        _sample(),
        _crops(),
    )

    assert sorted(path.name for path in sample_dir.iterdir()) == [
        "manifest.json",
        "point-19-after.png",
        "point-19-before.png",
        "point-22-after.png",
        "point-22-before.png",
        "point-64-after.png",
        "point-64-before.png",
        "point-67-after.png",
        "point-67-before.png",
    ]
    manifest_text = (sample_dir / "manifest.json").read_text("utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 1
    assert manifest["expected_outcome"] == "accept"
    assert manifest["observed_status"] == "accepted"
    assert manifest["ground_truth_moves_uci"] == ["h2e2", "h7e7"]
    assert manifest["crop_hashes"] == dict(sorted(manifest["crop_hashes"].items()))
    lowered = manifest_text.lower()
    for forbidden in (
        "window_title",
        "nickname",
        "avatar",
        "account",
        "api_key",
        "deepseek_request",
        "full_frame",
    ):
        assert forbidden not in lowered


def test_stage_c_recorder_rejects_mismatched_crops_before_retention(tmp_path: Path) -> None:
    recorder = HumanAiStageCSampleRecorder(tmp_path, enabled=True, retention_days=7)
    old = recorder.record(
        _sample(sample_id="old", created_at_utc="2026-08-01T00:00:00Z"),
        _crops(),
    )

    with pytest.raises(ValueError, match="match changed_points"):
        recorder.record(
            _sample(sample_id="invalid", created_at_utc="2026-09-01T00:00:00Z"),
            _crops((19, 22)),
        )

    assert old.exists()


def test_stage_c_recorder_enforces_capacity_without_partial_output(tmp_path: Path) -> None:
    with pytest.raises(SampleQuotaExceededError, match="capacity"):
        HumanAiStageCSampleRecorder(tmp_path, enabled=True, max_bytes=100).record(
            _sample(),
            _crops(),
        )

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_duplicate_sample_is_rejected_before_it_can_trigger_retention(tmp_path: Path) -> None:
    recorder = HumanAiStageCSampleRecorder(tmp_path, enabled=True, retention_days=7)
    original = recorder.record(
        _sample(sample_id="duplicate", created_at_utc="2026-08-01T00:00:00Z"),
        _crops(),
    )
    original_manifest = (original / "manifest.json").read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        recorder.record(
            _sample(sample_id="duplicate", created_at_utc="2026-09-01T00:00:00Z"),
            _crops(),
        )

    assert (original / "manifest.json").read_bytes() == original_manifest


def test_retention_keeps_a_sample_created_exactly_seven_days_ago(tmp_path: Path) -> None:
    recorder = HumanAiStageCSampleRecorder(tmp_path, enabled=True, retention_days=7)
    boundary = recorder.record(
        _sample(sample_id="boundary", created_at_utc="2026-08-25T00:00:00Z"),
        _crops(),
    )

    recorder.record(
        _sample(sample_id="current", created_at_utc="2026-09-01T00:00:00Z"),
        _crops(),
    )

    assert boundary.exists()
