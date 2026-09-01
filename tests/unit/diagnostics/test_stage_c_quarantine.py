from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    QuarantinedStageCEventV1,
    QuarantineEventIntegrityError,
    QuarantineEventLoader,
    QuarantineEventRecorder,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCObservedStatus,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.vision.occupancy import OccupancyEvidence

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
START = parse_fen(START_FEN)
START_ID = "132bdaf223100c4bd42ae8b81f0fb96c"


def test_quarantine_schema_and_manifest_contain_observation_but_no_truth(
    tmp_path: Path,
) -> None:
    event = _event()
    event_dir = QuarantineEventRecorder(tmp_path, enabled=True).record(event, _crops())
    payload = json.loads((event_dir / "manifest.json").read_text("utf-8"))
    forbidden = {
        "expected_outcome",
        "ground_truth_moves_uci",
        "expected_final_position_id",
        "label_kind",
        "scenario",
        "passed",
        "accepted_as_truth",
    }

    assert payload["observed_status"] == "accepted"
    assert payload["observed_moves_uci"] == ["h2e2", "h7e7"]
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(field.name for field in fields(QuarantinedStageCEventV1))


def test_quarantine_event_requires_consistent_observation_and_occupancy_versions() -> None:
    with pytest.raises(ValueError, match="accepted observation"):
        replace(_event(), observed_moves_uci=())
    with pytest.raises(ValueError, match="rejected observation"):
        replace(_rejected_event(), observed_moves_uci=("h2e2", "h7e7"))
    with pytest.raises(ValueError, match="algorithm versions"):
        replace(
            _event(),
            after_occupancy=replace(
                _event().after_occupancy,
                algorithm_version="other-v1",
            ),
        )


def test_quarantine_event_requires_bounded_sorted_numeric_evidence() -> None:
    with pytest.raises(ValueError, match="one through four"):
        replace(_event(), changed_points=())
    with pytest.raises(ValueError, match="stable ascending"):
        replace(_event(), changed_points=(25, 22))
    with pytest.raises(ValueError, match="exactly 90"):
        replace(_event(), local_differences=(1.0,) * 89)
    with pytest.raises(ValueError, match="finite non-negative"):
        replace(_event(), local_differences=(1.0,) * 89 + (float("nan"),))

    second = _candidate(moves=("b2b3", "b7b6"), score=10.0, final_id="2" * 32)
    third = _candidate(moves=("h0g2", "h9g7"), score=5.0, final_id="3" * 32)
    with pytest.raises(ValueError, match="at most two"):
        replace(_event(), candidates=(_candidate(), second, third))
    with pytest.raises(ValueError, match="ranked"):
        replace(_event(), candidates=(second, _candidate()))


def test_quarantine_recorder_is_disabled_by_default(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsDisabledError, match="explicitly enabled"):
        QuarantineEventRecorder(tmp_path).record(_event(), _crops())

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_quarantine_round_trip_writes_only_declared_small_crops(tmp_path: Path) -> None:
    source_crops = _crops()
    event_dir = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(),
        source_crops,
    )

    assert sorted(path.name for path in event_dir.iterdir()) == [
        "manifest.json",
        "point-22-after.png",
        "point-22-before.png",
        "point-25-after.png",
        "point-25-before.png",
        "point-67-after.png",
        "point-67-before.png",
        "point-70-after.png",
        "point-70-before.png",
    ]
    loaded = QuarantineEventLoader().load(event_dir)
    assert loaded.metadata == _event()
    assert loaded.directory == event_dir
    assert loaded.manifest_bytes == (event_dir / "manifest.json").read_bytes()
    assert tuple(crop.point_index for crop in loaded.crops) == (22, 25, 67, 70)
    assert all(crop.before.shape == (48, 48, 4) for crop in loaded.crops)
    assert all(not crop.before.flags.writeable for crop in loaded.crops)
    assert all(not crop.after.flags.writeable for crop in loaded.crops)

    source_crops[0].before[:] = 0
    assert int(loaded.crops[0].before[0, 0, 0]) == 20


def test_quarantine_loader_rejects_tampered_crop_and_extra_file(tmp_path: Path) -> None:
    first = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(event_id="tampered"),
        _crops(),
    )
    (first / "point-22-before.png").write_bytes(b"changed")
    with pytest.raises(QuarantineEventIntegrityError, match="hash mismatch"):
        QuarantineEventLoader().load(first)

    second = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(event_id="extra"),
        _crops(),
    )
    (second / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(QuarantineEventIntegrityError, match="exactly"):
        QuarantineEventLoader().load(second)


def test_quarantine_loader_rejects_unknown_truth_field_and_malformed_json(
    tmp_path: Path,
) -> None:
    first = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(event_id="truth-field"),
        _crops(),
    )
    manifest_path = first / "manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    payload["ground_truth_moves_uci"] = ["h2e2", "h7e7"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QuarantineEventIntegrityError, match="unexpected"):
        QuarantineEventLoader().load(first)

    second = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(event_id="bad-json"),
        _crops(),
    )
    (second / "manifest.json").write_bytes(b"\xff")
    with pytest.raises(QuarantineEventIntegrityError, match="UTF-8 JSON"):
        QuarantineEventLoader().load(second)


@pytest.mark.parametrize(
    "nested_key",
    ("capture_context", "before_occupancy", "candidate"),
)
def test_quarantine_loader_rejects_unknown_nested_private_fields(
    tmp_path: Path,
    nested_key: str,
) -> None:
    event_dir = QuarantineEventRecorder(tmp_path, enabled=True).record(
        _event(event_id=f"nested-{nested_key}"),
        _crops(),
    )
    manifest_path = event_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text("utf-8"))
    target = (
        payload["candidates"][0]
        if nested_key == "candidate"
        else payload[nested_key]
    )
    target["window_title"] = "private title"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QuarantineEventIntegrityError, match="unexpected"):
        QuarantineEventLoader().load(event_dir)


def test_invalid_or_duplicate_event_does_not_purge_existing_evidence(tmp_path: Path) -> None:
    recorder = QuarantineEventRecorder(tmp_path, enabled=True, retention_days=7)
    old = recorder.record(
        _event(event_id="old", created_at="2026-08-01T00:00:00Z"),
        _crops(),
    )
    original = (old / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="match changed_points"):
        recorder.record(
            _event(event_id="invalid", created_at="2026-09-01T00:00:00Z"),
            _crops((22, 25)),
        )
    with pytest.raises(FileExistsError, match="already exists"):
        recorder.record(
            _event(event_id="old", created_at="2026-09-01T00:00:00Z"),
            _crops(),
        )

    assert old.exists()
    assert (old / "manifest.json").read_bytes() == original


def test_quarantine_recorder_enforces_capacity_without_partial_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(SampleQuotaExceededError, match="capacity"):
        QuarantineEventRecorder(tmp_path, enabled=True, max_bytes=100).record(
            _event(),
            _crops(),
        )

    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_quarantine_recorder_rolls_back_when_temporary_verification_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_temporary_event(
        _loader: QuarantineEventLoader,
        _event_dir: Path,
    ) -> object:
        raise QuarantineEventIntegrityError("forced temporary verification failure")

    monkeypatch.setattr(QuarantineEventLoader, "load", reject_temporary_event)

    with pytest.raises(QuarantineEventIntegrityError, match="forced"):
        QuarantineEventRecorder(tmp_path, enabled=True).record(_event(), _crops())

    assert not tmp_path.exists() or not tuple(tmp_path.rglob("*"))


def test_quarantine_loader_rejects_a_symlinked_event_directory(tmp_path: Path) -> None:
    real = QuarantineEventRecorder(tmp_path / "real-root", enabled=True).record(
        _event(),
        _crops(),
    )
    alias = tmp_path / "event-alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(QuarantineEventIntegrityError, match="does not exist"):
        QuarantineEventLoader().load(alias)


def test_quarantine_event_rejects_path_unsafe_ids_and_non_zulu_time() -> None:
    with pytest.raises(ValueError, match="path-safe"):
        replace(_event(), event_id="../private")
    with pytest.raises(ValueError, match="ending in Z"):
        replace(_event(), created_at_utc="2026-09-01T00:00:00+00:00")


def test_quarantine_retention_keeps_boundary_and_removes_older_event(
    tmp_path: Path,
) -> None:
    recorder = QuarantineEventRecorder(tmp_path, enabled=True, retention_days=7)
    expired = recorder.record(
        _event(event_id="expired", created_at="2026-08-24T23:59:59Z"),
        _crops(),
    )
    boundary = recorder.record(
        _event(event_id="boundary", created_at="2026-08-25T00:00:00Z"),
        _crops(),
    )

    recorder.record(
        _event(event_id="current", created_at="2026-09-01T00:00:00Z"),
        _crops(),
    )

    assert not expired.exists()
    assert boundary.exists()


def test_quarantine_manifest_strings_do_not_contain_private_window_data(
    tmp_path: Path,
) -> None:
    event_dir = QuarantineEventRecorder(tmp_path, enabled=True).record(_event(), _crops())
    text = (event_dir / "manifest.json").read_text("utf-8").lower()

    for forbidden in (
        "window_title",
        "nickname",
        "avatar",
        "account",
        "api_key",
        "deepseek_request",
        "full_frame",
        "lenovo",
    ):
        assert forbidden not in text


def _event(
    *,
    event_id: str = "event-1",
    created_at: str = "2026-09-01T00:00:00Z",
) -> QuarantinedStageCEventV1:
    final = _two_ply_final(START)
    return QuarantinedStageCEventV1(
        event_id=event_id,
        session_id="session-1",
        created_at_utc=created_at,
        confirmed_fen=START_FEN,
        confirmed_position_id=START_ID,
        observed_status=StageCObservedStatus.ACCEPTED,
        observed_moves_uci=("h2e2", "h7e7"),
        observed_final_position_id=final.position_id,
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        changed_points=(22, 25, 67, 70),
        local_differences=tuple(float(index + 1) / 10.0 for index in range(90)),
        candidates=(_candidate(final_id=final.position_id),),
        rejection_reasons=(),
        capture_context=_context(),
        feature_version="two-ply-template-v1",
        threshold_profile_version="human-ai-two-ply-v1",
        decision_latency_ms=125.0,
        before_occupancy=_occupancy(START),
        after_occupancy=_occupancy(final),
    )


def _rejected_event() -> QuarantinedStageCEventV1:
    return replace(
        _event(),
        observed_status=StageCObservedStatus.REJECTED,
        observed_moves_uci=(),
        observed_final_position_id=START_ID,
        changed_points=(67,),
        candidates=(),
        rejection_reasons=("candidate_margin",),
        after_occupancy=_occupancy(START),
    )


def _candidate(
    *,
    moves: tuple[str, str] = ("h2e2", "h7e7"),
    score: float = 20.0,
    final_id: str | None = None,
) -> StageCCandidateRecord:
    return StageCCandidateRecord(
        moves_uci=moves,
        changed_points=(22, 25, 67, 70),
        expected_change_floor=20.0,
        unexpected_difference=1.0,
        maximum_template_distance=0.05,
        minimum_template_margin=0.1,
        minimum_template_confidence=0.9,
        score=score,
        final_position_id=final_id or _two_ply_final(START).position_id,
    )


def _context() -> CaptureContext:
    return CaptureContext(
        wgc_size=(2309, 1383),
        client_size=(1539, 922),
        dpi_scale=1.5,
        geometry_revision="quad-v1",
        theme_fingerprint="theme-fixed-v1",
        generation_id=1,
    )


def _occupancy(board: BoardState) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (0.95,) * 90,
        "circular-occupancy-v1",
    )


def _two_ply_final(board: BoardState) -> BoardState:
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    return apply_move(middle, second)


def _crops(
    points: tuple[int, ...] = (22, 25, 67, 70),
) -> tuple[TransitionPointCrops, ...]:
    result = []
    for offset, point in enumerate(points):
        before = np.full((48, 48, 4), 20 + offset, dtype=np.uint8)
        after = np.full((48, 48, 4), 60 + offset, dtype=np.uint8)
        before[..., 3] = 255
        after[..., 3] = 255
        result.append(TransitionPointCrops(point, before, after))
    return tuple(result)
