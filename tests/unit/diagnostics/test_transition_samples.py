from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.transition_samples import (
    TransitionPointCrops,
    TransitionSampleRecorder,
    TransitionSampleV2,
)
from xiangqi_agent.domain.board import Orientation

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _sample(
    *,
    sample_id: str = "transition-1",
    created_at_utc: str = "2026-09-01T00:00:00Z",
) -> TransitionSampleV2:
    return TransitionSampleV2(
        sample_id=sample_id,
        session_id="session-1",
        created_at_utc=created_at_utc,
        confirmed_fen=START,
        confirmed_position_id="132bdaf223100c4bd42ae8b81f0fb96c",
        final_position_id="1" * 32,
        moves_uci=("h2e2", "h7e7"),
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        changed_points=(19, 22, 64, 67),
        capture_context=CaptureContext(
            wgc_size=(216, 240),
            client_size=(216, 240),
            dpi_scale=1.0,
            geometry_revision="quad-v1",
            theme_fingerprint="theme-abc",
            generation_id=1,
        ),
        feature_version="two-ply-template-v1",
        threshold_profile_version="strict-v1",
        rejection_reasons=(),
    )


def _point_crops(points: tuple[int, ...] = (19, 22, 64, 67)) -> tuple[TransitionPointCrops, ...]:
    result = []
    for offset, point in enumerate(points):
        before = np.full((48, 48, 4), 20 + offset * 10, dtype=np.uint8)
        after = np.full((48, 48, 4), 60 + offset * 10, dtype=np.uint8)
        before[..., 3] = 255
        after[..., 3] = 255
        result.append(TransitionPointCrops(point, before, after))
    return tuple(result)


def test_transition_schema_requires_exactly_two_moves_and_two_to_four_points() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        replace(_sample(), moves_uci=("h2e2",))
    with pytest.raises(ValueError, match="two through four"):
        replace(_sample(), changed_points=(19,))
    with pytest.raises(ValueError, match="stable ascending"):
        replace(_sample(), changed_points=(22, 19))


def test_transition_recorder_is_disabled_by_default_and_writes_nothing(
    tmp_path: Path,
) -> None:
    with pytest.raises(DiagnosticsDisabledError, match="explicitly enabled"):
        TransitionSampleRecorder(tmp_path).record(_sample(), _point_crops())

    assert list(tmp_path.iterdir()) == []


def test_transition_recorder_writes_only_changed_point_crops_and_manifest(
    tmp_path: Path,
) -> None:
    sample_dir = TransitionSampleRecorder(tmp_path, enabled=True).record(
        _sample(),
        _point_crops(),
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
    assert manifest["schema_version"] == 2
    assert manifest["moves_uci"] == ["h2e2", "h7e7"]
    assert manifest["changed_points"] == [19, 22, 64, 67]
    assert "window_title" not in manifest_text
    assert "api_key" not in manifest_text
    assert "full_frame" not in manifest_text


def test_transition_recorder_rejects_crops_that_do_not_match_the_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="match changed_points"):
        TransitionSampleRecorder(tmp_path, enabled=True).record(
            _sample(),
            _point_crops((19, 22)),
        )

    assert list(tmp_path.iterdir()) == []


def test_invalid_transition_does_not_purge_existing_samples(tmp_path: Path) -> None:
    recorder = TransitionSampleRecorder(tmp_path, enabled=True, retention_days=7)
    old = recorder.record(
        _sample(
            sample_id="old-transition",
            created_at_utc="2026-08-01T00:00:00Z",
        ),
        _point_crops(),
    )
    crops = list(_point_crops())
    crops[0] = replace(crops[0], after=np.zeros((480, 640, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="48x48"):
        recorder.record(
            _sample(
                sample_id="invalid-transition",
                created_at_utc="2026-09-01T00:00:00Z",
            ),
            tuple(crops),
        )

    assert old.exists()


def test_transition_record_automatically_purges_expired_samples(tmp_path: Path) -> None:
    recorder = TransitionSampleRecorder(tmp_path, enabled=True, retention_days=7)
    old = recorder.record(
        _sample(
            sample_id="old-transition",
            created_at_utc="2026-08-01T00:00:00Z",
        ),
        _point_crops(),
    )

    current = recorder.record(
        _sample(
            sample_id="current-transition",
            created_at_utc="2026-09-01T00:00:00Z",
        ),
        _point_crops(),
    )

    assert not old.exists()
    assert current.exists()


def test_transition_recorder_enforces_capacity_before_writing(tmp_path: Path) -> None:
    recorder = TransitionSampleRecorder(tmp_path, enabled=True, max_bytes=100)

    with pytest.raises(SampleQuotaExceededError, match="capacity"):
        recorder.record(_sample(), _point_crops())

    assert list(tmp_path.iterdir()) == []


def test_transition_purge_requires_a_timezone_aware_clock(tmp_path: Path) -> None:
    recorder = TransitionSampleRecorder(tmp_path)
    naive = datetime(2026, 9, 1, tzinfo=UTC).replace(tzinfo=None)

    with pytest.raises(ValueError, match="timezone-aware"):
        recorder.purge_expired(naive)
