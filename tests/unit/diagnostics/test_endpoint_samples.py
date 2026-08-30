from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    EndpointCrops,
    EndpointSampleRecorder,
    EndpointSampleV1,
    SampleKind,
    SampleQuotaExceededError,
)
from xiangqi_agent.domain.board import Orientation

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _sample(*, sample_id: str = "sample-1", session_id: str = "session-1") -> EndpointSampleV1:
    return EndpointSampleV1(
        sample_id=sample_id,
        session_id=session_id,
        sample_kind=SampleKind.MOVE,
        created_at_utc="2026-08-30T12:00:00Z",
        confirmed_fen=START,
        confirmed_position_id="2af65e72bfc2c5c2a697f39c986df391",
        actual_uci="i0h0",
        probe_uci="i0h0",
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        source_index=89,
        target_index=88,
        top_k_candidates=({"uci": "i0h0", "score": 12.5},),
        rejection_reasons=(),
        capture_context=CaptureContext(
            wgc_size=(2313, 1385),
            client_size=(2313, 1385),
            dpi_scale=1.0,
            geometry_revision="quad-v1",
            theme_fingerprint="theme-abc",
            generation_id=1,
        ),
        feature_version="rgb-v1",
        threshold_profile_version="strict-v1",
        change_scores={"source": 14.1, "target": 27.7, "outside": 1.6},
    )


def _crops() -> EndpointCrops:
    arrays = [np.full((48, 48, 4), value, dtype=np.uint8) for value in (20, 40, 60, 80)]
    for array in arrays:
        array[..., 3] = 255
    return EndpointCrops(*arrays)


def test_recorder_writes_only_four_small_crops_and_sanitized_manifest(tmp_path: Path) -> None:
    sample_dir = EndpointSampleRecorder(tmp_path, enabled=True).record(_sample(), _crops())

    assert sorted(path.name for path in sample_dir.iterdir()) == [
        "manifest.json",
        "source_after.png",
        "source_before.png",
        "target_after.png",
        "target_before.png",
    ]
    manifest_text = (sample_dir / "manifest.json").read_text("utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 1
    assert manifest["sample_id"] == "sample-1"
    assert set(manifest["crop_hashes"]) == {
        "source_after.png",
        "source_before.png",
        "target_after.png",
        "target_before.png",
    }
    assert "window_title" not in manifest_text
    assert "api_key" not in manifest_text
    for image_path in sample_dir.glob("*.png"):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        assert image is not None
        assert image.shape == (48, 48, 4)


def test_recorder_is_disabled_by_default_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsDisabledError, match="explicitly enabled"):
        EndpointSampleRecorder(tmp_path).record(_sample(), _crops())

    assert list(tmp_path.iterdir()) == []


def test_recorder_rejects_any_crop_that_could_be_a_full_frame(tmp_path: Path) -> None:
    crops = _crops()
    oversized = np.zeros((480, 640, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="48x48"):
        EndpointSampleRecorder(tmp_path, enabled=True).record(
            _sample(),
            replace(crops, target_after=oversized),
        )

    assert list(tmp_path.iterdir()) == []


def test_recorder_rejects_new_sample_when_capacity_would_be_exceeded(tmp_path: Path) -> None:
    recorder = EndpointSampleRecorder(tmp_path, enabled=True, max_bytes=100)

    with pytest.raises(SampleQuotaExceededError, match="capacity"):
        recorder.record(_sample(), _crops())

    assert list(tmp_path.iterdir()) == []


def test_recorder_rejects_path_traversal_in_identifiers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="identifier"):
        EndpointSampleRecorder(tmp_path, enabled=True).record(
            _sample(session_id="../outside"),
            _crops(),
        )


def test_delete_session_removes_only_matching_samples(tmp_path: Path) -> None:
    recorder = EndpointSampleRecorder(tmp_path, enabled=True)
    first = recorder.record(_sample(sample_id="sample-1", session_id="session-1"), _crops())
    second = recorder.record(_sample(sample_id="sample-2", session_id="session-2"), _crops())

    removed = recorder.delete_session("session-1")

    assert removed == 1
    assert not first.exists()
    assert second.exists()


def test_delete_all_removes_samples_but_preserves_the_configured_root(tmp_path: Path) -> None:
    recorder = EndpointSampleRecorder(tmp_path, enabled=True)
    recorder.record(_sample(sample_id="sample-1"), _crops())
    recorder.record(_sample(sample_id="sample-2"), _crops())

    assert recorder.delete_all() == 2
    assert tmp_path.exists()
    assert list(tmp_path.iterdir()) == []
