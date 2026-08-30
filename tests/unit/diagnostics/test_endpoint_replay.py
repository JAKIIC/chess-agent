from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_replay import (
    EndpointReplayer,
    EndpointReplayGate,
    EndpointSampleLoader,
    ReplayThresholds,
    SampleIntegrityError,
)
from xiangqi_agent.diagnostics.endpoint_samples import (
    EndpointCrops,
    EndpointSampleRecorder,
    EndpointSampleV1,
    SampleKind,
)
from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.vision.endpoint_features import InstanceTransferExtractor

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _patch(value: int) -> np.ndarray:
    patch = np.full((48, 48, 4), value, dtype=np.uint8)
    patch[..., 3] = 255
    patch[14:34, 20:28, :3] = (20, 60, 180)
    return patch


def _crops() -> EndpointCrops:
    return EndpointCrops(
        source_before=_patch(30),
        source_after=np.full((48, 48, 4), (30, 30, 30, 255), dtype=np.uint8),
        target_before=np.full((48, 48, 4), (180, 180, 180, 255), dtype=np.uint8),
        target_after=_patch(180),
    )


def _sample() -> EndpointSampleV1:
    return EndpointSampleV1(
        sample_id="sample-1",
        session_id="session-1",
        sample_kind=SampleKind.MOVE,
        created_at_utc="2026-08-30T12:00:00Z",
        confirmed_fen=START,
        confirmed_position_id="132bdaf223100c4bd42ae8b81f0fb96c",
        actual_uci="h2e2",
        probe_uci="h2e2",
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        source_index=70,
        target_index=67,
        top_k_candidates=({"uci": "h2e2", "score": 12.0},),
        rejection_reasons=(),
        capture_context=CaptureContext(
            (2313, 1385),
            (2313, 1385),
            1.0,
            "quad-v1",
            "theme-1",
            1,
        ),
        feature_version="instance-transfer-v1",
        threshold_profile_version="test-v1",
        change_scores={"source": 10.0, "target": 11.0, "outside": 1.0},
    )


def _record(tmp_path: Path) -> Path:
    return EndpointSampleRecorder(tmp_path, enabled=True).record(_sample(), _crops())


def test_loader_round_trips_sanitized_metadata_and_four_crops(tmp_path: Path) -> None:
    sample_dir = _record(tmp_path)

    loaded = EndpointSampleLoader().load(sample_dir)

    assert loaded.metadata == _sample()
    assert np.array_equal(loaded.crops.source_before, _crops().source_before)
    assert np.array_equal(loaded.crops.target_after, _crops().target_after)


def test_loader_rejects_changed_crop_hash(tmp_path: Path) -> None:
    sample_dir = _record(tmp_path)
    (sample_dir / "source_before.png").write_bytes(b"changed")

    with pytest.raises(SampleIntegrityError, match="hash"):
        EndpointSampleLoader().load(sample_dir)


def test_loader_rejects_an_extra_image_even_when_manifest_is_unchanged(tmp_path: Path) -> None:
    sample_dir = _record(tmp_path)
    (sample_dir / "full-frame.png").write_bytes(b"not allowed")

    with pytest.raises(SampleIntegrityError, match="exactly four endpoint crops"):
        EndpointSampleLoader().load(sample_dir)


def test_same_sample_and_versions_replay_identically(tmp_path: Path) -> None:
    sample_dir = _record(tmp_path)
    replayer = EndpointReplayer(
        InstanceTransferExtractor(max_shift=3),
        EndpointReplayGate(
            ReplayThresholds(
                max_instance_distance=1.0,
                min_source_change=0.0,
                min_target_change=0.0,
                profile_version="permissive-test-v1",
            )
        ),
    )

    first = replayer.replay(sample_dir)
    second = replayer.replay(sample_dir)

    assert first.without_runtime() == second.without_runtime()
    assert first.accepted
    assert first.result_fen is not None
    assert first.result_fen != START


def test_rejected_replay_never_generates_a_new_fen(tmp_path: Path) -> None:
    sample_dir = _record(tmp_path)
    replayer = EndpointReplayer(
        InstanceTransferExtractor(max_shift=3),
        EndpointReplayGate(
            ReplayThresholds(
                max_instance_distance=0.0,
                min_source_change=1.0,
                min_target_change=1.0,
                profile_version="reject-test-v1",
            )
        ),
    )

    result = replayer.replay(sample_dir)

    assert not result.accepted
    assert result.rejection_reasons
    assert result.result_fen is None
