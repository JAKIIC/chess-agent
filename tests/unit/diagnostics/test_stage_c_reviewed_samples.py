from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import cv2
import pytest

from tests.unit.diagnostics.test_stage_c_quarantine import (
    START,
    _occupancy,
    _two_ply_final,
)
from tests.unit.diagnostics.test_stage_c_samples import _crops, _sample
from xiangqi_agent.diagnostics.stage_c_review import StageCReviewOutcome
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import (
    ReviewedStageCSampleIntegrityError,
    ReviewedStageCSampleLoader,
    ReviewedStageCSampleV2,
    purge_expired_reviewed_samples,
)
from xiangqi_agent.diagnostics.stage_c_samples import HumanAiStageCSampleV1


def test_v2_schema_reuses_v1_evidence_validation_and_requires_provenance() -> None:
    sample = _v2()
    assert sample.schema_version == 2

    with pytest.raises(ValueError, match="64 lowercase"):
        replace(sample, source_event_manifest_sha256="BAD")
    with pytest.raises(ValueError, match="label_source"):
        replace(sample, label_source="observer_guess")
    with pytest.raises(ValueError, match="review outcome"):
        replace(sample, review_outcome=StageCReviewOutcome.DISCARDED)
    with pytest.raises(ValueError, match="verifier versions"):
        replace(sample, occupancy_verifier_version="")
    with pytest.raises(ValueError, match="promoted_at_utc"):
        replace(sample, promoted_at_utc="2026-09-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="schema_version"):
        replace(sample, schema_version=1)
    with pytest.raises(ValueError, match="exactly 90"):
        replace(sample, local_differences=(1.0,) * 89)


def test_v2_loader_round_trips_only_self_contained_provenance_and_crops(
    tmp_path: Path,
) -> None:
    sample_dir = _write_fixture(tmp_path, _v2())
    loaded = ReviewedStageCSampleLoader().load(sample_dir)
    encoded = _encoded_crops(loaded.metadata.changed_points)
    crop_hashes = _crop_hashes(encoded)
    source_bytes = _source_manifest_bytes(loaded.metadata, crop_hashes)
    review_bytes = _review_manifest_bytes(loaded.metadata, source_bytes)

    assert loaded.metadata == _v2()
    assert loaded.source_event_manifest_bytes == source_bytes
    assert loaded.review_manifest_bytes == review_bytes
    assert tuple(crop.point_index for crop in loaded.crops) == (22, 25, 67, 70)
    assert sorted(path.name for path in sample_dir.iterdir()) == [
        "manifest.json",
        "point-22-after.png",
        "point-22-before.png",
        "point-25-after.png",
        "point-25-before.png",
        "point-67-after.png",
        "point-67-before.png",
        "point-70-after.png",
        "point-70-before.png",
        "review-manifest.json",
        "source-event-manifest.json",
    ]


@pytest.mark.parametrize(
    "filename",
    (
        "source-event-manifest.json",
        "review-manifest.json",
        "point-22-before.png",
    ),
)
def test_v2_loader_rejects_any_changed_provenance_or_crop(
    tmp_path: Path,
    filename: str,
) -> None:
    sample_dir = _write_fixture(tmp_path, _v2())
    (sample_dir / filename).write_bytes(b"changed")

    with pytest.raises(ReviewedStageCSampleIntegrityError, match="hash"):
        ReviewedStageCSampleLoader().load(sample_dir)


def test_v2_loader_rejects_extra_files_and_path_id_mismatch(tmp_path: Path) -> None:
    sample_dir = _write_fixture(tmp_path, _v2())
    (sample_dir / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ReviewedStageCSampleIntegrityError, match="exactly"):
        ReviewedStageCSampleLoader().load(sample_dir)

    other = _write_fixture(tmp_path / "other", _v2())
    renamed = other.with_name("wrong-sample")
    other.rename(renamed)
    with pytest.raises(ReviewedStageCSampleIntegrityError, match="path"):
        ReviewedStageCSampleLoader().load(renamed)


def test_reviewed_cleanup_is_deterministic_and_preserves_frozen_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-c-reviewed"
    expired = _write_fixture(
        root,
        _v2(sample_id="expired", promoted="2026-07-31T23:59:59Z"),
    )
    protected = _write_fixture(
        root,
        _v2(sample_id="protected", promoted="2026-07-01T00:00:00Z"),
    )
    boundary = _write_fixture(
        root,
        _v2(sample_id="boundary", promoted="2026-08-02T00:00:00Z"),
    )

    removed = purge_expired_reviewed_samples(
        root,
        protected_relative_paths=frozenset({"session-1/protected"}),
        now_utc=datetime(2026, 9, 1, tzinfo=UTC),
        retention_days=30,
    )

    assert removed == (expired,)
    assert not expired.exists()
    assert protected.exists()
    assert boundary.exists()


def test_reviewed_cleanup_validates_every_sample_before_deleting_anything(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-c-reviewed"
    expired = _write_fixture(
        root,
        _v2(sample_id="expired", promoted="2026-07-01T00:00:00Z"),
    )
    malformed = _write_fixture(
        root,
        _v2(sample_id="malformed", promoted="2026-09-01T00:00:00Z"),
    )
    (malformed / "review-manifest.json").write_bytes(b"changed")

    with pytest.raises(ReviewedStageCSampleIntegrityError):
        purge_expired_reviewed_samples(
            root,
            protected_relative_paths=frozenset(),
            now_utc=datetime(2026, 9, 1, tzinfo=UTC),
        )
    assert expired.exists()


def test_reviewed_cleanup_requires_explicit_safe_protection_set(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage-c-reviewed"
    with pytest.raises(TypeError, match="frozenset"):
        purge_expired_reviewed_samples(  # type: ignore[arg-type]
            root,
            protected_relative_paths=set(),
            now_utc=datetime(2026, 9, 1, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="relative"):
        purge_expired_reviewed_samples(
            root,
            protected_relative_paths=frozenset({"../outside"}),
            now_utc=datetime(2026, 9, 1, tzinfo=UTC),
        )


@pytest.mark.parametrize(
    ("sidecar_name", "field", "changed"),
    (
        ("source-event-manifest.json", "event_id", "other-event"),
        ("source-event-manifest.json", "candidates", []),
        ("source-event-manifest.json", "window_title", "private-title"),
        ("review-manifest.json", "event_id", "other-event"),
        ("review-manifest.json", "account", "private-account"),
    ),
)
def test_v2_loader_rejects_semantically_rewritten_provenance_even_with_new_hash(
    tmp_path: Path,
    sidecar_name: str,
    field: str,
    changed: object,
) -> None:
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _record,
        _review_valid,
        _reviewed_root,
    )
    from tests.unit.diagnostics.test_stage_c_quarantine import _event
    from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService

    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    sample_dir = StageCPromotionService().promote(
        event_dir,
        review_path,
        _reviewed_root(tmp_path),
    )
    sidecar = sample_dir / sidecar_name
    payload = json.loads(sidecar.read_text("utf-8"))
    payload[field] = changed
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = sample_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    hash_field = (
        "source_event_manifest_sha256"
        if sidecar_name.startswith("source")
        else "review_manifest_sha256"
    )
    manifest[hash_field] = sha256(sidecar.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewedStageCSampleIntegrityError, match="provenance"):
        ReviewedStageCSampleLoader().load(sample_dir)


def test_v2_loader_rejects_coordinated_candidate_provenance_rewrite(
    tmp_path: Path,
) -> None:
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _record,
        _review_valid,
        _reviewed_root,
    )
    from tests.unit.diagnostics.test_stage_c_quarantine import _event
    from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService

    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    sample_dir = StageCPromotionService().promote(
        event_dir,
        review_path,
        _reviewed_root(tmp_path),
    )
    source_path = sample_dir / "source-event-manifest.json"
    source = json.loads(source_path.read_text("utf-8"))
    source["candidates"] = []
    source_path.write_text(
        json.dumps(source, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_path = sample_dir / "review-manifest.json"
    review = json.loads(review_path.read_text("utf-8"))
    review["event_manifest_sha256"] = sha256(source_path.read_bytes()).hexdigest()
    review_path.write_text(
        json.dumps(review, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = sample_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["source_event_manifest_sha256"] = sha256(
        source_path.read_bytes()
    ).hexdigest()
    manifest["review_manifest_sha256"] = sha256(review_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReviewedStageCSampleIntegrityError, match="provenance"):
        ReviewedStageCSampleLoader().load(sample_dir)


def _v2(
    *,
    sample_id: str = "stage-c-1",
    promoted: str = "2026-09-01T00:00:00Z",
) -> ReviewedStageCSampleV2:
    base = _sample(sample_id=sample_id)
    encoded = _encoded_crops(base.changed_points)
    source_bytes = _source_manifest_bytes(base, _crop_hashes(encoded))
    review_bytes = _review_manifest_bytes(base, source_bytes)
    return ReviewedStageCSampleV2(
        sample_id=base.sample_id,
        session_id=base.session_id,
        created_at_utc=base.created_at_utc,
        confirmed_fen=base.confirmed_fen,
        confirmed_position_id=base.confirmed_position_id,
        expected_outcome=base.expected_outcome,
        scenario=base.scenario,
        ground_truth_moves_uci=base.ground_truth_moves_uci,
        expected_final_position_id=base.expected_final_position_id,
        observed_status=base.observed_status,
        observed_moves_uci=base.observed_moves_uci,
        observed_final_position_id=base.observed_final_position_id,
        side_to_move=base.side_to_move,
        orientation=base.orientation,
        changed_points=base.changed_points,
        local_differences=base.local_differences,
        candidates=base.candidates,
        rejection_reasons=base.rejection_reasons,
        capture_context=base.capture_context,
        feature_version=base.feature_version,
        threshold_profile_version=base.threshold_profile_version,
        decision_latency_ms=base.decision_latency_ms,
        source_event_manifest_sha256=sha256(source_bytes).hexdigest(),
        review_manifest_sha256=sha256(review_bytes).hexdigest(),
        review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
        occupancy_verifier_version="circular-occupancy-v1",
        promotion_verifier_version="stage-c-promotion-v1",
        promoted_at_utc=promoted,
    )


def _write_fixture(root: Path, sample: ReviewedStageCSampleV2) -> Path:
    directory = root / sample.session_id / sample.sample_id
    directory.mkdir(parents=True)
    encoded = _encoded_crops(sample.changed_points)
    crop_hashes = _crop_hashes(encoded)
    source_bytes = _source_manifest_bytes(sample, crop_hashes)
    review_bytes = _review_manifest_bytes(sample, source_bytes)
    assert sha256(source_bytes).hexdigest() == sample.source_event_manifest_sha256
    assert sha256(review_bytes).hexdigest() == sample.review_manifest_sha256
    for filename, contents in encoded.items():
        (directory / filename).write_bytes(contents)
    (directory / "source-event-manifest.json").write_bytes(source_bytes)
    (directory / "review-manifest.json").write_bytes(review_bytes)
    payload = asdict(sample)
    payload["expected_outcome"] = sample.expected_outcome.value
    payload["scenario"] = sample.scenario.value
    payload["observed_status"] = sample.observed_status.value
    payload["orientation"] = sample.orientation.value
    payload["review_outcome"] = sample.review_outcome.value
    payload["crop_hashes"] = crop_hashes
    (directory / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def _encoded_crops(points: tuple[int, ...]) -> dict[str, bytes]:
    encoded: dict[str, bytes] = {}
    for crop in _crops(points):
        for suffix, pixels in (("before", crop.before), ("after", crop.after)):
            ok, buffer = cv2.imencode(".png", pixels)
            assert ok
            encoded[f"point-{crop.point_index:02d}-{suffix}.png"] = buffer.tobytes()
    return encoded


def _crop_hashes(encoded: dict[str, bytes]) -> dict[str, str]:
    return {
        name: sha256(contents).hexdigest()
        for name, contents in sorted(encoded.items())
    }


def _source_manifest_bytes(
    sample: HumanAiStageCSampleV1 | ReviewedStageCSampleV2,
    crop_hashes: dict[str, str],
) -> bytes:
    payload = {
        "event_id": sample.sample_id,
        "session_id": sample.session_id,
        "created_at_utc": sample.created_at_utc,
        "confirmed_fen": sample.confirmed_fen,
        "confirmed_position_id": sample.confirmed_position_id,
        "observed_status": sample.observed_status.value,
        "observed_moves_uci": list(sample.observed_moves_uci),
        "observed_final_position_id": sample.observed_final_position_id,
        "side_to_move": sample.side_to_move,
        "orientation": sample.orientation.value,
        "changed_points": list(sample.changed_points),
        "local_differences": list(sample.local_differences),
        "candidates": [asdict(candidate) for candidate in sample.candidates],
        "rejection_reasons": list(sample.rejection_reasons),
        "capture_context": asdict(sample.capture_context),
        "feature_version": sample.feature_version,
        "threshold_profile_version": sample.threshold_profile_version,
        "decision_latency_ms": sample.decision_latency_ms,
        "before_occupancy": asdict(_occupancy(START)),
        "after_occupancy": asdict(_occupancy(_two_ply_final(START))),
        "schema_version": 1,
        "crop_hashes": crop_hashes,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _review_manifest_bytes(
    sample: HumanAiStageCSampleV1 | ReviewedStageCSampleV2,
    source_bytes: bytes,
) -> bytes:
    outcome = (
        sample.review_outcome
        if isinstance(sample, ReviewedStageCSampleV2)
        else StageCReviewOutcome.CANDIDATE_CONFIRMED
    )
    payload = {
        "review_id": f"review-{sample.sample_id}",
        "event_id": sample.sample_id,
        "session_id": sample.session_id,
        "created_at_utc": sample.created_at_utc,
        "event_manifest_sha256": sha256(source_bytes).hexdigest(),
        "label_kind": "valid_two_ply",
        "moves_uci": list(sample.ground_truth_moves_uci),
        "expected_final_position_id": sample.expected_final_position_id,
        "scenario": None,
        "review_outcome": outcome.value,
        "supersedes_review_id": None,
        "reviewer_kind": "local_user",
        "ui_version": "stage-c-review-v1",
        "rules_version": "xiangqi-rules-v1",
        "schema_version": 1,
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
