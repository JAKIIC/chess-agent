from __future__ import annotations

import json
import shutil
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath

import pytest

from scripts.freeze_human_ai_stage_c import main
from tests.unit.diagnostics.test_stage_c_promotion import (
    _record as record_quarantine_event,
)
from tests.unit.diagnostics.test_stage_c_promotion import (
    _rejection_event,
    _review_rejection,
    _review_valid,
    _reviewed_root,
)
from tests.unit.diagnostics.test_stage_c_quarantine import _event
from tests.unit.diagnostics.test_stage_c_samples import _crops, _sample
from xiangqi_agent.diagnostics.stage_c_gate import (
    DEFAULT_STAGE_C_FEATURE_VERSION,
    DEFAULT_STAGE_C_THRESHOLD_PROFILE,
    StageCGateIntegrityError,
    freeze_human_ai_stage_c,
    freeze_reviewed_human_ai_stage_c,
)
from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleRecorder,
    StageCScenario,
)

THRESHOLD_FIELDS = {
    "min_local_difference",
    "max_unexpected_difference",
    "min_score",
    "min_margin",
    "max_template_distance",
    "min_template_margin",
    "min_template_confidence",
    "profile_version",
}


def test_legacy_v1_freeze_remains_available_as_a_read_only_api(tmp_path: Path) -> None:
    root = tmp_path / "legacy-samples"
    sample = _sample(sample_id="legacy", session_id="legacy-session")
    HumanAiStageCSampleRecorder(root, enabled=True).record(sample, _crops())

    output = freeze_human_ai_stage_c(
        root,
        "legacy-frozen.json",
        feature_version="two-ply-template-v1",
        threshold_profile=replace(
            DEFAULT_STAGE_C_THRESHOLD_PROFILE,
            profile_version="human-ai-two-ply-v1",
        ),
        created_at_utc="2026-09-01T00:00:00Z",
    )

    assert output.is_file()
    assert json.loads(output.read_text("utf-8"))["samples"][0]["sample_id"] == "legacy"


def _record(
    root: Path,
    *,
    sample_id: str,
    session_id: str,
    feature: str | None = None,
    profile: str | None = None,
) -> Path:
    source_root = root.parent.parent / "stage-c-test-sources" / session_id / sample_id
    event = replace(
        _event(),
        event_id=sample_id,
        session_id=session_id,
        feature_version=feature or DEFAULT_STAGE_C_FEATURE_VERSION,
        threshold_profile_version=(
            profile or DEFAULT_STAGE_C_THRESHOLD_PROFILE.profile_version
        ),
    )
    event_dir = record_quarantine_event(source_root, event)
    review_path = _review_valid(source_root, event_dir)
    return StageCPromotionService().promote(event_dir, review_path, root)


def _record_rejection(
    root: Path,
    *,
    sample_id: str,
    session_id: str,
    scenario: StageCScenario,
) -> Path:
    source_root = root.parent.parent / "stage-c-test-sources" / session_id / sample_id
    event, moves = _rejection_event(scenario)
    if scenario is StageCScenario.MULTIPLE_CANDIDATES:
        first, second = event.candidates
        event = replace(
            event,
            candidates=(first, replace(second, score=first.score - 1.0)),
            rejection_reasons=("candidate_margin",),
        )
    elif scenario is StageCScenario.THREE_PLY:
        event = replace(event, rejection_reasons=("no_legal_candidates",))
    event = replace(
        event,
        event_id=sample_id,
        session_id=session_id,
        feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
        threshold_profile_version=DEFAULT_STAGE_C_THRESHOLD_PROFILE.profile_version,
    )
    event_dir = record_quarantine_event(source_root, event)
    review_path = _review_rejection(source_root, event_dir, scenario, moves)
    return StageCPromotionService().promote(event_dir, review_path, root)


def test_freeze_cli_writes_sorted_portable_hash_locked_manifest(
    tmp_path: Path,
) -> None:
    root = _reviewed_root(tmp_path)
    second = _record(root, sample_id="sample-b", session_id="session-b")
    first = _record(root, sample_id="sample-a", session_id="session-a")

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 0
    output = root / "frozen-stage-c.json"
    payload = json.loads(output.read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["feature_version"] == DEFAULT_STAGE_C_FEATURE_VERSION
    assert set(payload["threshold_profile"]) == THRESHOLD_FIELDS
    assert payload["threshold_profile"] == {
        "max_template_distance": 0.18,
        "max_unexpected_difference": 3.0,
        "min_local_difference": 5.0,
        "min_margin": 5.0,
        "min_score": 5.0,
        "min_template_confidence": 0.8,
        "min_template_margin": 0.02,
        "profile_version": "human-ai-two-ply-v2",
    }
    paths = [entry["relative_path"] for entry in payload["samples"]]
    assert paths == sorted(paths)
    assert paths == ["session-a/sample-a", "session-b/sample-b"]
    assert all(not PurePosixPath(path).is_absolute() and ".." not in path for path in paths)
    assert all("\\" not in path for path in paths)
    hashes = {entry["sample_id"]: entry["manifest_sha256"] for entry in payload["samples"]}
    assert hashes["sample-a"] == sha256((first / "manifest.json").read_bytes()).hexdigest()
    assert hashes["sample-b"] == sha256((second / "manifest.json").read_bytes()).hexdigest()


def test_freeze_cli_refuses_to_replace_an_existing_output(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="sample-a", session_id="session-a")
    output = root / "frozen-stage-c.json"
    output.write_bytes(b"keep-this-exact-file")

    exit_code = main([str(root), "--output", output.name])

    assert exit_code == 2
    assert output.read_bytes() == b"keep-this-exact-file"


def test_freeze_cli_refuses_absolute_and_parent_traversal_output_names(
    tmp_path: Path,
) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="sample-a", session_id="session-a")

    absolute_code = main([str(root), "--output", str(tmp_path / "outside.json")])
    traversal_code = main([str(root), "--output", "../outside.json"])

    assert absolute_code == 2
    assert traversal_code == 2
    assert not (tmp_path / "outside.json").exists()


def test_freeze_cli_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="same-id", session_id="session-a")
    _record(root, sample_id="same-id", session_id="session-b")

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_a_directory_renamed_away_from_its_anonymous_ids(
    tmp_path: Path,
) -> None:
    root = _reviewed_root(tmp_path)
    sample_dir = _record(root, sample_id="sample-a", session_id="session-a")
    renamed = root / "renamed-session" / sample_dir.name
    renamed.parent.mkdir()
    sample_dir.rename(renamed)

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_mixed_feature_versions(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="sample-a", session_id="session-a")
    _record(
        root,
        sample_id="sample-b",
        session_id="session-b",
        feature="two-ply-template-v1",
    )

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_mixed_threshold_profile_versions(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="sample-a", session_id="session-a")
    _record(
        root,
        sample_id="sample-b",
        session_id="session-b",
        profile="other-profile",
    )

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_has_no_runtime_threshold_override_arguments(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record(root, sample_id="sample-a", session_id="session-a")

    with pytest.raises(SystemExit) as error:
        main(
            [
                str(root),
                "--output",
                "frozen-stage-c.json",
                "--min-score",
                "0",
            ]
        )

    assert error.value.code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_reviewed_only_freeze_accepts_promoted_v2_and_is_self_contained(
    tmp_path: Path,
) -> None:
    event_dir = record_quarantine_event(
        tmp_path,
        replace(
            _event(),
            feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
            threshold_profile_version=(
                DEFAULT_STAGE_C_THRESHOLD_PROFILE.profile_version
            ),
        ),
    )
    review_path = _review_valid(tmp_path, event_dir)
    reviewed = _reviewed_root(tmp_path)
    sample = StageCPromotionService().promote(event_dir, review_path, reviewed)

    output = freeze_reviewed_human_ai_stage_c(
        reviewed,
        "reviewed-frozen.json",
        created_at_utc="2026-09-01T00:00:00Z",
    )
    payload = json.loads(output.read_text("utf-8"))

    assert payload["samples"][0]["relative_path"] == "session-1/event-1"
    assert payload["samples"][0]["manifest_sha256"] == sha256(
        (sample / "manifest.json").read_bytes()
    ).hexdigest()

    # The frozen sample remains replayable after its mutable source roots disappear.
    shutil.rmtree(event_dir.parent.parent)
    shutil.rmtree(review_path.parents[2])
    assert sample.exists()


def test_reviewed_only_freeze_rejects_v1_quarantine_and_mutated_provenance(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy" / "stage-c-reviewed"
    legacy = _sample(sample_id="legacy", session_id="session-legacy")
    HumanAiStageCSampleRecorder(legacy_root, enabled=True).record(legacy, _crops())
    with pytest.raises(StageCGateIntegrityError, match="V2"):
        freeze_reviewed_human_ai_stage_c(legacy_root, "frozen.json")

    event_dir = record_quarantine_event(
        tmp_path / "source",
        replace(_event(), feature_version=DEFAULT_STAGE_C_FEATURE_VERSION),
    )
    with pytest.raises(StageCGateIntegrityError, match="reviewed"):
        freeze_reviewed_human_ai_stage_c(
            event_dir.parent.parent,
            "frozen.json",
        )

    review_path = _review_valid(tmp_path / "source", event_dir)
    reviewed = _reviewed_root(tmp_path / "source")
    sample = StageCPromotionService().promote(event_dir, review_path, reviewed)
    sidecar = sample / "source-event-manifest.json"
    sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(StageCGateIntegrityError, match="provenance"):
        freeze_reviewed_human_ai_stage_c(reviewed, "frozen.json")


def test_reviewed_only_freeze_rejects_unknown_manifest_placement(
    tmp_path: Path,
) -> None:
    reviewed = tmp_path / ".local" / "stage-c-reviewed"
    nested = reviewed / "session" / "sample" / "nested"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(StageCGateIntegrityError, match="placement"):
        freeze_reviewed_human_ai_stage_c(reviewed, "frozen.json")


def test_reviewed_only_freeze_rejects_unknown_root_entries(tmp_path: Path) -> None:
    reviewed = _reviewed_root(tmp_path)
    _record(reviewed, sample_id="sample-a", session_id="session-a")
    (reviewed / "unexpected.txt").write_text("not a sample", encoding="utf-8")

    with pytest.raises(StageCGateIntegrityError, match="layout"):
        freeze_reviewed_human_ai_stage_c(reviewed, "frozen.json")
