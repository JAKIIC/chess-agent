from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path, PurePosixPath

import pytest

from scripts.freeze_human_ai_stage_c import main
from tests.unit.diagnostics.test_stage_c_samples import _crops, _sample
from xiangqi_agent.diagnostics.stage_c_samples import HumanAiStageCSampleRecorder

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


def _record(
    root: Path,
    *,
    sample_id: str,
    session_id: str,
    feature: str | None = None,
    profile: str | None = None,
) -> Path:
    sample = _sample(sample_id=sample_id, session_id=session_id)
    if feature is not None:
        sample = replace(sample, feature_version=feature)
    if profile is not None:
        sample = replace(sample, threshold_profile_version=profile)
    return HumanAiStageCSampleRecorder(root, enabled=True).record(sample, _crops())


def test_freeze_cli_writes_sorted_portable_hash_locked_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path / "samples"
    second = _record(root, sample_id="sample-b", session_id="session-b")
    first = _record(root, sample_id="sample-a", session_id="session-a")

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 0
    output = root / "frozen-stage-c.json"
    payload = json.loads(output.read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["feature_version"] == "two-ply-template-v1"
    assert set(payload["threshold_profile"]) == THRESHOLD_FIELDS
    assert payload["threshold_profile"] == {
        "max_template_distance": 0.18,
        "max_unexpected_difference": 3.0,
        "min_local_difference": 5.0,
        "min_margin": 5.0,
        "min_score": 5.0,
        "min_template_confidence": 0.8,
        "min_template_margin": 0.02,
        "profile_version": "human-ai-two-ply-v1",
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
    root = tmp_path / "samples"
    _record(root, sample_id="sample-a", session_id="session-a")
    output = root / "frozen-stage-c.json"
    output.write_bytes(b"keep-this-exact-file")

    exit_code = main([str(root), "--output", output.name])

    assert exit_code == 2
    assert output.read_bytes() == b"keep-this-exact-file"


def test_freeze_cli_refuses_absolute_and_parent_traversal_output_names(
    tmp_path: Path,
) -> None:
    root = tmp_path / "samples"
    _record(root, sample_id="sample-a", session_id="session-a")

    absolute_code = main([str(root), "--output", str(tmp_path / "outside.json")])
    traversal_code = main([str(root), "--output", "../outside.json"])

    assert absolute_code == 2
    assert traversal_code == 2
    assert not (tmp_path / "outside.json").exists()


def test_freeze_cli_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    root = tmp_path / "samples"
    _record(root, sample_id="same-id", session_id="session-a")
    _record(root, sample_id="same-id", session_id="session-b")

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_a_directory_renamed_away_from_its_anonymous_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / "samples"
    sample_dir = _record(root, sample_id="sample-a", session_id="session-a")
    renamed = root / "renamed-session" / sample_dir.name
    renamed.parent.mkdir()
    sample_dir.rename(renamed)

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_mixed_feature_versions(tmp_path: Path) -> None:
    root = tmp_path / "samples"
    _record(root, sample_id="sample-a", session_id="session-a")
    _record(
        root,
        sample_id="sample-b",
        session_id="session-b",
        feature="two-ply-template-v2",
    )

    exit_code = main([str(root), "--output", "frozen-stage-c.json"])

    assert exit_code == 2
    assert not (root / "frozen-stage-c.json").exists()


def test_freeze_cli_rejects_mixed_threshold_profile_versions(tmp_path: Path) -> None:
    root = tmp_path / "samples"
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
    root = tmp_path / "samples"
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
