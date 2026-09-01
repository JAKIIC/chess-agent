from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_human_ai_stage_c import main as evaluate_main
from scripts.freeze_human_ai_stage_c import main as freeze_main
from tests.unit.diagnostics.test_stage_c_promotion import _reviewed_root
from tests.unit.scripts.test_freeze_human_ai_stage_c import (
    _record as _record_reviewed_valid,
)
from tests.unit.scripts.test_freeze_human_ai_stage_c import (
    _record_rejection as _record_reviewed_rejection,
)
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario

REJECTION_SCENARIOS = (
    StageCScenario.MULTIPLE_CANDIDATES,
    StageCScenario.SELECTION_HIGHLIGHT,
    StageCScenario.CONTINUOUS_ANIMATION,
    StageCScenario.OCCLUSION,
    StageCScenario.RESIZE,
    StageCScenario.THREE_PLY,
)


def _record_valid(root: Path, index: int) -> Path:
    return _record_reviewed_valid(
        root,
        sample_id=f"valid-{index:03d}",
        session_id=f"valid-session-{index:03d}",
    )


def _record_rejection(root: Path, index: int) -> Path:
    return _record_reviewed_rejection(
        root,
        sample_id=f"reject-{index:03d}",
        session_id=f"reject-session-{index:03d}",
        scenario=REJECTION_SCENARIOS[index % len(REJECTION_SCENARIOS)],
    )


def _freeze(root: Path) -> Path:
    assert freeze_main([str(root), "--output", "frozen-stage-c.json"]) == 0
    return root / "frozen-stage-c.json"


def test_metric_failure_writes_auditable_privacy_safe_report_and_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _reviewed_root(tmp_path)
    _record_valid(root, 0)
    manifest = _freeze(root)
    capsys.readouterr()
    report_path = tmp_path / "report.json"

    exit_code = evaluate_main([str(manifest), "--output", str(report_path)])

    assert exit_code == 1
    report = json.loads(report_path.read_text("utf-8"))
    assert report["release_pass"] is False
    assert report["metrics"]["valid_samples"] == 1
    assert "minimum_rejection_events" in report["reasons"]
    serialized = json.dumps(report, sort_keys=True)
    assert "confirmed_fen" not in serialized
    assert "h2e2" not in serialized
    assert "valid-session-000/valid-000" not in serialized
    assert json.loads(capsys.readouterr().out)["release_pass"] is False


def test_changed_sample_manifest_is_an_integrity_error_and_returns_two(
    tmp_path: Path,
) -> None:
    root = _reviewed_root(tmp_path)
    sample_dir = _record_valid(root, 0)
    manifest = _freeze(root)
    sample_manifest = sample_dir / "manifest.json"
    sample_manifest.write_bytes(sample_manifest.read_bytes() + b"\n")
    report_path = tmp_path / "report.json"

    exit_code = evaluate_main([str(manifest), "--output", str(report_path)])

    assert exit_code == 2
    assert not report_path.exists()


def test_frozen_parent_traversal_path_is_an_integrity_error(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record_valid(root, 0)
    manifest = _freeze(root)
    payload = json.loads(manifest.read_text("utf-8"))
    payload["samples"][0]["relative_path"] = "../outside/sample"
    manifest.write_text(json.dumps(payload), "utf-8")
    report_path = tmp_path / "report.json"

    exit_code = evaluate_main([str(manifest), "--output", str(report_path)])

    assert exit_code == 2
    assert not report_path.exists()


def test_frozen_duplicate_sample_path_is_an_integrity_error(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record_valid(root, 0)
    _record_valid(root, 1)
    manifest = _freeze(root)
    payload = json.loads(manifest.read_text("utf-8"))
    payload["samples"][1]["relative_path"] = payload["samples"][0]["relative_path"]
    manifest.write_text(json.dumps(payload), "utf-8")
    report_path = tmp_path / "report.json"

    exit_code = evaluate_main([str(manifest), "--output", str(report_path)])

    assert exit_code == 2
    assert not report_path.exists()


def test_evaluator_never_overwrites_the_frozen_manifest(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    _record_valid(root, 0)
    manifest = _freeze(root)
    original = manifest.read_bytes()

    exit_code = evaluate_main([str(manifest), "--output", str(manifest)])

    assert exit_code == 2
    assert manifest.read_bytes() == original


def test_complete_real_frozen_dataset_passes_and_returns_zero(tmp_path: Path) -> None:
    root = _reviewed_root(tmp_path)
    for index in range(30):
        _record_valid(root, index)
        _record_rejection(root, index)
    manifest = _freeze(root)
    report_path = tmp_path / "report.json"

    exit_code = evaluate_main([str(manifest), "--output", str(report_path)])

    assert exit_code == 0
    report = json.loads(report_path.read_text("utf-8"))
    assert report["release_pass"] is True
    assert report["reasons"] == []
    assert report["metrics"]["valid_samples"] == 30
    assert report["metrics"]["distinct_valid_sessions"] == 30
    assert report["metrics"]["rejection_samples"] == 30
    assert report["metrics"]["false_accepts"] == 0
    assert report["metrics"]["coverage"] == 1.0
