from __future__ import annotations

import json
from pathlib import Path

from scripts.replay_endpoint_samples import main
from tests.unit.diagnostics.test_endpoint_replay import _record


def test_replay_cli_outputs_deterministic_feature_and_decision_json(
    tmp_path: Path,
    capsys: object,
) -> None:
    sample_dir = _record(tmp_path)

    exit_code = main(
        [
            str(sample_dir),
            "--max-instance-distance",
            "1.0",
            "--min-source-change",
            "0.0",
            "--min-target-change",
            "0.0",
            "--profile-version",
            "cli-test-v1",
        ]
    )
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert exit_code == 0
    assert output["sample_id"] == "sample-1"
    assert output["feature_version"] == "instance-transfer-v1"
    assert output["threshold_profile_version"] == "cli-test-v1"
    assert output["accepted"] is True
    assert "runtime_ns" in output
