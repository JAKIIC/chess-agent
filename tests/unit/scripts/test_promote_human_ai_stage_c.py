from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.promote_human_ai_stage_c import main
from tests.unit.diagnostics.test_stage_c_promotion import (
    _local_root,
    _record,
    _review_service,
    _review_valid,
    _reviewed_root,
)
from tests.unit.diagnostics.test_stage_c_quarantine import _event
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewOutcome,
)


def test_cli_promotes_and_prints_only_anonymous_sample_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)

    assert (
        main(
            _args(tmp_path, event_dir, review_path),
            allowed_local_root=_local_root(tmp_path),
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.out.strip() == "event-1"
    assert "h2e2" not in output.out
    assert "stage-c-quarantine" not in output.out


def test_cli_returns_one_for_needs_review_and_writes_nothing(
    tmp_path: Path,
) -> None:
    event = _event()
    confidence = list(event.after_occupancy.confidences)
    confidence[22] = 0.64
    event = replace(
        event,
        after_occupancy=replace(
            event.after_occupancy,
            confidences=tuple(confidence),
        ),
    )
    event_dir = _record(tmp_path, event)
    review_path = _review_valid(tmp_path, event_dir)

    assert (
        main(
            _args(tmp_path, event_dir, review_path),
            allowed_local_root=_local_root(tmp_path),
        )
        == 1
    )
    assert not _reviewed_root(tmp_path).exists()


def test_cli_returns_two_for_integrity_rejection(
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    (event_dir / "point-22-before.png").write_bytes(b"changed")
    assert (
        main(
            _args(tmp_path, event_dir, review_path),
            allowed_local_root=_local_root(tmp_path),
        )
        == 2
    )


def test_cli_never_promotes_a_discard_review(tmp_path: Path) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_service(tmp_path).submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.DISCARD,
            moves_uci=(),
            scenario=None,
            review_outcome=StageCReviewOutcome.DISCARDED,
        ),
    )

    assert (
        main(
            _args(tmp_path, event_dir, review_path),
            allowed_local_root=_local_root(tmp_path),
        )
        == 2
    )
    assert not _reviewed_root(tmp_path).exists()


def test_cli_rejects_paths_outside_fixed_local_roots(tmp_path: Path) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    args = [
        "--event-dir",
        str(event_dir),
        "--review-path",
        str(review_path),
        "--reviewed-root",
        str(tmp_path / "outside"),
    ]
    assert main(args, allowed_local_root=_local_root(tmp_path)) == 2
    assert not (tmp_path / "outside").exists()


def _args(tmp_path: Path, event_dir: Path, review_path: Path) -> list[str]:
    return [
        "--event-dir",
        str(event_dir),
        "--review-path",
        str(review_path),
        "--reviewed-root",
        str(_reviewed_root(tmp_path)),
    ]
