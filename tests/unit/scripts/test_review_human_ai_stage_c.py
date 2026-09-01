from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.review_human_ai_stage_c import _parse_args, main
from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    QuarantinedStageCEventV1,
    QuarantineEventRecorder,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCReviewStore,
    load_stage_c_review,
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


def test_cli_submits_candidate_confirmation_without_candidate_scores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_dir = _record_event(tmp_path)
    exit_code = main(
        _base_args(tmp_path, event_dir)
        + [
            "--label",
            "valid_two_ply",
            "--moves",
            "h2e2,h7e7",
            "--outcome",
            "candidate_confirmed",
        ],
        allowed_local_root=_local_root(tmp_path),
    )
    output = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(output.out)
    assert payload["status"] == "reviewed"
    assert payload["event_id"] == "event-1"
    assert "score" not in output.out.lower()
    assert "h2e2" not in output.out
    reviews = tuple(_review_root(tmp_path).rglob("*.json"))
    assert len(reviews) == 1
    assert load_stage_c_review(reviews[0]).review_id == payload["review_id"]


def test_cli_lists_only_chinese_and_uci_legal_choices_without_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_dir = _record_event(tmp_path)
    assert (
        main(
            _base_args(tmp_path, event_dir) + ["--list-legal"],
            allowed_local_root=_local_root(tmp_path),
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "legal_choices"
    assert payload["choices"]
    assert all(set(choice) == {"chinese", "uci"} for choice in payload["choices"])
    assert payload["choices"] == sorted(
        payload["choices"], key=lambda choice: (choice["chinese"], choice["uci"])
    )
    assert not _review_root(tmp_path).exists()


def test_cli_lists_second_ply_choices_after_one_legal_prefix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_dir = _record_event(tmp_path)
    assert (
        main(
            _base_args(tmp_path, event_dir)
            + ["--list-legal", "--moves", "h2e2"],
            allowed_local_root=_local_root(tmp_path),
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["prefix_length"] == 1
    assert any(choice["uci"] == "h7e7" for choice in payload["choices"])


def test_cli_legal_listing_cannot_silently_ignore_review_fields(
    tmp_path: Path,
) -> None:
    event_dir = _record_event(tmp_path)
    assert (
        main(
            _base_args(tmp_path, event_dir)
            + ["--list-legal", "--label", "discard"],
            allowed_local_root=_local_root(tmp_path),
        )
        == 2
    )
    assert not _review_root(tmp_path).exists()


@pytest.mark.parametrize(
    "arguments",
    (
        ["--label", "valid_two_ply", "--moves", "h2e2,h7e7"],
        ["--label", "expected_rejection", "--scenario", "valid_two_ply"],
        ["--label", "discard", "--moves", "h2e2"],
    ),
)
def test_cli_fails_closed_for_invalid_label_arguments(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    event_dir = _record_event(tmp_path)
    assert (
        main(
            _base_args(tmp_path, event_dir) + arguments,
            allowed_local_root=_local_root(tmp_path),
        )
        == 2
    )
    assert not _review_root(tmp_path).exists()


def test_cli_rejects_event_or_review_paths_outside_fixed_local_roots(
    tmp_path: Path,
) -> None:
    event_dir = _record_event(tmp_path)
    outside = tmp_path / "reviews"
    args = [
        "--event-dir",
        str(event_dir),
        "--review-root",
        str(outside),
        "--label",
        "discard",
    ]
    assert main(args, allowed_local_root=_local_root(tmp_path)) == 2
    assert not outside.exists()


def test_cli_review_chain_requires_explicit_superseded_review_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    event_dir = _record_event(tmp_path)
    first_args = _base_args(tmp_path, event_dir) + [
        "--label",
        "valid_two_ply",
        "--moves",
        "h2e2,h7e7",
        "--outcome",
        "candidate_confirmed",
    ]
    assert main(first_args, allowed_local_root=_local_root(tmp_path)) == 0
    first = json.loads(capsys.readouterr().out)

    correction = _base_args(tmp_path, event_dir) + [
        "--label",
        "valid_two_ply",
        "--moves",
        "b2b3,b7b6",
        "--outcome",
        "legal_move_correction",
        "--supersedes-review-id",
        first["review_id"],
    ]
    assert main(correction, allowed_local_root=_local_root(tmp_path)) == 0
    capsys.readouterr()
    store = StageCReviewStore(_review_root(tmp_path), enabled=True)
    active = store.active_review("session-1", "event-1")
    assert active is not None
    assert active.supersedes_review_id == first["review_id"]


def test_parse_rejects_empty_move_items() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            [
                "--event-dir",
                "event",
                "--review-root",
                "reviews",
                "--label",
                "valid_two_ply",
                "--moves",
                "h2e2,,h7e7",
                "--outcome",
                "candidate_confirmed",
            ]
        )


def _base_args(tmp_path: Path, event_dir: Path) -> list[str]:
    return [
        "--event-dir",
        str(event_dir),
        "--review-root",
        str(_review_root(tmp_path)),
    ]


def _record_event(tmp_path: Path) -> Path:
    final = _two_ply_final(START)
    event = QuarantinedStageCEventV1(
        event_id="event-1",
        session_id="session-1",
        created_at_utc="2026-09-01T00:00:00Z",
        confirmed_fen=START_FEN,
        confirmed_position_id=START.position_id,
        observed_status=StageCObservedStatus.ACCEPTED,
        observed_moves_uci=("h2e2", "h7e7"),
        observed_final_position_id=final.position_id,
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        changed_points=(22, 25, 67, 70),
        local_differences=tuple(float(index + 1) / 10 for index in range(90)),
        candidates=(
            StageCCandidateRecord(
                moves_uci=("h2e2", "h7e7"),
                changed_points=(22, 25, 67, 70),
                expected_change_floor=20.0,
                unexpected_difference=1.0,
                maximum_template_distance=0.05,
                minimum_template_margin=0.1,
                minimum_template_confidence=0.9,
                score=20.0,
                final_position_id=final.position_id,
            ),
        ),
        rejection_reasons=(),
        capture_context=CaptureContext(
            wgc_size=(216, 240),
            client_size=(216, 240),
            dpi_scale=1.0,
            geometry_revision="quad-v1",
            theme_fingerprint="theme-v1",
            generation_id=1,
        ),
        feature_version="two-ply-template-v1",
        threshold_profile_version="human-ai-two-ply-v1",
        decision_latency_ms=100.0,
        before_occupancy=_occupancy(START),
        after_occupancy=_occupancy(final),
    )
    crops = tuple(
        TransitionPointCrops(
            point,
            np.full((48, 48, 4), (20, 20, 20, 255), dtype=np.uint8),
            np.full((48, 48, 4), (60, 60, 60, 255), dtype=np.uint8),
        )
        for point in event.changed_points
    )
    return QuarantineEventRecorder(
        _local_root(tmp_path) / "stage-c-quarantine",
        enabled=True,
    ).record(event, crops)


def _two_ply_final(board: BoardState) -> BoardState:
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    return apply_move(middle, second)


def _occupancy(board: BoardState) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (0.95,) * 90,
        "circular-occupancy-v1",
    )


def _local_root(tmp_path: Path) -> Path:
    return tmp_path / ".local"


def _review_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-reviews"
