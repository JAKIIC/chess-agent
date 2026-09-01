from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.diagnostics.test_stage_c_quarantine import (
    START,
    _candidate,
    _crops,
    _event,
    _occupancy,
    _two_ply_final,
)
from xiangqi_agent.diagnostics.stage_c_promotion import (
    PromotionStatus,
    StageCPromotionBlockedError,
    StageCPromotionService,
    StageCPromotionVerifier,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    QuarantinedStageCEventV1,
    QuarantineEventRecorder,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewOutcome,
    StageCReviewService,
    StageCReviewStore,
)
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import (
    ReviewedStageCSampleLoader,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.rules import apply_move, legal_moves


def test_valid_review_is_promotable_and_written_as_self_contained_v2(
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    event_before = {path.name: path.read_bytes() for path in event_dir.iterdir()}
    review_before = review_path.read_bytes()

    verifier = StageCPromotionVerifier()
    decision = verifier.verify(event_dir, review_path)
    assert decision.status is PromotionStatus.PROMOTABLE
    assert decision.projected_final_position_id == _two_ply_final(START).position_id

    output = StageCPromotionService(
        verifier,
        now_utc=lambda: datetime(2026, 9, 1, 1, tzinfo=UTC),
    ).promote(event_dir, review_path, _reviewed_root(tmp_path))
    loaded = ReviewedStageCSampleLoader().load(output)

    assert loaded.metadata.sample_id == "event-1"
    assert loaded.metadata.ground_truth_moves_uci == ("h2e2", "h7e7")
    assert loaded.metadata.review_outcome is StageCReviewOutcome.CANDIDATE_CONFIRMED
    assert loaded.source_event_manifest_bytes == event_before["manifest.json"]
    assert loaded.review_manifest_bytes == review_before
    assert {path.name: path.read_bytes() for path in event_dir.iterdir()} == event_before
    assert review_path.read_bytes() == review_before


def test_valid_user_correction_can_promote_when_observer_rejected_and_omitted_truth(
    tmp_path: Path,
) -> None:
    final = _two_ply_final(START)
    event = replace(
        _event(),
        observed_status=StageCObservedStatus.REJECTED,
        observed_moves_uci=(),
        observed_final_position_id=START.position_id,
        candidates=(),
        rejection_reasons=("candidate_score",),
        after_occupancy=_occupancy(final),
    )
    event_dir = _record(tmp_path, event)
    review_path = _review_valid(
        tmp_path,
        event_dir,
        outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
    )

    decision = StageCPromotionVerifier().verify(event_dir, review_path)
    assert decision.status is PromotionStatus.PROMOTABLE


def test_low_confidence_rule_changed_point_needs_review(tmp_path: Path) -> None:
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

    decision = StageCPromotionVerifier().verify(event_dir, review_path)
    assert decision.status is PromotionStatus.NEEDS_REVIEW
    assert decision.reason_codes == ("changed_point_confidence",)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("baseline_confidence", "baseline_confidence"),
        ("baseline_occupancy", "baseline_occupancy_mismatch"),
        ("final_occupancy", "final_occupancy_mismatch"),
        ("local_difference", "missing_local_change"),
    ),
)
def test_occupancy_and_local_evidence_fail_closed(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    event = _event()
    if mutation == "baseline_confidence":
        values = list(event.before_occupancy.confidences)
        values[0] = 0.64
        event = replace(
            event,
            before_occupancy=replace(event.before_occupancy, confidences=tuple(values)),
        )
    elif mutation == "baseline_occupancy":
        values = list(event.before_occupancy.occupied)
        values[0] = not values[0]
        event = replace(
            event,
            before_occupancy=replace(event.before_occupancy, occupied=tuple(values)),
        )
    elif mutation == "final_occupancy":
        values = list(event.after_occupancy.occupied)
        values[22] = not values[22]
        event = replace(
            event,
            after_occupancy=replace(event.after_occupancy, occupied=tuple(values)),
        )
    else:
        values = list(event.local_differences)
        values[22] = 0.0
        event = replace(event, local_differences=tuple(values))
    event_dir = _record(tmp_path, event)
    review_path = _review_valid(tmp_path, event_dir)

    decision = StageCPromotionVerifier().verify(event_dir, review_path)
    expected = (
        PromotionStatus.NEEDS_REVIEW
        if mutation == "baseline_confidence"
        else PromotionStatus.REJECTED
    )
    assert decision.status is expected
    assert decision.reason_codes == (reason,)


def test_tampered_crop_or_inactive_review_is_rejected(tmp_path: Path) -> None:
    event_dir = _record(tmp_path, _event())
    first = _review_valid(tmp_path, event_dir)
    second = _review_valid(
        tmp_path,
        event_dir,
        supersedes=first.stem,
        outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
        moves=("b2b3", "b7b6"),
    )
    decision = StageCPromotionVerifier().verify(event_dir, first)
    assert decision.status is PromotionStatus.REJECTED
    assert decision.reason_codes == ("inactive_review",)

    (event_dir / "point-22-before.png").write_bytes(b"tampered")
    decision = StageCPromotionVerifier().verify(event_dir, second)
    assert decision.status is PromotionStatus.REJECTED
    assert decision.reason_codes == ("event_integrity",)


@pytest.mark.parametrize(
    "scenario",
    (
        StageCScenario.MULTIPLE_CANDIDATES,
        StageCScenario.SELECTION_HIGHLIGHT,
        StageCScenario.CONTINUOUS_ANIMATION,
        StageCScenario.OCCLUSION,
        StageCScenario.RESIZE,
        StageCScenario.THREE_PLY,
    ),
)
def test_all_six_frozen_rejection_scenarios_can_be_verified(
    tmp_path: Path,
    scenario: StageCScenario,
) -> None:
    event, moves = _rejection_event(scenario)
    event_dir = _record(tmp_path, event)
    review_path = _review_rejection(tmp_path, event_dir, scenario, moves)

    decision = StageCPromotionVerifier().verify(event_dir, review_path)
    assert decision.status is PromotionStatus.PROMOTABLE


@pytest.mark.parametrize("scenario", (StageCScenario.OCCLUSION, StageCScenario.RESIZE))
def test_ordinary_accepted_event_cannot_be_relabelled_as_machine_failure(
    tmp_path: Path,
    scenario: StageCScenario,
) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_rejection(tmp_path, event_dir, scenario, ())

    decision = StageCPromotionVerifier().verify(event_dir, review_path)
    assert decision.status is PromotionStatus.REJECTED
    assert decision.reason_codes == ("scenario_evidence_mismatch",)


def test_three_ply_and_multiple_candidate_evidence_are_required(
    tmp_path: Path,
) -> None:
    event, _moves = _rejection_event(StageCScenario.THREE_PLY)
    event_dir = _record(tmp_path / "three", event)
    review_path = _review_rejection(
        tmp_path / "three",
        event_dir,
        StageCScenario.THREE_PLY,
        ("h2e2", "h7e7"),
    )
    assert StageCPromotionVerifier().verify(event_dir, review_path).reason_codes == (
        "three_ply_count",
    )

    plain, _ = _rejection_event(StageCScenario.SELECTION_HIGHLIGHT)
    event_dir = _record(tmp_path / "multiple", plain)
    review_path = _review_rejection(
        tmp_path / "multiple",
        event_dir,
        StageCScenario.MULTIPLE_CANDIDATES,
        (),
    )
    assert StageCPromotionVerifier().verify(event_dir, review_path).reason_codes == (
        "multiple_candidate_evidence",
    )


def test_duplicate_candidate_records_do_not_prove_ambiguity(tmp_path: Path) -> None:
    event, _ = _rejection_event(StageCScenario.SELECTION_HIGHLIGHT)
    duplicate = _candidate()
    event = replace(
        event,
        changed_points=(22, 25, 67, 70),
        candidates=(duplicate, duplicate),
    )
    event_dir = _record(tmp_path, event)
    review_path = _review_rejection(
        tmp_path,
        event_dir,
        StageCScenario.MULTIPLE_CANDIDATES,
        (),
    )

    assert StageCPromotionVerifier().verify(event_dir, review_path).reason_codes == (
        "multiple_candidate_evidence",
    )


def test_promotion_refuses_existing_target_overlapping_roots_and_source_mutation(
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    reviewed = _reviewed_root(tmp_path)
    target = reviewed / "session-1" / "event-1"
    target.mkdir(parents=True)
    service = StageCPromotionService(StageCPromotionVerifier())
    with pytest.raises(FileExistsError):
        service.promote(event_dir, review_path, reviewed)

    with pytest.raises(ValueError, match="overlap"):
        service.promote(event_dir, review_path, _quarantine_root(tmp_path))

    target.rmdir()

    def mutate() -> None:
        (event_dir / "manifest.json").write_bytes((event_dir / "manifest.json").read_bytes() + b" ")

    service = StageCPromotionService(
        StageCPromotionVerifier(),
        before_copy=mutate,
    )
    with pytest.raises(StageCPromotionBlockedError, match="source_mutated"):
        service.promote(event_dir, review_path, reviewed)
    assert not target.exists()


def _record(tmp_path: Path, event: QuarantinedStageCEventV1) -> Path:
    return QuarantineEventRecorder(_quarantine_root(tmp_path), enabled=True).record(
        event,
        _crops(event.changed_points),
    )


def _review_valid(
    tmp_path: Path,
    event_dir: Path,
    *,
    outcome: StageCReviewOutcome = StageCReviewOutcome.CANDIDATE_CONFIRMED,
    moves: tuple[str, str] = ("h2e2", "h7e7"),
    supersedes: str | None = None,
) -> Path:
    return _review_service(tmp_path).submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.VALID_TWO_PLY,
            moves_uci=moves,
            scenario=None,
            review_outcome=outcome,
            supersedes_review_id=supersedes,
        ),
    )


def _review_rejection(
    tmp_path: Path,
    event_dir: Path,
    scenario: StageCScenario,
    moves: tuple[str, ...],
) -> Path:
    return _review_service(tmp_path).submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.EXPECTED_REJECTION,
            moves_uci=moves,
            scenario=scenario,
            review_outcome=StageCReviewOutcome.EXPECTED_REJECTION,
        ),
    )


def _review_service(tmp_path: Path) -> StageCReviewService:
    counter = len(tuple(_review_root(tmp_path).rglob("*.json"))) + 1
    return StageCReviewService(
        StageCReviewStore(_review_root(tmp_path), enabled=True),
        now_utc=lambda: datetime(2026, 9, 1, counter, tzinfo=UTC),
        review_id_factory=lambda: f"review-{counter}",
    )


def _rejection_event(
    scenario: StageCScenario,
) -> tuple[QuarantinedStageCEventV1, tuple[str, ...]]:
    if scenario is StageCScenario.THREE_PLY:
        moves = ("h2e2", "h7e7", "e2h2")
        final = _project(START, moves)
        return (
            replace(
                _event(),
                observed_status=StageCObservedStatus.REJECTED,
                observed_moves_uci=(),
                observed_final_position_id=START.position_id,
                changed_points=(22, 25),
                candidates=(),
                rejection_reasons=("candidate_margin",),
                after_occupancy=_occupancy(final),
            ),
            moves,
        )
    if scenario is StageCScenario.MULTIPLE_CANDIDATES:
        other_final = _project(START, ("b2b3", "b7b6"))
        other = StageCCandidateRecord(
            moves_uci=("b2b3", "b7b6"),
            changed_points=(19, 28, 55, 64),
            expected_change_floor=18.0,
            unexpected_difference=1.5,
            maximum_template_distance=0.06,
            minimum_template_margin=0.09,
            minimum_template_confidence=0.88,
            score=10.0,
            final_position_id=other_final.position_id,
        )
        return (
            replace(
                _event(),
                observed_status=StageCObservedStatus.REJECTED,
                observed_moves_uci=(),
                observed_final_position_id=START.position_id,
                candidates=(_candidate(), other),
                rejection_reasons=("candidate_margin",),
                after_occupancy=_occupancy(START),
            ),
            (),
        )
    reason = "frame_size_changed" if scenario is StageCScenario.RESIZE else "template_unavailable"
    return (
        replace(
            _event(),
            observed_status=StageCObservedStatus.REJECTED,
            observed_moves_uci=(),
            observed_final_position_id=START.position_id,
            changed_points=(67,),
            candidates=(),
            rejection_reasons=(reason,),
            after_occupancy=_occupancy(START),
        ),
        (),
    )


def _project(board: BoardState, moves: tuple[str, ...]) -> BoardState:
    projected = board
    for uci in moves:
        move = next(move for move in legal_moves(projected) if move.uci == uci)
        projected = apply_move(projected, move)
    return projected


def _local_root(tmp_path: Path) -> Path:
    return tmp_path / ".local"


def _quarantine_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-quarantine"


def _review_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-reviews"


def _reviewed_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-reviewed"
