from __future__ import annotations

from dataclasses import replace

from xiangqi_agent.diagnostics.stage_c_gate import (
    DEFAULT_STAGE_C_FEATURE_VERSION,
    evaluate_stage_c_results,
)
from xiangqi_agent.diagnostics.stage_c_replay import HumanAiStageCReplayResult
from xiangqi_agent.diagnostics.stage_c_review import StageCReviewOutcome
from xiangqi_agent.diagnostics.stage_c_samples import StageCExpectedOutcome, StageCScenario

REJECTION_SCENARIOS = (
    StageCScenario.MULTIPLE_CANDIDATES,
    StageCScenario.SELECTION_HIGHLIGHT,
    StageCScenario.CONTINUOUS_ANIMATION,
    StageCScenario.OCCLUSION,
    StageCScenario.RESIZE,
    StageCScenario.THREE_PLY,
)


def _valid_result(
    index: int,
    *,
    accepted: bool = True,
    session_id: str | None = None,
    latency_ms: float = 100.0,
    review_outcome: StageCReviewOutcome | None = None,
) -> HumanAiStageCReplayResult:
    moves = ("h2e2", "h7e7") if accepted else ()
    return HumanAiStageCReplayResult(
        sample_id=f"valid-{index:03d}",
        session_id=session_id or f"valid-session-{index:03d}",
        scenario=StageCScenario.VALID_TWO_PLY,
        expected_outcome=StageCExpectedOutcome.ACCEPT,
        ground_truth_moves_uci=("h2e2", "h7e7"),
        accepted=accepted,
        replayed_moves_uci=moves,
        replayed_final_position_id="d4c03fbf35e2cedd0dff78f5897df228" if accepted else None,
        rejection_reasons=() if accepted else ("candidate_margin",),
        correct_accept=accepted,
        false_accept=False,
        correct_reject=False,
        missed_valid=not accepted,
        recorded_observation_matches_replay=True,
        decision_latency_ms=latency_ms,
        feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
        threshold_profile_version="human-ai-two-ply-v1",
        runtime_ns=2_000_000,
        review_outcome=review_outcome,
    )


def _rejection_result(
    index: int,
    *,
    scenario: StageCScenario | None = None,
    latency_ms: float = 100.0,
    review_outcome: StageCReviewOutcome | None = None,
) -> HumanAiStageCReplayResult:
    chosen = scenario or REJECTION_SCENARIOS[index % len(REJECTION_SCENARIOS)]
    return HumanAiStageCReplayResult(
        sample_id=f"reject-{index:03d}",
        session_id=f"reject-session-{index:03d}",
        scenario=chosen,
        expected_outcome=StageCExpectedOutcome.REJECT,
        ground_truth_moves_uci=(),
        accepted=False,
        replayed_moves_uci=(),
        replayed_final_position_id=None,
        rejection_reasons=("candidate_margin",),
        correct_accept=False,
        false_accept=False,
        correct_reject=True,
        missed_valid=False,
        recorded_observation_matches_replay=True,
        decision_latency_ms=latency_ms,
        feature_version=DEFAULT_STAGE_C_FEATURE_VERSION,
        threshold_profile_version="human-ai-two-ply-v1",
        runtime_ns=3_000_000,
        review_outcome=review_outcome,
    )


def _report(
    *,
    valid: tuple[HumanAiStageCReplayResult, ...] | None = None,
    rejected: tuple[HumanAiStageCReplayResult, ...] | None = None,
):
    return evaluate_stage_c_results(
        valid or tuple(_valid_result(index) for index in range(30)),
        rejected or tuple(_rejection_result(index) for index in range(30)),
    )


def test_release_gate_targets_the_current_two_ply_feature_version() -> None:
    assert DEFAULT_STAGE_C_FEATURE_VERSION == "two-ply-template-transfer-v5"


def test_29_valid_events_fail_the_count_gate() -> None:
    report = _report(valid=tuple(_valid_result(index) for index in range(29)))

    assert not report.release_pass
    assert "minimum_valid_events" in report.reasons


def test_30_valid_events_must_come_from_30_distinct_sessions() -> None:
    report = _report(
        valid=tuple(
            _valid_result(index, session_id="one-repeated-session")
            for index in range(30)
        )
    )

    assert not report.release_pass
    assert "minimum_distinct_valid_sessions" in report.reasons


def test_29_rejection_events_fail_the_count_gate() -> None:
    report = _report(rejected=tuple(_rejection_result(index) for index in range(29)))

    assert not report.release_pass
    assert "minimum_rejection_events" in report.reasons


def test_every_required_rejection_scenario_must_be_present() -> None:
    rejected = tuple(
        _rejection_result(index, scenario=StageCScenario.MULTIPLE_CANDIDATES)
        for index in range(30)
    )

    report = _report(rejected=rejected)

    assert not report.release_pass
    assert "missing_rejection_scenario:selection_highlight" in report.reasons
    assert "missing_rejection_scenario:three_ply" in report.reasons


def test_one_wrong_sequence_accept_fails_the_zero_false_accept_gate() -> None:
    valid = [_valid_result(index) for index in range(30)]
    valid[0] = replace(
        valid[0],
        replayed_moves_uci=("b2b3", "b7b6"),
        replayed_final_position_id="82dab2674504ae6edc4a430afef9277e",
        correct_accept=False,
        false_accept=True,
    )

    report = _report(valid=tuple(valid))

    assert report.metrics.false_accepts == 1
    assert not report.release_pass
    assert "zero_false_accepts" in report.reasons


def test_one_accepted_rejection_fails_the_zero_false_accept_gate() -> None:
    rejected = [_rejection_result(index) for index in range(30)]
    rejected[0] = replace(
        rejected[0],
        accepted=True,
        replayed_moves_uci=("h2e2", "h7e7"),
        replayed_final_position_id="d4c03fbf35e2cedd0dff78f5897df228",
        rejection_reasons=(),
        false_accept=True,
        correct_reject=False,
    )

    report = _report(rejected=tuple(rejected))

    assert report.metrics.false_accepts == 1
    assert not report.release_pass


def test_24_of_30_valid_accepts_passes_the_80_percent_coverage_boundary() -> None:
    valid = tuple(
        _valid_result(index, accepted=index < 24)
        for index in range(30)
    )

    report = _report(valid=valid)

    assert report.metrics.coverage == 0.8
    assert report.release_pass


def test_23_of_30_valid_accepts_fails_the_coverage_gate() -> None:
    valid = tuple(
        _valid_result(index, accepted=index < 23)
        for index in range(30)
    )

    report = _report(valid=valid)

    assert report.metrics.coverage == 23 / 30
    assert not report.release_pass
    assert "minimum_valid_coverage" in report.reasons


def test_500_ms_p95_passes_and_any_greater_boundary_fails() -> None:
    at_boundary = _report(
        valid=tuple(_valid_result(index, latency_ms=500.0) for index in range(30)),
        rejected=tuple(_rejection_result(index, latency_ms=500.0) for index in range(30)),
    )
    above_boundary = _report(
        valid=tuple(_valid_result(index, latency_ms=500.1) for index in range(30)),
        rejected=tuple(_rejection_result(index, latency_ms=500.1) for index in range(30)),
    )

    assert at_boundary.metrics.p95_decision_latency_ms == 500.0
    assert at_boundary.release_pass
    assert above_boundary.metrics.p95_decision_latency_ms == 500.1
    assert not above_boundary.release_pass
    assert "maximum_decision_latency_p95" in above_boundary.reasons


def test_release_pass_requires_replay_to_match_recorded_observation() -> None:
    valid = [_valid_result(index) for index in range(30)]
    valid[0] = replace(valid[0], recorded_observation_matches_replay=False)

    report = _report(valid=tuple(valid))

    assert report.metrics.recorded_consistency_failures == 1
    assert not report.release_pass
    assert "recorded_replay_consistency" in report.reasons


def test_complete_safe_dataset_passes_all_hard_gates() -> None:
    report = _report()

    assert report.release_pass
    assert report.reasons == ()
    assert report.metrics.valid_samples == 30
    assert report.metrics.rejection_samples == 30
    assert report.metrics.distinct_valid_sessions == 30
    assert report.metrics.correct_accepts == 30
    assert report.metrics.false_accepts == 0
    assert report.metrics.accepted_precision == 1.0
    assert report.metrics.final_position_accuracy == 1.0
    assert report.metrics.p95_replay_runtime_ms == 3.0


def test_review_outcome_metrics_count_only_reviewed_v2_provenance() -> None:
    valid = tuple(
        _valid_result(
            index,
            review_outcome=(
                StageCReviewOutcome.CANDIDATE_CONFIRMED
                if index < 10
                else StageCReviewOutcome.LEGAL_MOVE_CORRECTION
            ),
        )
        for index in range(20)
    ) + tuple(_valid_result(index + 20) for index in range(10))
    rejected = tuple(
        _rejection_result(
            index,
            review_outcome=StageCReviewOutcome.EXPECTED_REJECTION,
        )
        for index in range(15)
    ) + tuple(_rejection_result(index + 15) for index in range(15))

    report = _report(valid=valid, rejected=rejected)

    assert dict(report.metrics.review_outcome_counts) == {
        "candidate_confirmed": 10,
        "expected_rejection": 15,
        "legal_move_correction": 10,
    }
    assert report.metrics.total_samples == 60
