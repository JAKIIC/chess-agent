from dataclasses import replace

import pytest

from xiangqi_agent.domain.board import Move
from xiangqi_agent.sync.evidence import SequenceCandidateEvidence
from xiangqi_agent.sync.sequence_gate import (
    SequenceDecisionGate,
    SequenceThresholdProfile,
)

PROFILE = SequenceThresholdProfile(
    min_local_difference=5.0,
    max_unexpected_difference=3.0,
    min_score=5.0,
    min_margin=5.0,
    max_template_distance=0.18,
    min_template_margin=0.02,
    min_template_confidence=0.8,
    profile_version="human-ai-two-ply-v1",
)
V2_PROFILE = replace(PROFILE, profile_version="human-ai-two-ply-v2")
FIRST = Move("a0a1", 81, 72)
SECOND = Move("a9a8", 0, 9)
OTHER_FIRST = Move("b0c2", 82, 65)
OTHER_SECOND = Move("b9c7", 1, 20)
GOOD = SequenceCandidateEvidence(
    moves=(FIRST, SECOND),
    changed_points=(0, 9, 72, 81),
    expected_change_floor=20.0,
    unexpected_difference=1.0,
    maximum_template_distance=0.05,
    minimum_template_margin=0.1,
    minimum_template_confidence=0.9,
    score=20.0,
    final_position_id="1" * 32,
)
DISTANT_RUNNER_UP = SequenceCandidateEvidence(
    moves=(OTHER_FIRST, OTHER_SECOND),
    changed_points=(1, 20, 65, 82),
    expected_change_floor=10.0,
    unexpected_difference=1.0,
    maximum_template_distance=0.05,
    minimum_template_margin=0.1,
    minimum_template_confidence=0.9,
    score=10.0,
    final_position_id="2" * 32,
)


def test_gate_accepts_only_the_unique_candidate_passing_every_hard_threshold() -> None:
    decision = SequenceDecisionGate(PROFILE).evaluate((GOOD, DISTANT_RUNNER_UP))

    assert decision.accepted
    assert decision.candidate == GOOD
    assert decision.rejection_reasons == ()


def test_gate_rejects_equal_scoring_candidates_without_exposing_a_candidate() -> None:
    equal_runner = replace(DISTANT_RUNNER_UP, score=GOOD.score)

    decision = SequenceDecisionGate(PROFILE).evaluate((GOOD, equal_runner))

    assert not decision.accepted
    assert decision.candidate is None
    assert decision.rejection_reasons == ("candidate_margin",)


@pytest.mark.parametrize(
    ("candidate", "reason"),
    (
        (replace(GOOD, expected_change_floor=4.0), "expected_change"),
        (replace(GOOD, unexpected_difference=4.0), "outside_change"),
        (replace(GOOD, score=4.0), "candidate_score"),
        (replace(GOOD, maximum_template_distance=0.19), "template_distance"),
        (replace(GOOD, minimum_template_margin=0.01), "template_margin"),
        (replace(GOOD, minimum_template_confidence=0.79), "template_confidence"),
    ),
)
def test_gate_rejects_each_failed_hard_threshold(
    candidate: SequenceCandidateEvidence,
    reason: str,
) -> None:
    decision = SequenceDecisionGate(PROFILE).evaluate((candidate,))

    assert not decision.accepted
    assert decision.candidate is None
    assert reason in decision.rejection_reasons


def test_v2_gate_tolerates_only_bounded_semantically_verified_artifacts() -> None:
    highlighted = replace(
        GOOD,
        unexpected_difference=6.0,
        minimum_template_margin=-0.01,
        score=14.0,
    )

    runner_up = replace(DISTANT_RUNNER_UP, score=8.0)

    decision = SequenceDecisionGate(V2_PROFILE).evaluate((highlighted, runner_up))

    assert decision.accepted
    assert decision.candidate == highlighted


def test_v2_gate_still_rejects_an_artifact_above_the_relative_cap() -> None:
    too_strong = replace(
        GOOD,
        unexpected_difference=6.01,
        minimum_template_margin=-0.01,
        score=13.99,
    )

    runner_up = replace(DISTANT_RUNNER_UP, score=8.0)

    decision = SequenceDecisionGate(V2_PROFILE).evaluate((too_strong, runner_up))

    assert not decision.accepted
    assert "outside_change" in decision.rejection_reasons


def test_v2_gate_keeps_semantic_confidence_as_a_hard_threshold() -> None:
    semantic_change = replace(
        GOOD,
        unexpected_difference=6.0,
        minimum_template_margin=-0.01,
        minimum_template_confidence=0.79,
        score=14.0,
    )

    runner_up = replace(DISTANT_RUNNER_UP, score=8.0)

    decision = SequenceDecisionGate(V2_PROFILE).evaluate((semantic_change, runner_up))

    assert not decision.accepted
    assert "template_confidence" in decision.rejection_reasons


def test_gate_distinguishes_no_legal_candidates_from_unavailable_templates() -> None:
    gate = SequenceDecisionGate(PROFILE)

    no_candidates = gate.evaluate(())
    unavailable = gate.evaluate((), template_unavailable=True)

    assert no_candidates.rejection_reasons == ("no_legal_candidates",)
    assert unavailable.rejection_reasons == ("template_unavailable",)


def test_profile_rejects_an_invalid_probability_like_evidence_threshold() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        replace(PROFILE, min_template_confidence=1.1)
