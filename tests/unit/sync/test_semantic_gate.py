from __future__ import annotations

from dataclasses import replace

import pytest

from xiangqi_agent.domain.board import Move
from xiangqi_agent.sync.evidence import CandidateEvidence
from xiangqi_agent.sync.semantic_gate import (
    MoveSemanticGate,
    SemanticThresholds,
)
from xiangqi_agent.vision.endpoint_features import EndpointFeatures


def _candidate() -> CandidateEvidence:
    return CandidateEvidence(
        move=Move("i0h0", 89, 88),
        source_difference=14.0,
        destination_difference=27.0,
        unexpected_difference=1.0,
        source_expected_distance=0.08,
        destination_expected_distance=0.09,
        semantic_margin=0.04,
        source_semantic_evidence_score=0.9,
        destination_semantic_evidence_score=0.9,
        score=13.0,
    )


def _features() -> EndpointFeatures:
    return EndpointFeatures(
        feature_version="instance-transfer-v1",
        instance_distance=0.1,
        instance_evidence_score=0.75,
        color_distance=0.1,
        gradient_distance=0.1,
        source_change_distance=0.3,
        target_change_distance=0.3,
        best_shift=(0, 0),
    )


def _gate() -> MoveSemanticGate:
    return MoveSemanticGate(
        SemanticThresholds(
            max_source_empty_distance=0.18,
            min_source_empty_evidence_score=0.8,
            max_destination_side_distance=0.18,
            min_destination_side_evidence_score=0.8,
            max_instance_distance=0.2,
            min_instance_evidence_score=0.6,
            min_semantic_margin=0.02,
        )
    )


@pytest.mark.parametrize(
    ("expected_reason", "candidate", "features"),
    [
        (
            "source_empty",
            replace(_candidate(), source_semantic_evidence_score=0.2),
            _features(),
        ),
        (
            "side",
            replace(_candidate(), destination_expected_distance=0.3),
            _features(),
        ),
        (
            "semantic_margin",
            replace(_candidate(), semantic_margin=0.0),
            _features(),
        ),
        (
            "instance",
            _candidate(),
            replace(_features(), instance_distance=0.4),
        ),
    ],
)
def test_any_failed_semantic_gate_rejects_without_weighted_compensation(
    expected_reason: str,
    candidate: CandidateEvidence,
    features: EndpointFeatures,
) -> None:
    result = _gate().evaluate(candidate, features)

    assert not result.accepted
    assert result.rejection_reasons == (expected_reason,)


def test_all_semantic_hard_gates_must_pass_together() -> None:
    result = _gate().evaluate(_candidate(), _features())

    assert result.accepted
    assert result.rejection_reasons == ()
