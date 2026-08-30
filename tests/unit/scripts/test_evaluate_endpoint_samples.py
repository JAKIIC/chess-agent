from __future__ import annotations

from dataclasses import replace

from scripts.evaluate_endpoint_samples import evaluate_results
from xiangqi_agent.diagnostics.endpoint_replay import EndpointReplayResult
from xiangqi_agent.vision.endpoint_features import EndpointFeatures


def _result(
    *,
    actual_uci: str | None,
    probe_uci: str = "h2e2",
    accepted: bool,
    runtime_ns: int,
) -> EndpointReplayResult:
    return EndpointReplayResult(
        sample_id=f"sample-{runtime_ns}",
        actual_uci=actual_uci,
        probe_uci=probe_uci,
        feature_version="instance-transfer-v1",
        threshold_profile_version="test-v1",
        accepted=accepted,
        rejection_reasons=() if accepted else ("instance",),
        features=EndpointFeatures(
            feature_version="instance-transfer-v1",
            instance_distance=0.1,
            instance_evidence_score=0.8,
            color_distance=0.1,
            gradient_distance=0.1,
            source_change_distance=0.3,
            target_change_distance=0.3,
            best_shift=(0, 0),
        ),
        result_fen="changed" if accepted else None,
        runtime_ns=runtime_ns,
    )


def test_evaluator_reports_top1_precision_coverage_false_accepts_and_p95() -> None:
    correct_accepted = _result(actual_uci="h2e2", accepted=True, runtime_ns=10)
    correct_rejected = _result(actual_uci="h2e2", accepted=False, runtime_ns=20)
    wrong_top1 = _result(actual_uci="b2b3", accepted=False, runtime_ns=30)
    negative_rejected = _result(actual_uci=None, accepted=False, runtime_ns=40)
    negative_accepted = _result(actual_uci=None, accepted=True, runtime_ns=50)

    report = evaluate_results(
        (
            correct_accepted,
            correct_rejected,
            wrong_top1,
            negative_rejected,
            negative_accepted,
        )
    )

    assert report.positive_samples == 3
    assert report.negative_samples == 2
    assert report.top1_correct == 2
    assert report.top1_accuracy == 2 / 3
    assert report.correct_accepts == 1
    assert report.false_accepts == 1
    assert report.accepted_precision == 0.5
    assert report.coverage == 1 / 3
    assert report.p95_runtime_ns == 50


def test_evaluator_counts_an_accepted_wrong_top1_as_a_false_accept() -> None:
    wrong = replace(_result(actual_uci="b2b3", accepted=False, runtime_ns=1), accepted=True)

    report = evaluate_results((wrong,))

    assert report.false_accepts == 1
    assert report.correct_accepts == 0
