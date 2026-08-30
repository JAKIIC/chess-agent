from __future__ import annotations

from tests.integration.coach.test_deepseek_client import _evidence
from xiangqi_agent.coach.client import DeepSeekClient
from xiangqi_agent.coach.fallback import local_explanation


def test_no_api_key_never_calls_network_and_uses_deterministic_fallback() -> None:
    evidence = _evidence()
    client = DeepSeekClient(api_key=None)

    first = client.explain(evidence, "为什么？")
    second = client.explain(evidence, "为什么？")

    assert first == second
    assert first.source == "local_fallback"
    assert first.position_id == evidence.position_id
    assert first.candidate_id == "candidate_1"
    assert evidence.allowed_move_map["candidate_1"] in first.why


def test_local_fallback_only_references_allowed_candidates() -> None:
    evidence = _evidence()

    result = local_explanation(evidence, "有什么计划？")

    assert result.candidate_id in evidence.allowed_move_map
    assert set(result.alternatives).issubset(evidence.allowed_move_map)
