from __future__ import annotations

import json

import httpx

from xiangqi_agent.coach.client import DeepSeekClient
from xiangqi_agent.coach.evidence import build_evidence
from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.coach import CoachEvidence
from xiangqi_agent.domain.fen import parse_fen

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _evidence() -> CoachEvidence:
    board = parse_fen(START)
    lines = tuple(
        EngineLine(
            position_id=board.position_id,
            depth=18,
            seldepth=24,
            multipv=index,
            score_cp=score,
            mate_in=None,
            nodes=10_000,
            nps=100_000,
            time_ms=500,
            pv=(uci,),
        )
        for index, (uci, score) in enumerate(
            (("h2e2", 35), ("b0c2", 25), ("g3g4", 15)), start=1
        )
    )
    analysis = EngineAnalysis(
        position_id=board.position_id,
        duration_ms=500,
        depth=18,
        nodes=10_000,
        lines=lines,
        bestmove="h2e2",
        engine_name="test",
    )
    return build_evidence(board, analysis, user_side="w")


def _content(**overrides: object) -> str:
    payload: dict[str, object] = {
        "position_summary": "双方子力完整，当前处于开局。",
        "main_plan": "优先争夺中路并协调子力。",
        "candidate_id": "candidate_1",
        "why": "第一候选能立即建立中心压力。",
        "opponent_threat": "需要留意黑方发展子力后的反击。",
        "alternatives": ["candidate_2"],
        "training_question": "你能找出第一候选控制了哪些关键线路吗？",
        "confidence": 0.88,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_request_uses_official_json_mode_and_returns_validated_explanation() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _content()}}]},
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = DeepSeekClient(api_key="test-secret", http_client=http, timeout_seconds=1)

    result = client.explain(_evidence(), "为什么推荐第一步？")
    body = json.loads(requests[0].content)

    assert result.source == "deepseek"
    assert result.candidate_id == "candidate_1"
    assert result.position_id == _evidence().position_id
    assert requests[0].url == "https://api.deepseek.com/chat/completions"
    assert requests[0].headers["authorization"] == "Bearer test-secret"
    assert body["model"] == "deepseek-v4-flash"
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert "screenshot" not in requests[0].content.decode().lower()


def test_invalid_candidate_retries_once_then_uses_local_fallback() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _content(candidate_id="candidate_99")}}]},
        )

    client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.explain(_evidence(), "下一步呢？")

    assert calls == 2
    assert result.source == "local_fallback"
    assert result.candidate_id == "candidate_1"


def test_unlisted_concrete_move_in_text_is_rejected() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": _content(why="建议直接走车一进一获得优势。")}}
                ]
            },
        )

    client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.explain(_evidence(), "为什么？")

    assert result.source == "local_fallback"


def test_http_payment_error_falls_back_without_exposing_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, text="balance unavailable")

    client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.explain(_evidence(), "解释一下")

    assert result.source == "local_fallback"


def test_timeout_and_repeated_empty_content_both_fall_back() -> None:
    evidence = _evidence()

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    timeout_client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(timeout_handler)),
    )
    assert timeout_client.explain(evidence, "解释").source == "local_fallback"

    empty_calls = 0

    def empty_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal empty_calls
        empty_calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    empty_client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(empty_handler)),
    )
    assert empty_client.explain(evidence, "解释").source == "local_fallback"
    assert empty_calls == 2


def test_deep_mode_uses_pro_with_thinking_enabled() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": _content()}}]})

    client = DeepSeekClient(
        api_key="test-secret",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.explain(_evidence(), "深度复盘", deep=True)

    assert result.source == "deepseek"
    assert bodies[0]["model"] == "deepseek-v4-pro"
    assert bodies[0]["thinking"] == {"type": "enabled", "reasoning_effort": "high"}
