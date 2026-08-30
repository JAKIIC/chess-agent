from __future__ import annotations

import json
import re
from typing import Any

import httpx
from pydantic import ValidationError

from xiangqi_agent.coach.fallback import local_explanation
from xiangqi_agent.coach.prompts import SYSTEM_PROMPT
from xiangqi_agent.domain.coach import CoachEvidence, CoachExplanation

_CANDIDATE_REFERENCE = re.compile(r"candidate_[0-9]+")
_UCI_REFERENCE = re.compile(r"(?<![a-z0-9])[a-i][0-9][a-i][0-9](?![a-z0-9])")
_CHINESE_MOVE = re.compile(
    r"(?:前|后|中)?[车马炮兵卒帅将仕士相象][一二三四五六七八九123456789]?"
    r"[进退平][一二三四五六七八九123456789]"
)


class CoachResponseError(ValueError):
    """DeepSeek returned content outside the grounded coaching contract."""


class DeepSeekClient:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "deepseek-v4-flash",
        deep_model: str = "deepseek-v4-pro",
        timeout_seconds: float = 12.0,
        http_client: httpx.Client | None = None,
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("DeepSeek timeout must be positive")
        self._api_key = api_key.strip() if api_key is not None else None
        self._model = model
        self._deep_model = deep_model
        self._timeout = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._http = http_client
        self._owns_http = http_client is None

    def explain(
        self,
        evidence: CoachEvidence,
        question: str,
        *,
        deep: bool = False,
    ) -> CoachExplanation:
        if not isinstance(evidence, CoachEvidence):
            raise TypeError("coach client requires CoachEvidence")
        if not self._api_key:
            return local_explanation(evidence, question)
        validation_error: str | None = None
        for _attempt in range(2):
            try:
                explanation = self._request(evidence, question, deep, validation_error)
                _validate_grounding(evidence, explanation)
                return explanation
            except httpx.HTTPError:
                return local_explanation(evidence, question)
            except (CoachResponseError, ValidationError, json.JSONDecodeError, KeyError, TypeError) as exc:
                validation_error = str(exc)
        return local_explanation(evidence, question)

    def close(self) -> None:
        if self._owns_http and self._http is not None:
            self._http.close()
        self._http = None

    def _request(
        self,
        evidence: CoachEvidence,
        question: str,
        deep: bool,
        validation_error: str | None,
    ) -> CoachExplanation:
        client = self._http
        if client is None:
            client = httpx.Client()
            self._http = client
        user_payload: dict[str, Any] = {
            "evidence": evidence.model_dump(mode="json"),
            "question": question.strip() or "请解释第一候选。",
        }
        if validation_error is not None:
            user_payload["previous_validation_error"] = validation_error
            user_payload["instruction"] = "修正错误并重新输出完整 JSON。"
        thinking: dict[str, str] = {"type": "disabled"}
        if deep:
            thinking = {"type": "enabled", "reasoning_effort": "high"}
        response = client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._deep_model if deep else self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "thinking": thinking,
                "response_format": {"type": "json_object"},
                "max_tokens": 1400,
                "stream": False,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise CoachResponseError("DeepSeek returned empty JSON content")
        decoded = json.loads(content)
        if not isinstance(decoded, dict):
            raise CoachResponseError("DeepSeek response must be a JSON object")
        return CoachExplanation.model_validate(
            {
                **decoded,
                "position_id": evidence.position_id,
                "source": "deepseek",
            }
        )


def _validate_grounding(evidence: CoachEvidence, explanation: CoachExplanation) -> None:
    allowed_ids = set(evidence.allowed_move_map)
    if explanation.position_id != evidence.position_id:
        raise CoachResponseError("explanation position does not match evidence")
    if explanation.candidate_id not in allowed_ids:
        raise CoachResponseError("explanation selected an unlisted candidate")
    if any(candidate not in allowed_ids for candidate in explanation.alternatives):
        raise CoachResponseError("explanation alternatives contain an unlisted candidate")
    text = (
        f"{explanation.position_summary}\n{explanation.main_plan}\n{explanation.why}\n"
        f"{explanation.opponent_threat}\n{explanation.training_question}"
    )
    if any(reference not in allowed_ids for reference in _CANDIDATE_REFERENCE.findall(text)):
        raise CoachResponseError("explanation text references an unlisted candidate")
    allowed_uci = {candidate.uci for candidate in evidence.candidates}
    if any(reference not in allowed_uci for reference in _UCI_REFERENCE.findall(text)):
        raise CoachResponseError("explanation contains a move outside the candidate whitelist")
    allowed_notation = set(evidence.allowed_move_map.values())
    if any(reference not in allowed_notation for reference in _CHINESE_MOVE.findall(text)):
        raise CoachResponseError("explanation contains unverified Chinese move notation")
