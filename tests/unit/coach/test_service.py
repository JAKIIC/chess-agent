from __future__ import annotations

from threading import Event

from tests.integration.coach.test_deepseek_client import _evidence
from xiangqi_agent.coach.fallback import local_explanation
from xiangqi_agent.coach.service import CoachService
from xiangqi_agent.domain.coach import CoachEvidence, CoachExplanation


class ImmediateExplainer:
    def __init__(self) -> None:
        self.closed = False

    def explain(
        self, evidence: CoachEvidence, question: str, *, deep: bool = False
    ) -> CoachExplanation:
        return local_explanation(evidence, question)

    def close(self) -> None:
        self.closed = True


def test_coach_service_emits_current_explanation_and_closes() -> None:
    explainer = ImmediateExplainer()
    ready = Event()
    seen: list[CoachExplanation] = []
    service = CoachService(
        explainer,
        on_ready=lambda explanation: (seen.append(explanation), ready.set()),
    )

    service.submit(_evidence(), "为什么？")
    assert ready.wait(2.0)
    service.close()
    service.close()

    assert len(seen) == 1
    assert explainer.closed


def test_clearing_for_a_new_position_discards_old_answer() -> None:
    class BlockingExplainer(ImmediateExplainer):
        def __init__(self) -> None:
            super().__init__()
            self.started = Event()
            self.release = Event()

        def explain(
            self, evidence: CoachEvidence, question: str, *, deep: bool = False
        ) -> CoachExplanation:
            self.started.set()
            assert self.release.wait(2.0)
            return local_explanation(evidence, question)

    explainer = BlockingExplainer()
    seen: list[CoachExplanation] = []
    service = CoachService(explainer, on_ready=seen.append)

    service.submit(_evidence(), "旧问题")
    assert explainer.started.wait(2.0)
    service.clear()
    explainer.release.set()
    service.close()

    assert seen == []


def test_explainer_failure_is_reported_without_a_stale_answer() -> None:
    class FailingExplainer(ImmediateExplainer):
        def explain(
            self, evidence: CoachEvidence, question: str, *, deep: bool = False
        ) -> CoachExplanation:
            raise RuntimeError("coach unavailable")

    failed = Event()
    errors: list[str] = []
    service = CoachService(
        FailingExplainer(), on_error=lambda message: (errors.append(message), failed.set())
    )

    service.submit(_evidence(), "问题")
    assert failed.wait(2.0)
    service.close()

    assert errors == ["coach unavailable"]
