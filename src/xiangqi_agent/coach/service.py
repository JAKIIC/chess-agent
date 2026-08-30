from __future__ import annotations

from collections.abc import Callable
from threading import Condition, Thread
from typing import Protocol

from xiangqi_agent.domain.coach import CoachEvidence, CoachExplanation

type ExplanationCallback = Callable[[CoachExplanation], None]
type ErrorCallback = Callable[[str], None]


class CoachExplainer(Protocol):
    def explain(
        self, evidence: CoachEvidence, question: str, *, deep: bool = False
    ) -> CoachExplanation: ...

    def close(self) -> None: ...


class CoachService:
    """Run one grounded coaching request at a time and suppress stale answers."""

    def __init__(
        self,
        explainer: CoachExplainer,
        *,
        on_ready: ExplanationCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._explainer = explainer
        self._on_ready = on_ready or _ignore_explanation
        self._on_error = on_error or _ignore_error
        self._condition = Condition()
        self._generation = 0
        self._pending: tuple[int, CoachEvidence, str, bool] | None = None
        self._active_generation: int | None = None
        self._closed = False
        self._worker: Thread | None = None

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def submit(self, evidence: CoachEvidence, question: str, *, deep: bool = False) -> int:
        if not isinstance(evidence, CoachEvidence):
            raise TypeError("coach service requires CoachEvidence")
        with self._condition:
            if self._closed:
                raise RuntimeError("coach service is closed")
            self._generation += 1
            generation = self._generation
            self._pending = (generation, evidence, question, deep)
            if self._worker is None:
                self._worker = Thread(target=self._run, daemon=True, name="coach-service")
                self._worker.start()
            self._condition.notify_all()
            return generation

    def clear(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            self._pending = None
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._pending = None
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=0.5)
        self._explainer.close()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            task = self._next_task()
            if task is None:
                return
            generation, evidence, question, deep = task
            try:
                explanation = self._explainer.explain(evidence, question, deep=deep)
            except (OSError, RuntimeError, ValueError) as exc:
                if self._is_current(generation):
                    self._on_error(str(exc))
                self._finish(generation)
                continue
            if self._is_current(generation):
                if explanation.position_id == evidence.position_id:
                    self._on_ready(explanation)
                else:
                    self._on_error("coach explanation position did not match evidence")
            self._finish(generation)

    def _next_task(self) -> tuple[int, CoachEvidence, str, bool] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._pending is not None)
            if self._closed:
                return None
            if self._pending is None:
                raise RuntimeError("coach service woke without a pending request")
            task = self._pending
            self._pending = None
            self._active_generation = task[0]
            return task

    def _is_current(self, generation: int) -> bool:
        with self._condition:
            return not self._closed and generation == self._generation

    def _finish(self, generation: int) -> None:
        with self._condition:
            if self._active_generation == generation:
                self._active_generation = None


def _ignore_explanation(_explanation: CoachExplanation) -> None:
    pass


def _ignore_error(_message: str) -> None:
    pass
