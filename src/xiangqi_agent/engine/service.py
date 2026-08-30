from __future__ import annotations

from collections.abc import Callable
from threading import Condition, Thread
from typing import Protocol

from xiangqi_agent.domain.analysis import EngineAnalysis
from xiangqi_agent.domain.board import BoardState

type AnalysisCallback = Callable[[EngineAnalysis], None]
type ErrorCallback = Callable[[str], None]


class AnalysisEngine(Protocol):
    def start(self) -> None: ...

    def analyse(
        self,
        board: BoardState,
        *,
        movetime_ms: int,
        multipv: int,
        timeout: float | None = None,
    ) -> EngineAnalysis: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class AnalysisService:
    """Run quick/deep UCI analysis serially and discard stale generations."""

    def __init__(
        self,
        engine: AnalysisEngine,
        *,
        quick_ms: int = 500,
        deep_ms: int = 3000,
        multipv: int = 3,
        on_quick: AnalysisCallback | None = None,
        on_deep: AnalysisCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (quick_ms, deep_ms, multipv)
        ):
            raise ValueError("analysis durations and multipv must be positive integers")
        self._engine = engine
        self._quick_ms = quick_ms
        self._deep_ms = deep_ms
        self._multipv = multipv
        self._on_quick = on_quick or _ignore_analysis
        self._on_deep = on_deep or _ignore_analysis
        self._on_error = on_error or _ignore_error
        self._condition = Condition()
        self._generation = 0
        self._pending: tuple[int, BoardState] | None = None
        self._active_generation: int | None = None
        self._closed = False
        self._worker: Thread | None = None

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    def submit(self, board: BoardState) -> int:
        if not isinstance(board, BoardState):
            raise TypeError("analysis requires a confirmed BoardState")
        with self._condition:
            if self._closed:
                raise RuntimeError("analysis service is closed")
            self._generation += 1
            generation = self._generation
            self._pending = (generation, board)
            should_stop = self._active_generation is not None
            if self._worker is None:
                self._worker = Thread(
                    target=self._run,
                    daemon=True,
                    name="analysis-service",
                )
                self._worker.start()
            self._condition.notify_all()
        if should_stop:
            self._engine.stop()
        return generation

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._pending = None
            should_stop = self._active_generation is not None
            worker = self._worker
            self._condition.notify_all()
        if should_stop:
            self._engine.stop()
        if worker is not None:
            worker.join(timeout=10.0)
            if worker.is_alive():
                self._engine.close()
                worker.join(timeout=2.0)
        else:
            self._engine.close()

    def _run(self) -> None:
        try:
            self._engine.start()
            while True:
                task = self._next_task()
                if task is None:
                    return
                generation, board = task
                if not self._analyse_phase(generation, board, self._quick_ms, self._on_quick):
                    self._finish_generation(generation)
                    continue
                self._analyse_phase(generation, board, self._deep_ms, self._on_deep)
                self._finish_generation(generation)
        except (OSError, RuntimeError, ValueError) as exc:
            if not self._is_closed():
                self._on_error(str(exc))
        finally:
            self._engine.close()

    def _next_task(self) -> tuple[int, BoardState] | None:
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._pending is not None)
            if self._closed:
                return None
            if self._pending is None:
                raise RuntimeError("analysis service woke without a pending task")
            task = self._pending
            self._pending = None
            self._active_generation = task[0]
            return task

    def _analyse_phase(
        self,
        generation: int,
        board: BoardState,
        movetime_ms: int,
        callback: AnalysisCallback,
    ) -> bool:
        try:
            analysis = self._engine.analyse(
                board,
                movetime_ms=movetime_ms,
                multipv=self._multipv,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if self._is_current(generation):
                self._on_error(str(exc))
            return False
        if not self._is_current(generation):
            return False
        if analysis.position_id != board.position_id:
            self._on_error("engine analysis position_id did not match the confirmed board")
            return False
        callback(analysis)
        return True

    def _is_current(self, generation: int) -> bool:
        with self._condition:
            return not self._closed and generation == self._generation

    def _is_closed(self) -> bool:
        with self._condition:
            return self._closed

    def _finish_generation(self, generation: int) -> None:
        with self._condition:
            if self._active_generation == generation:
                self._active_generation = None


def _ignore_analysis(_analysis: EngineAnalysis) -> None:
    pass


def _ignore_error(_message: str) -> None:
    pass
