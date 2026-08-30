from __future__ import annotations

from threading import Event, Lock

from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.engine.service import AnalysisService

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _analysis(board: BoardState, movetime_ms: int) -> EngineAnalysis:
    line = EngineLine(
        position_id=board.position_id,
        depth=12 if movetime_ms < 100 else 18,
        seldepth=20,
        multipv=1,
        score_cp=35,
        mate_in=None,
        nodes=1000,
        nps=10000,
        time_ms=movetime_ms,
        pv=("h2e2",),
    )
    return EngineAnalysis(
        position_id=board.position_id,
        duration_ms=movetime_ms,
        depth=line.depth,
        nodes=1000,
        lines=(line,),
        bestmove="h2e2",
        engine_name="fake",
    )


class ImmediateEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.stop_calls = 0
        self.closed = False

    def start(self) -> None:
        pass

    def analyse(
        self,
        board: BoardState,
        *,
        movetime_ms: int,
        multipv: int,
        timeout: float | None = None,
    ) -> EngineAnalysis:
        self.calls.append((board.position_id, movetime_ms, multipv))
        return _analysis(board, movetime_ms)

    def stop(self) -> None:
        self.stop_calls += 1

    def close(self) -> None:
        self.closed = True


class BlockingFirstEngine(ImmediateEngine):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = Event()
        self.release_first = Event()
        self._call_lock = Lock()
        self._call_count = 0

    def analyse(
        self,
        board: BoardState,
        *,
        movetime_ms: int,
        multipv: int,
        timeout: float | None = None,
    ) -> EngineAnalysis:
        with self._call_lock:
            self._call_count += 1
            call_number = self._call_count
        self.calls.append((board.position_id, movetime_ms, multipv))
        if call_number == 1:
            self.first_started.set()
            assert self.release_first.wait(2.0)
        return _analysis(board, movetime_ms)

    def stop(self) -> None:
        super().stop()
        self.release_first.set()


def test_service_emits_quick_then_deep_for_one_confirmed_position() -> None:
    engine = ImmediateEngine()
    seen: list[tuple[str, str]] = []
    completed = Event()
    board = parse_fen(START)
    service = AnalysisService(
        engine,
        quick_ms=50,
        deep_ms=100,
        multipv=3,
        on_quick=lambda item: seen.append(("quick", item.position_id)),
        on_deep=lambda item: (seen.append(("deep", item.position_id)), completed.set()),
    )

    service.submit(board)
    assert completed.wait(2.0)
    service.close()
    service.close()

    assert seen == [("quick", board.position_id), ("deep", board.position_id)]
    assert engine.calls == [
        (board.position_id, 50, 3),
        (board.position_id, 100, 3),
    ]
    assert engine.closed


def test_new_position_stops_old_search_and_never_emits_stale_result() -> None:
    engine = BlockingFirstEngine()
    seen: list[tuple[str, str]] = []
    completed = Event()
    old_board = parse_fen(START)
    move = next(move for move in legal_moves(old_board) if move.uci == "h2e2")
    new_board = apply_move(old_board, move)
    service = AnalysisService(
        engine,
        quick_ms=50,
        deep_ms=100,
        multipv=3,
        on_quick=lambda item: seen.append(("quick", item.position_id)),
        on_deep=lambda item: (seen.append(("deep", item.position_id)), completed.set()),
    )

    service.submit(old_board)
    assert engine.first_started.wait(2.0)
    service.submit(new_board)
    assert completed.wait(2.0)
    service.close()

    assert engine.stop_calls >= 1
    assert seen == [
        ("quick", new_board.position_id),
        ("deep", new_board.position_id),
    ]


def test_service_reports_current_generation_failure_without_emitting_analysis() -> None:
    class FailingEngine(ImmediateEngine):
        def analyse(
            self,
            board: BoardState,
            *,
            movetime_ms: int,
            multipv: int,
            timeout: float | None = None,
        ) -> EngineAnalysis:
            raise RuntimeError("engine failed")

    errors: list[str] = []
    failed = Event()
    service = AnalysisService(
        FailingEngine(),
        on_error=lambda message: (errors.append(message), failed.set()),
    )

    service.submit(parse_fen(START))
    assert failed.wait(2.0)
    service.close()

    assert errors == ["engine failed"]
