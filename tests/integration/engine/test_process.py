from __future__ import annotations

import sys
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import pytest

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.engine.process import (
    EngineCrashedError,
    EngineProcessError,
    EngineTimeoutError,
    PikafishProcess,
)

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
FIXTURE = Path(__file__).parents[2] / "fixtures" / "engine" / "fake_uci_engine.py"


def _engine(mode: str = "normal") -> PikafishProcess:
    return PikafishProcess(
        Path(sys.executable),
        arguments=(str(FIXTURE), mode),
        threads=2,
        hash_mb=256,
        startup_timeout=2.0,
        shutdown_timeout=1.0,
    )


def test_managed_process_handshakes_analyses_multipv_and_closes_without_residue() -> None:
    engine = _engine()
    board = parse_fen(START)

    engine.start()
    analysis = engine.analyse(board, movetime_ms=100, multipv=2)
    process_id = engine.process_id
    engine.close()
    engine.close()

    assert engine.engine_name == "Fake Pikafish 1.0"
    assert analysis.position_id == board.position_id
    assert analysis.bestmove == "h2e2"
    assert analysis.depth == 14
    assert [line.multipv for line in analysis.lines] == [1, 2]
    assert analysis.lines[0].score_cp == 42
    assert analysis.lines[1].mate_in == -4
    assert process_id is not None
    assert not engine.is_running


def test_black_to_move_scores_are_normalized_to_red_perspective() -> None:
    engine = _engine()
    board = parse_fen(START.replace(" w", " b"))
    engine.start()

    try:
        analysis = engine.analyse(board, movetime_ms=100, multipv=2)
    finally:
        engine.close()

    assert analysis.lines[0].score_cp == -42
    assert analysis.lines[1].mate_in == 4


def test_engine_crash_during_analysis_is_reported_and_close_remains_safe() -> None:
    engine = _engine("crash")
    engine.start()

    with pytest.raises(EngineCrashedError, match="exited"):
        engine.analyse(parse_fen(START), movetime_ms=50, multipv=1)

    engine.close()
    assert not engine.is_running


def test_engine_analysis_timeout_sends_stop_and_can_be_closed() -> None:
    engine = _engine("hang")
    engine.start()

    with pytest.raises(EngineTimeoutError, match="timed out"):
        engine.analyse(parse_fen(START), movetime_ms=10, multipv=1, timeout=0.05)

    engine.close()
    assert not engine.is_running


def test_stop_can_interrupt_an_analysis_from_another_thread() -> None:
    engine = _engine("interruptible")
    engine.start()
    failures: list[BaseException] = []

    def analyse() -> None:
        try:
            engine.analyse(parse_fen(START), movetime_ms=1000, multipv=1, timeout=1.0)
        except EngineProcessError as exc:  # the stopped search has no analysis line
            failures.append(exc)

    worker = Thread(target=analyse)
    worker.start()
    sleep(0.05)
    started = monotonic()
    engine.stop()
    worker.join(timeout=0.3)
    elapsed = monotonic() - started
    engine.close()

    assert not worker.is_alive()
    assert elapsed < 0.3
    assert failures


def test_eval_file_in_unicode_workspace_is_loaded_from_engine_working_directory(
    tmp_path: Path,
) -> None:
    eval_file = tmp_path / "象棋模型" / "pikafish.nnue"
    eval_file.parent.mkdir()
    eval_file.write_bytes(b"test network")
    engine = PikafishProcess(
        Path(sys.executable),
        arguments=(str(FIXTURE.resolve()), "require_local_eval"),
        eval_file=eval_file,
        startup_timeout=2.0,
        shutdown_timeout=1.0,
    )

    engine.start()
    try:
        analysis = engine.analyse(parse_fen(START), movetime_ms=50, multipv=1)
    finally:
        engine.close()

    assert analysis.bestmove == "h2e2"
