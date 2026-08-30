from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, RLock, Thread
from time import monotonic

from xiangqi_agent.domain.analysis import EngineAnalysis, EngineLine
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.engine.protocol import parse_bestmove_line, parse_info_line


class EngineProcessError(RuntimeError):
    """The managed UCI process could not complete an operation."""


class EngineCrashedError(EngineProcessError):
    """The UCI process exited before completing the protocol operation."""


class EngineTimeoutError(EngineProcessError):
    """The UCI process did not respond before the bounded deadline."""


class PikafishProcess:
    def __init__(
        self,
        executable: Path,
        *,
        arguments: tuple[str, ...] = (),
        threads: int = 2,
        hash_mb: int = 256,
        eval_file: Path | None = None,
        startup_timeout: float = 5.0,
        shutdown_timeout: float = 2.0,
    ) -> None:
        if not isinstance(executable, Path):
            raise TypeError("engine executable must be a Path")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (threads, hash_mb)):
            raise ValueError("engine threads and hash must be positive integers")
        if startup_timeout <= 0 or shutdown_timeout <= 0:
            raise ValueError("engine timeouts must be positive")
        self._executable = executable
        self._arguments = arguments
        self._threads = threads
        self._hash_mb = hash_mb
        self._eval_file = eval_file
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._lines: Queue[str | None] = Queue()
        self._lock = RLock()
        self._analysis_lock = Lock()
        self._stdin_lock = Lock()
        self._engine_name = "unknown"

    @property
    def engine_name(self) -> str:
        return self._engine_name

    @property
    def process_id(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            if not self._executable.is_file():
                raise FileNotFoundError(self._executable)
            if self._eval_file is not None and not self._eval_file.is_file():
                raise FileNotFoundError(self._eval_file)
            self._lines = Queue()
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            working_directory = (
                self._eval_file.parent if self._eval_file is not None else self._executable.parent
            )
            self._process = subprocess.Popen(
                [str(self._executable), *self._arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=creationflags,
                cwd=working_directory,
            )
            self._reader = Thread(target=self._read_stdout, daemon=True, name="uci-reader")
            self._reader.start()
            try:
                self._send("uci")
                self._wait_for("uciok", self._startup_timeout, capture_identity=True)
                self._send(f"setoption name Threads value {self._threads}")
                self._send(f"setoption name Hash value {self._hash_mb}")
                if self._eval_file is not None:
                    self._send(f"setoption name EvalFile value {self._eval_file.name}")
                self._send("isready")
                self._wait_for("readyok", self._startup_timeout)
            except BaseException:
                self.close()
                raise

    def analyse(
        self,
        board: BoardState,
        *,
        movetime_ms: int,
        multipv: int = 3,
        timeout: float | None = None,
    ) -> EngineAnalysis:
        if isinstance(movetime_ms, bool) or not isinstance(movetime_ms, int) or movetime_ms <= 0:
            raise ValueError("movetime_ms must be a positive integer")
        if isinstance(multipv, bool) or not isinstance(multipv, int) or multipv <= 0:
            raise ValueError("multipv must be a positive integer")
        with self._analysis_lock:
            self._require_running()
            self._drain_stale_lines()
            self._send(f"setoption name MultiPV value {multipv}")
            self._send(f"position fen {board.fen}")
            self._send(f"go movetime {movetime_ms}")
            started = monotonic()
            deadline = started + (timeout if timeout is not None else movetime_ms / 1000 + 2.0)
            latest: dict[int, EngineLine] = {}
            bestmove: str | None = None
            try:
                while True:
                    line = self._next_line(deadline)
                    if line.startswith("bestmove"):
                        bestmove = parse_bestmove_line(line)
                        break
                    parsed = parse_info_line(line, board.position_id)
                    if parsed is None or parsed.multipv > multipv:
                        continue
                    previous = latest.get(parsed.multipv)
                    if previous is None or parsed.depth >= previous.depth:
                        latest[parsed.multipv] = parsed
            except EngineTimeoutError:
                self.stop()
                raise
            if not latest:
                raise EngineProcessError("engine returned bestmove without analysis lines")
            lines = tuple(
                _normalize_red_perspective(latest[index], board)
                for index in sorted(latest)
            )
            duration_ms = round((monotonic() - started) * 1000)
            return EngineAnalysis(
                position_id=board.position_id,
                duration_ms=duration_ms,
                depth=max(line.depth for line in lines),
                nodes=max((line.nodes or 0) for line in lines),
                lines=lines,
                bestmove=bestmove,
                engine_name=self._engine_name,
            )

    def stop(self) -> None:
        with self._lock:
            if self.is_running:
                self._send("stop")

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                try:
                    self._send("stop")
                    self._send("quit")
                except EngineProcessError:
                    pass
                try:
                    process.wait(timeout=self._shutdown_timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=self._shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=self._shutdown_timeout)
            if self._reader is not None:
                self._reader.join(timeout=self._shutdown_timeout)
            self._process = None
            self._reader = None

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            self._lines.put(None)
            return
        try:
            for line in process.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def _send(self, command: str) -> None:
        with self._stdin_lock:
            process = self._require_running()
            if process.stdin is None:
                raise EngineCrashedError("engine stdin is unavailable")
            try:
                process.stdin.write(command + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise EngineCrashedError("engine exited while receiving a command") from exc

    def _wait_for(self, expected: str, timeout: float, *, capture_identity: bool = False) -> None:
        deadline = monotonic() + timeout
        while True:
            line = self._next_line(deadline)
            if capture_identity and line.startswith("id name "):
                self._engine_name = line.removeprefix("id name ").strip() or "unknown"
            if line == expected:
                return

    def _next_line(self, deadline: float) -> str:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise EngineTimeoutError("engine operation timed out")
        try:
            line = self._lines.get(timeout=remaining)
        except Empty as exc:
            raise EngineTimeoutError("engine operation timed out") from exc
        if line is None:
            code = self._process.poll() if self._process is not None else None
            raise EngineCrashedError(f"engine exited unexpectedly with code {code}")
        return line

    def _require_running(self) -> subprocess.Popen[str]:
        if not self.is_running or self._process is None:
            code = self._process.poll() if self._process is not None else None
            raise EngineCrashedError(f"engine is not running; exited with code {code}")
        return self._process

    def _drain_stale_lines(self) -> None:
        while True:
            try:
                line = self._lines.get_nowait()
            except Empty:
                return
            if line is None:
                code = self._process.poll() if self._process is not None else None
                raise EngineCrashedError(f"engine exited unexpectedly with code {code}")


def _normalize_red_perspective(line: EngineLine, board: BoardState) -> EngineLine:
    if board.side_to_move == "w":
        return line
    return replace(
        line,
        score_cp=None if line.score_cp is None else -line.score_cp,
        mate_in=None if line.mate_in is None else -line.mate_in,
    )
