from __future__ import annotations

import sys
from time import sleep


def emit(line: str) -> None:
    print(line, flush=True)


mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
for raw in sys.stdin:
    command = raw.strip()
    if command == "uci":
        emit("id name Fake Pikafish 1.0")
        emit("option name Threads type spin default 1 min 1 max 128")
        emit("uciok")
    elif command == "isready":
        emit("readyok")
    elif command.startswith("go "):
        if mode == "crash":
            raise SystemExit(7)
        if mode == "hang":
            sleep(60)
        emit("info depth 12 multipv 1 score cp 35 nodes 1000 time 50 pv h2e2 h9g7")
        emit("info depth 10 multipv 2 score mate -4 nodes 800 time 40 pv b2b6")
        emit("info depth 14 multipv 1 score cp 42 nodes 1400 time 75 pv h2e2 h9g7")
        emit("bestmove h2e2")
    elif command == "stop":
        emit("bestmove (none)")
    elif command == "quit":
        raise SystemExit(0)
