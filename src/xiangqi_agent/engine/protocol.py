from __future__ import annotations

import re

from xiangqi_agent.domain.analysis import EngineLine

_MOVE = re.compile(r"^[a-i][0-9][a-i][0-9]$")


def parse_info_line(text: str, position_id: str) -> EngineLine | None:
    tokens = text.split()
    if not tokens or tokens[0] != "info":
        return None
    values: dict[str, int] = {}
    score_cp: int | None = None
    mate_in: int | None = None
    pv: tuple[str, ...] = ()
    index = 1
    try:
        while index < len(tokens):
            token = tokens[index]
            if token in {"depth", "seldepth", "multipv", "nodes", "nps", "time"}:
                values[token] = int(tokens[index + 1])
                index += 2
                continue
            if token == "score":
                score_kind = tokens[index + 1]
                score_value = int(tokens[index + 2])
                if score_kind == "cp":
                    score_cp = score_value
                elif score_kind == "mate":
                    mate_in = score_value
                else:
                    return None
                index += 3
                continue
            if token == "pv":
                pv = tuple(tokens[index + 1 :])
                break
            index += 1
    except (IndexError, ValueError):
        return None
    if "depth" not in values or not pv or (score_cp is None) == (mate_in is None):
        return None
    try:
        return EngineLine(
            position_id=position_id,
            depth=values["depth"],
            seldepth=values.get("seldepth"),
            multipv=values.get("multipv", 1),
            score_cp=score_cp,
            mate_in=mate_in,
            nodes=values.get("nodes"),
            nps=values.get("nps"),
            time_ms=values.get("time"),
            pv=pv,
        )
    except ValueError:
        return None


def parse_bestmove_line(text: str) -> str | None:
    tokens = text.split()
    if len(tokens) < 2 or tokens[0] != "bestmove":
        return None
    move = tokens[1]
    return move if _MOVE.fullmatch(move) is not None else None
