from xiangqi_agent.engine.protocol import parse_bestmove_line, parse_info_line


def test_parse_multipv_cp_line() -> None:
    line = parse_info_line(
        "info depth 18 seldepth 27 multipv 2 score cp -43 nodes 900 nps 1000 "
        "time 88 pv h2e2 h9g7",
        position_id="abc",
    )

    assert line is not None
    assert line.position_id == "abc"
    assert line.depth == 18
    assert line.seldepth == 27
    assert line.multipv == 2
    assert line.score_cp == -43
    assert line.mate_in is None
    assert line.nodes == 900
    assert line.nps == 1000
    assert line.time_ms == 88
    assert line.pv == ("h2e2", "h9g7")


def test_parse_mate_separately_from_cp_and_default_multipv() -> None:
    line = parse_info_line("info depth 20 score mate 3 pv e4e9", "abc")

    assert line is not None
    assert line.multipv == 1
    assert line.mate_in == 3
    assert line.score_cp is None


def test_parse_info_ignores_bounds_and_unknown_tokens() -> None:
    line = parse_info_line(
        "info string ignored depth 10 score cp 12 lowerbound hashfull 3 pv a0a1",
        "abc",
    )

    assert line is not None
    assert line.depth == 10
    assert line.score_cp == 12
    assert line.pv == ("a0a1",)


def test_parse_info_returns_none_for_non_info_incomplete_or_malformed_lines() -> None:
    assert parse_info_line("bestmove h2e2", "abc") is None
    assert parse_info_line("info depth x score cp 3 pv h2e2", "abc") is None
    assert parse_info_line("info depth 3 pv h2e2", "abc") is None
    assert parse_info_line("info depth 3 score cp 4", "abc") is None


def test_parse_bestmove_accepts_move_and_none_but_rejects_other_lines() -> None:
    assert parse_bestmove_line("bestmove h2e2 ponder h9g7") == "h2e2"
    assert parse_bestmove_line("bestmove (none)") is None
    assert parse_bestmove_line("info depth 1") is None
