from __future__ import annotations

import numpy as np

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.ui.board_widget import BoardWidget

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


def _pixels(widget: BoardWidget) -> np.ndarray:
    image = widget.grab().toImage()
    return np.frombuffer(image.bits(), dtype=np.uint8).copy()


def test_board_widget_draws_the_confirmed_last_move_highlight(qtbot: object) -> None:
    board = parse_fen(START)
    move = next(move for move in legal_moves(board) if move.uci == "h2e2")
    after = apply_move(board, move)
    widget = BoardWidget()
    widget.resize(520, 650)
    qtbot.addWidget(widget)  # type: ignore[attr-defined]

    widget.set_board(after)
    plain = _pixels(widget)
    widget.set_board(after, last_move=move)
    highlighted = _pixels(widget)

    assert widget.last_move == move
    assert np.count_nonzero(plain != highlighted) > 100
