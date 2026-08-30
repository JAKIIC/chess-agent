import numpy as np
import pytest

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.templates import PieceTemplateBank, TemplateExtractionError

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}


def _geometry() -> BoardGeometry:
    return BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(
            ((CELL // 2, CELL // 2), (CELL * 8 + CELL // 2, CELL // 2),
             (CELL * 8 + CELL // 2, CELL * 9 + CELL // 2),
             (CELL // 2, CELL * 9 + CELL // 2)),
            (CELL * 9, CELL * 10),
        ),
        (CELL * 9, CELL * 10),
    )


def _render(pieces: tuple[str, ...]) -> np.ndarray:
    frame = np.zeros((CELL * 10, CELL * 9, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(pieces):
        row, column = divmod(index, 9)
        value = PALETTE[symbol]
        frame[row * CELL : (row + 1) * CELL, column * CELL : (column + 1) * CELL, :3] = value
    return frame


def test_initial_position_extracts_all_piece_and_empty_templates() -> None:
    board = parse_fen(START)

    templates = PieceTemplateBank.from_position(board, _geometry(), _render(board.pieces), patch_size=CELL)

    assert templates.symbols == frozenset(".KABNRCPkabnrcp")
    assert templates.example_count("R") == 2
    assert templates.example_count("p") == 5
    assert templates.example_count(".") == 58


def test_template_distance_prefers_the_matching_fixed_theme_symbol() -> None:
    board = parse_fen(START)
    frame = _render(board.pieces)
    templates = PieceTemplateBank.from_position(board, _geometry(), frame, patch_size=CELL)
    red_rook_patch = _geometry().crop_intersections(frame, size=CELL)[81]

    assert templates.distance("R", red_rook_patch) == pytest.approx(0.0)
    assert templates.distance("p", red_rook_patch) > 0.1


def test_partial_position_extracts_only_the_available_theme_classes() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")

    templates = PieceTemplateBank.from_position(
        board,
        _geometry(),
        _render(board.pieces),
        patch_size=CELL,
    )

    assert templates.symbols == frozenset(".Kk")


def test_template_extraction_can_require_all_fixed_theme_classes() -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    with pytest.raises(TemplateExtractionError, match="all 15"):
        PieceTemplateBank.from_position(
            board,
            _geometry(),
            _render(board.pieces),
            patch_size=CELL,
            require_complete=True,
        )
