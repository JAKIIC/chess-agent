import cv2
import numpy as np
import pytest

from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.templates import PieceTemplateBank, TemplateExtractionError

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
THEMED_CELL = 48
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


def _themed_geometry() -> BoardGeometry:
    return BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(
            (
                (THEMED_CELL // 2, THEMED_CELL // 2),
                (THEMED_CELL * 8 + THEMED_CELL // 2, THEMED_CELL // 2),
                (THEMED_CELL * 8 + THEMED_CELL // 2, THEMED_CELL * 9 + THEMED_CELL // 2),
                (THEMED_CELL // 2, THEMED_CELL * 9 + THEMED_CELL // 2),
            ),
            (THEMED_CELL * 9, THEMED_CELL * 10),
        ),
        (THEMED_CELL * 9, THEMED_CELL * 10),
    )


def _themed_patch(symbol: str, *, shift: tuple[int, int] = (0, 0), highlighted: bool = False) -> np.ndarray:
    background = (157, 202, 236, 255)
    patch = np.full((THEMED_CELL, THEMED_CELL, 4), background, dtype=np.uint8)
    if symbol != ".":
        cv2.circle(patch, (24, 24), 18, (143, 207, 254, 255), thickness=-1)
        glyph = (40, 40, 170, 255) if symbol.isupper() else (45, 55, 60, 255)
        cv2.rectangle(patch, (22, 14), (25, 33), glyph, thickness=-1)
        cv2.rectangle(patch, (16, 22), (31, 25), glyph, thickness=-1)
    dx, dy = shift
    if dx or dy:
        matrix = np.asarray(((1.0, 0.0, float(dx)), (0.0, 1.0, float(dy))), dtype=np.float32)
        patch = cv2.warpAffine(
            patch,
            matrix,
            (THEMED_CELL, THEMED_CELL),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=background,
        )
    if highlighted:
        cv2.circle(patch, (24, 24), 21, (105, 140, 155, 255), thickness=2)
    return np.asarray(patch, dtype=np.uint8)


def _themed_frame(pieces: tuple[str, ...]) -> np.ndarray:
    frame = np.empty((THEMED_CELL * 10, THEMED_CELL * 9, 4), dtype=np.uint8)
    for index, symbol in enumerate(pieces):
        row, column = divmod(index, 9)
        frame[
            row * THEMED_CELL : (row + 1) * THEMED_CELL,
            column * THEMED_CELL : (column + 1) * THEMED_CELL,
        ] = _themed_patch(symbol)
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


def test_template_group_match_accepts_any_piece_from_the_expected_side() -> None:
    board = parse_fen(START)
    frame = _render(board.pieces)
    geometry = _geometry()
    templates = PieceTemplateBank.from_position(board, geometry, frame, patch_size=CELL)
    red_pawn_patch = geometry.crop_intersections(frame, size=CELL)[54]

    match = templates.match_any(
        frozenset(symbol for symbol in templates.symbols if symbol.isupper()),
        red_pawn_patch,
    )

    assert match.expected_symbol == "P"
    assert match.distance == pytest.approx(0.0)
    assert match.margin > 0.02
    assert match.confidence > 0.99


@pytest.mark.parametrize("symbol", ["K", "k"])
def test_template_group_survives_piece_shift_and_selection_highlight(symbol: str) -> None:
    board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    templates = PieceTemplateBank.from_position(
        board,
        _themed_geometry(),
        _themed_frame(board.pieces),
        patch_size=THEMED_CELL,
    )

    match = templates.match_any(frozenset({symbol}), _themed_patch(symbol, shift=(4, 3), highlighted=True))

    assert match.expected_symbol == symbol
    assert match.distance < 0.18
    assert match.margin >= 0.02
    assert match.confidence >= 0.8


def test_template_group_confidence_is_independent_of_same_side_class_count() -> None:
    full_board = parse_fen(START)
    partial_board = parse_fen("4k4/9/9/9/9/9/9/9/9/4K4 w")
    geometry = _geometry()
    full_frame = _render(full_board.pieces)
    partial_frame = _render(partial_board.pieces)
    full_templates = PieceTemplateBank.from_position(
        full_board,
        geometry,
        full_frame,
        patch_size=CELL,
    )
    partial_templates = PieceTemplateBank.from_position(
        partial_board,
        geometry,
        partial_frame,
        patch_size=CELL,
    )
    red_king_patch = geometry.crop_intersections(full_frame, size=CELL)[85]

    full_match = full_templates.match_any(
        frozenset(symbol for symbol in full_templates.symbols if symbol.isupper()),
        red_king_patch,
    )
    partial_match = partial_templates.match_any(
        frozenset(symbol for symbol in partial_templates.symbols if symbol.isupper()),
        red_king_patch,
    )

    assert full_match.confidence == pytest.approx(
        partial_match.confidence,
        rel=0.0,
        abs=1e-10,
    )


def test_template_group_match_rejects_symbols_from_opposite_semantic_groups() -> None:
    board = parse_fen(START)
    frame = _render(board.pieces)
    geometry = _geometry()
    templates = PieceTemplateBank.from_position(board, geometry, frame, patch_size=CELL)
    patch = geometry.crop_intersections(frame, size=CELL)[0]

    with pytest.raises(ValueError, match="one semantic group"):
        templates.match_any(frozenset({"R", "r"}), patch)


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
