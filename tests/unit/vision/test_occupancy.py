from dataclasses import fields, replace
from math import nan

import cv2
import numpy as np
import pytest

from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.occupancy import (
    CircularOccupancyObserver,
    KnownPositionOccupancyObserver,
    OccupancyComparison,
    OccupancyEvidence,
    compare_occupancy,
)

START_BOARD = parse_fen(
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
)


def test_occupancy_evidence_requires_ninety_finite_typed_values() -> None:
    occupied = tuple(piece != "." for piece in START_BOARD.pieces)
    evidence = OccupancyEvidence(occupied, (0.9,) * 90, "test-v1")

    assert evidence.occupied == occupied
    assert evidence.confidences == (0.9,) * 90

    with pytest.raises(ValueError, match="90 booleans"):
        OccupancyEvidence(occupied[:-1], (0.9,) * 90, "test-v1")
    with pytest.raises(ValueError, match="90 finite confidences"):
        OccupancyEvidence(occupied, (0.9,) * 89 + (nan,), "test-v1")
    with pytest.raises(ValueError, match="90 finite confidences"):
        OccupancyEvidence(occupied, (0.9,) * 89 + (True,), "test-v1")
    with pytest.raises(ValueError, match="between zero and one"):
        OccupancyEvidence(occupied, (0.9,) * 89 + (1.1,), "test-v1")
    with pytest.raises(ValueError, match="algorithm_version"):
        OccupancyEvidence(occupied, (0.9,) * 90, " ")


def test_compare_occupancy_reports_mismatch_and_low_confidence_separately() -> None:
    evidence = _evidence_for(START_BOARD, confidence=0.95)
    occupied = list(evidence.occupied)
    occupied[0] = not occupied[0]
    confidences = list(evidence.confidences)
    confidences[1] = 0.49

    comparison = compare_occupancy(
        replace(
            evidence,
            occupied=tuple(occupied),
            confidences=tuple(confidences),
        ),
        START_BOARD,
        minimum_confidence=0.50,
    )

    assert comparison == OccupancyComparison(
        mismatched_points=(0,),
        low_confidence_points=(1,),
    )
    assert not comparison.accepted


def test_compare_occupancy_accepts_only_a_complete_confident_match() -> None:
    evidence = _evidence_for(START_BOARD, confidence=0.65)

    comparison = compare_occupancy(
        evidence,
        START_BOARD,
        minimum_confidence=0.65,
    )

    assert comparison.accepted
    assert comparison.mismatched_points == ()
    assert comparison.low_confidence_points == ()


def test_compare_occupancy_validates_public_inputs() -> None:
    evidence = _evidence_for(START_BOARD, confidence=0.9)

    with pytest.raises(TypeError, match="BoardState"):
        compare_occupancy(evidence, object(), minimum_confidence=0.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="between zero and one"):
        compare_occupancy(evidence, START_BOARD, minimum_confidence=-0.1)
    with pytest.raises(ValueError, match="between zero and one"):
        compare_occupancy(evidence, START_BOARD, minimum_confidence=1.1)
    with pytest.raises(TypeError, match="number"):
        compare_occupancy(evidence, START_BOARD, minimum_confidence=True)


@pytest.mark.parametrize(
    ("frame_size", "orientation"),
    (
        ((450, 500), Orientation.RED_BOTTOM),
        ((900, 1000), Orientation.RED_BOTTOM),
        ((450, 500), Orientation.BLACK_BOTTOM),
    ),
)
def test_circular_observer_detects_synthetic_board_at_two_scales_and_orientations(
    frame_size: tuple[int, int],
    orientation: Orientation,
) -> None:
    frame, geometry = _render_board(START_BOARD, frame_size, orientation)
    observer = CircularOccupancyObserver()

    first = observer.observe(frame, geometry)
    second = observer.observe(frame, geometry)

    assert first == second
    assert first.occupied == tuple(piece != "." for piece in START_BOARD.pieces)
    assert all(confidence >= 0.55 for confidence in first.confidences)
    assert first.algorithm_version == "circular-occupancy-v1"


def test_observer_evidence_owns_no_frame_or_patch_arrays() -> None:
    frame, geometry = _render_board(
        START_BOARD,
        (450, 500),
        Orientation.RED_BOTTOM,
    )
    evidence = CircularOccupancyObserver().observe(frame, geometry)

    frame[:] = 0

    assert evidence.occupied == tuple(piece != "." for piece in START_BOARD.pieces)
    assert all(not isinstance(getattr(evidence, field.name), np.ndarray) for field in fields(evidence))


def test_uniform_overlay_cannot_match_a_confirmed_starting_board() -> None:
    _, geometry = _render_board(
        START_BOARD,
        (450, 500),
        Orientation.RED_BOTTOM,
    )
    white = np.full((500, 450, 4), 255, dtype=np.uint8)

    evidence = CircularOccupancyObserver().observe(white, geometry)
    comparison = compare_occupancy(
        evidence,
        START_BOARD,
        minimum_confidence=0.65,
    )

    assert not comparison.accepted
    assert comparison.mismatched_points or comparison.low_confidence_points


def test_known_position_observer_calibrates_theme_then_tracks_changed_occupancy() -> None:
    baseline, geometry = _render_board(
        START_BOARD,
        (900, 1000),
        Orientation.RED_BOTTOM,
    )
    move = next(move for move in legal_moves(START_BOARD) if move.uci == "h2e2")
    after = apply_move(START_BOARD, move)
    moved, moved_geometry = _render_board(
        after,
        (900, 1000),
        Orientation.RED_BOTTOM,
    )
    observer = KnownPositionOccupancyObserver(START_BOARD)

    baseline_evidence = observer.observe(baseline, geometry)
    moved_evidence = observer.observe(moved, moved_geometry)

    assert compare_occupancy(
        baseline_evidence,
        START_BOARD,
        minimum_confidence=0.65,
    ).accepted
    assert compare_occupancy(
        moved_evidence,
        after,
        minimum_confidence=0.65,
    ).accepted
    assert moved_evidence.algorithm_version == "known-position-template-occupancy-v1"


def test_known_position_observer_rejects_an_unseparated_uniform_baseline() -> None:
    _, geometry = _render_board(
        START_BOARD,
        (900, 1000),
        Orientation.RED_BOTTOM,
    )
    white = np.full((1000, 900, 4), 255, dtype=np.uint8)

    evidence = KnownPositionOccupancyObserver(START_BOARD).observe(white, geometry)
    comparison = compare_occupancy(
        evidence,
        START_BOARD,
        minimum_confidence=0.65,
    )

    assert not comparison.accepted
    assert comparison.low_confidence_points == tuple(range(90))


def _evidence_for(board: BoardState, *, confidence: float) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (confidence,) * 90,
        "test-v1",
    )


def _render_board(
    board: BoardState,
    frame_size: tuple[int, int],
    orientation: Orientation,
) -> tuple[np.ndarray, BoardGeometry]:
    width, height = frame_size
    margin_x = round(width * 0.06)
    margin_y = round(height * 0.06)
    pixel_quad = (
        (margin_x, margin_y),
        (width - margin_x, margin_y),
        (width - margin_x, height - margin_y),
        (margin_x, height - margin_y),
    )
    geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(pixel_quad, frame_size),
        frame_size,
        orientation,
    )
    frame = np.full((height, width, 4), (190, 214, 232, 255), dtype=np.uint8)
    points = geometry.grid_points()
    line_width = max(1, round(width / 450 * 2))

    physical_rows = tuple(tuple(points[row * 9 + column] for column in range(9)) for row in range(10))
    for row in physical_rows:
        cv2.line(
            frame,
            tuple(round(value) for value in row[0]),
            tuple(round(value) for value in row[-1]),
            (70, 100, 120, 255),
            line_width,
            cv2.LINE_AA,
        )
    for column in range(9):
        cv2.line(
            frame,
            tuple(round(value) for value in physical_rows[0][column]),
            tuple(round(value) for value in physical_rows[-1][column]),
            (70, 100, 120, 255),
            line_width,
            cv2.LINE_AA,
        )

    radius = max(16, round(width / 450 * 19))
    border = max(2, round(width / 450 * 3))
    for index, piece in enumerate(board.pieces):
        if piece == ".":
            continue
        center = tuple(round(value) for value in points[index])
        cv2.circle(frame, center, radius, (35, 55, 75, 255), border, cv2.LINE_AA)
        cv2.circle(frame, center, radius - border, (95, 140, 175, 255), -1, cv2.LINE_AA)
        cv2.line(
            frame,
            (center[0] - radius // 2, center[1]),
            (center[0] + radius // 2, center[1]),
            (25, 40, 55, 255),
            border,
            cv2.LINE_AA,
        )
    return frame, geometry
