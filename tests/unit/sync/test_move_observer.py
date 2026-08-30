import numpy as np
import pytest

from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver, ObservationStatus
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad

START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
CELL = 24
PALETTE = {symbol: index * 15 for index, symbol in enumerate(".KABNRCPkabnrcp", start=1)}


def _geometry() -> BoardGeometry:
    return BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(
            ((12, 12), (204, 12), (204, 228), (12, 228)),
            (216, 240),
        ),
        (216, 240),
    )


def _render(board: BoardState) -> np.ndarray:
    frame = np.zeros((240, 216, 4), dtype=np.uint8)
    frame[..., 3] = 255
    for index, symbol in enumerate(board.pieces):
        row, column = divmod(index, 9)
        value = PALETTE[symbol]
        frame[row * CELL : (row + 1) * CELL, column * CELL : (column + 1) * CELL, :3] = value
    return frame


def _move(board: BoardState, uci: str):
    return next(move for move in legal_moves(board) if move.uci == uci)


def test_observer_accepts_the_unique_legal_origin_and_destination_pair() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after_board = apply_move(board, move)

    result = LegalMoveDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        _render(after_board),
        _geometry(),
    )

    assert result.status is ObservationStatus.ACCEPTED
    assert result.move == move
    assert not hasattr(result, "after")
    assert result.evidence_score > 0.9
    assert result.evidence.candidates[0].move.uci == "h2e2"


def test_observer_does_not_accept_only_one_changed_intersection() -> None:
    board = parse_fen(START)
    before = _render(board)
    after = before.copy()
    move = _move(board, "h2e2")
    source_row, source_column = divmod(move.from_index, 9)
    after[
        source_row * CELL : (source_row + 1) * CELL,
        source_column * CELL : (source_column + 1) * CELL,
        :3,
    ] = PALETTE["."]

    result = LegalMoveDiffObserver(patch_size=CELL).observe(board, before, after, _geometry())

    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert result.evidence.rejection_reasons


def test_observer_rejects_two_highlighted_intersections_when_no_piece_moved() -> None:
    board = parse_fen(START)
    before = _render(board)
    after = before.copy()
    move = _move(board, "h2e2")
    for index in (move.from_index, move.to_index):
        row, column = divmod(index, 9)
        patch = after[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ]
        patch[:] = np.clip(patch.astype(np.int16) + 60, 0, 255).astype(np.uint8)

    result = LegalMoveDiffObserver(patch_size=CELL).observe(board, before, after, _geometry())

    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert result.evidence.rejection_reasons


def test_observer_honors_a_stricter_configured_minimum_evidence_score() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = _render(apply_move(board, move))

    result = LegalMoveDiffObserver(patch_size=CELL, min_evidence_score=0.999).observe(
        board,
        _render(board),
        after,
        _geometry(),
    )

    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert "evidence_score" in result.evidence.rejection_reasons


def test_observer_keeps_the_default_semantic_margin_above_one_sample() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    before = _render(board)
    after = before.copy()
    for index, value in ((move.from_index, PALETTE["."]), (move.to_index, 125)):
        row, column = divmod(index, 9)
        after[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = value

    result = LegalMoveDiffObserver(patch_size=CELL).observe(
        board,
        before,
        after,
        _geometry(),
    )

    candidate = result.evidence.candidates[0]
    assert candidate.semantic_margin == pytest.approx(5 / 255, abs=1e-6)
    assert candidate.destination_semantic_evidence_score > 0.8
    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert "semantic_margin" in result.evidence.rejection_reasons


def test_observer_accepts_same_side_piece_appearance_variation_at_destination() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    after = _render(apply_move(board, move))
    destination_row, destination_column = divmod(move.to_index, 9)
    after[
        destination_row * CELL : (destination_row + 1) * CELL,
        destination_column * CELL : (destination_column + 1) * CELL,
        :3,
    ] = PALETTE["P"]

    result = LegalMoveDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        after,
        _geometry(),
    )

    assert result.status is ObservationStatus.ACCEPTED
    assert result.move == move
    assert not hasattr(result, "after")


def test_observer_rejects_opposite_side_appearance_at_destination() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    before = _render(board)
    after = before.copy()
    for index, value in ((move.from_index, PALETTE["."]), (move.to_index, PALETTE["k"])):
        row, column = divmod(index, 9)
        after[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = value

    result = LegalMoveDiffObserver(patch_size=CELL).observe(
        board,
        before,
        after,
        _geometry(),
    )

    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert result.evidence.rejection_reasons


def test_observer_applies_the_configured_evidence_score_boundary() -> None:
    board = parse_fen(START)
    move = _move(board, "h2e2")
    before = _render(board)
    after = before.copy()
    for index, value in ((move.from_index, PALETTE["."]), (move.to_index, 126)):
        row, column = divmod(index, 9)
        after[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = value

    accepted = LegalMoveDiffObserver(
        patch_size=CELL,
        min_semantic_margin=0.01,
        min_evidence_score=0.8,
    ).observe(board, before, after, _geometry())
    rejected = LegalMoveDiffObserver(
        patch_size=CELL,
        min_semantic_margin=0.01,
        min_evidence_score=0.82,
    ).observe(board, before, after, _geometry())

    assert accepted.status is ObservationStatus.ACCEPTED
    assert 0.8 < accepted.evidence_score < 0.82
    assert rejected.status is ObservationStatus.AMBIGUOUS


def test_observer_pauses_when_two_legal_moves_appear_between_frames() -> None:
    board = parse_fen(START)
    before = _render(board)
    after_pieces = list(board.pieces)
    for uci in ("b2b3", "h2h3"):
        move = _move(board, uci)
        after_pieces[move.to_index] = after_pieces[move.from_index]
        after_pieces[move.from_index] = "."
    after = _render(BoardState(tuple(after_pieces), side_to_move="w"))

    result = LegalMoveDiffObserver(patch_size=CELL).observe(board, before, after, _geometry())

    assert result.status is ObservationStatus.AMBIGUOUS
    assert result.move is None
    assert result.evidence.rejection_reasons


def test_observer_reports_no_change_without_inventing_a_move() -> None:
    board = parse_fen(START)
    frame = _render(board)

    result = LegalMoveDiffObserver(patch_size=CELL).observe(board, frame, frame.copy(), _geometry())

    assert result.status is ObservationStatus.NO_CHANGE
    assert result.move is None
    assert result.evidence.candidates == ()


@pytest.mark.parametrize("threshold", [0.0, float("nan"), float("inf")])
def test_observer_rejects_non_positive_or_non_finite_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        LegalMoveDiffObserver(min_score=threshold)
