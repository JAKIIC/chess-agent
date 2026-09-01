from time import perf_counter

import numpy as np

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.evidence import ObservationStatus
from xiangqi_agent.sync.sequence_observer import LegalTwoPlyDiffObserver
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
        frame[
            row * CELL : (row + 1) * CELL,
            column * CELL : (column + 1) * CELL,
            :3,
        ] = value
    return frame


def _move(board: BoardState, uci: str) -> Move:
    return next(move for move in legal_moves(board) if move.uci == uci)


def test_two_ply_observer_accepts_the_only_legal_chain_matching_final_frame() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    assert proposal.evidence.candidates[0].final_position_id == final.position_id


def test_two_ply_observer_reports_no_change_for_the_confirmed_frame() -> None:
    board = parse_fen(START)
    frame = _render(board)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        frame,
        frame.copy(),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.NO_CHANGE
    assert proposal.moves == ()


def test_two_ply_observer_handles_same_symbol_recapture_with_two_changed_points() -> None:
    board = parse_fen("r3k4/9/9/9/r3p4/9/9/9/9/R3K4 w")
    first = _move(board, "a0a5")
    middle = apply_move(board, first)
    second = _move(middle, "a9a5")
    final = apply_move(middle, second)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    assert proposal.evidence.candidates[0].changed_points == (0, 81)


def test_two_ply_observer_rejects_a_three_ply_final_frame() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    after_second = apply_move(middle, second)
    third = _move(after_second, "b2b3")
    after_third = apply_move(after_second, third)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        _render(after_third),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.AMBIGUOUS
    assert proposal.moves == ()


def test_two_ply_observer_rejects_an_unrelated_strong_change() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    frame = _render(final)
    unrelated_index = 40
    row, column = divmod(unrelated_index, 9)
    frame[
        row * CELL : (row + 1) * CELL,
        column * CELL : (column + 1) * CELL,
        :3,
    ] = 255

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        frame,
        _geometry(),
    )

    assert proposal.status is ObservationStatus.AMBIGUOUS
    assert proposal.moves == ()
    assert "outside_change" in proposal.evidence.rejection_reasons


def test_two_ply_observer_meets_the_stable_frame_decision_budget() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    observer = LegalTwoPlyDiffObserver(patch_size=CELL)

    started = perf_counter()
    proposal = observer.observe(board, _render(board), _render(final), _geometry())
    elapsed_ms = (perf_counter() - started) * 1000

    assert proposal.status is ObservationStatus.ACCEPTED
    assert elapsed_ms < 500
