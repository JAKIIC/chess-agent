from time import perf_counter
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter
from xiangqi_agent.sync.evidence import ObservationStatus
from xiangqi_agent.sync.move_observer import LegalMoveDiffObserver
from xiangqi_agent.sync.sequence_observer import LegalTwoPlyDiffObserver
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.templates import PieceTemplateBank, PieceTemplateBankCache

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


def _paint_cell(frame: np.ndarray, index: int, value: int) -> None:
    row, column = divmod(index, 9)
    frame[
        row * CELL : (row + 1) * CELL,
        column * CELL : (column + 1) * CELL,
        :3,
    ] = value


def _paint_move_marker(frame: np.ndarray, index: int) -> None:
    row, column = divmod(index, 9)
    center = (column * CELL + CELL // 2, row * CELL + CELL // 2)
    cv2.circle(frame, center, 7, (255, 255, 255, 255), thickness=1)


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
    assert proposal.evidence.feature_version == "two-ply-template-transfer-v5"


def test_two_ply_observer_only_scores_replies_after_a_confirmed_first_move() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe_after_first(
        board,
        first,
        _render(board),
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    assert proposal.evidence.feature_version == "two-ply-template-transfer-v5"
    assert all(
        candidate.moves[0] == first for candidate in proposal.evidence.candidates
    )


def test_two_ply_observer_only_expands_first_moves_from_changed_sources() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    committer = Mock(wraps=RuleStateCommitter())

    proposal = LegalTwoPlyDiffObserver(
        patch_size=CELL,
        committer=committer,
    ).observe(
        board,
        _render(board),
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    committer.project_two_ply.assert_not_called()
    expanded_first_moves = tuple(
        call.args[1] for call in committer.project_replies.call_args_list
    )
    assert expanded_first_moves
    assert {move.from_index for move in expanded_first_moves} == {first.from_index}


def test_two_ply_observer_classifies_each_after_patch_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    real_classify = PieceTemplateBank.classify
    calls_by_patch: dict[int, int] = {}

    def counted_classify(self: PieceTemplateBank, patch: np.ndarray):
        key = id(patch)
        calls_by_patch[key] = calls_by_patch.get(key, 0) + 1
        return real_classify(self, patch)

    monkeypatch.setattr(PieceTemplateBank, "classify", counted_classify)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert calls_by_patch
    assert max(calls_by_patch.values()) == 1


def test_single_and_two_ply_observers_share_one_confirmed_template_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    before = _render(board)
    cache = PieceTemplateBankCache()
    real_from_position = PieceTemplateBank.from_position.__func__
    calls = 0

    def counted_from_position(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_from_position(cls, *args, **kwargs)

    monkeypatch.setattr(
        PieceTemplateBank,
        "from_position",
        classmethod(counted_from_position),
    )
    single = LegalMoveDiffObserver(patch_size=CELL, template_cache=cache)
    sequence = LegalTwoPlyDiffObserver(patch_size=CELL, template_cache=cache)

    first_proposal = single.observe(
        board,
        before,
        _render(middle),
        _geometry(),
    )
    sequence_proposal = sequence.observe_after_first(
        board,
        first,
        before,
        _render(final),
        _geometry(),
    )

    assert first_proposal.status is ObservationStatus.ACCEPTED
    assert sequence_proposal.status is ObservationStatus.ACCEPTED
    assert calls == 1


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


def test_two_ply_observer_tolerates_weak_highlight_artifacts_that_keep_semantics() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    frame = _render(final)
    # Model fixed-theme last-move tint: the final pieces resemble another piece
    # of the same side, so exact-symbol margin falls while side/occupancy remains.
    _paint_cell(frame, 22, PALETTE["r"])
    _paint_cell(frame, 67, PALETTE["R"])
    # Model weaker shadow/highlight spill on unchanged empty intersections.
    _paint_cell(frame, 16, 21)
    _paint_cell(frame, 78, 21)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        frame,
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    # Semantically verified highlight spill is not an unrelated board change.
    assert proposal.evidence.candidates[0].unexpected_difference == 0.0


def test_two_ply_observer_ignores_disappearing_prior_highlights_that_keep_semantics() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    before = _render(board)
    # The confirmed baseline can still contain the previous move's tint.  Both
    # affected intersections remain occupied by the same red/black piece class
    # after the next two plies, so their visual reset is not a board change.
    _paint_cell(before, 81, PALETTE["K"])
    _paint_cell(before, 8, PALETTE["k"])

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        before,
        _render(final),
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)


def test_two_ply_observer_uses_piece_transfer_when_empty_sources_keep_move_markers() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    after = _render(final)
    _paint_move_marker(after, first.from_index)
    _paint_move_marker(after, second.from_index)

    proposal = LegalTwoPlyDiffObserver(
        patch_size=CELL,
        min_template_confidence=0.999,
    ).observe(
        board,
        _render(board),
        after,
        _geometry(),
    )

    assert proposal.status is ObservationStatus.ACCEPTED
    assert proposal.moves == (first, second)
    assert proposal.evidence.feature_version == "two-ply-template-transfer-v5"


def test_piece_transfer_cannot_turn_move_markers_without_piece_motion_into_a_move() -> None:
    board = parse_fen(START)
    after = _render(board)
    candidate = _move(board, "h2e2")
    _paint_move_marker(after, candidate.from_index)
    _paint_move_marker(after, candidate.to_index)

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        after,
        _geometry(),
    )

    assert proposal.status is not ObservationStatus.ACCEPTED
    assert proposal.moves == ()


def test_two_ply_observer_rejects_a_weak_outside_semantic_change() -> None:
    board = parse_fen(START)
    first = _move(board, "h2e2")
    middle = apply_move(board, first)
    second = _move(middle, "h7e7")
    final = apply_move(middle, second)
    frame = _render(final)
    _paint_cell(frame, 22, PALETTE["r"])
    _paint_cell(frame, 67, PALETTE["R"])
    # This is weak enough for the artifact ratio, but changes empty -> occupied.
    _paint_cell(frame, 16, PALETTE["K"])

    proposal = LegalTwoPlyDiffObserver(patch_size=CELL).observe(
        board,
        _render(board),
        frame,
        _geometry(),
    )

    assert proposal.status is ObservationStatus.AMBIGUOUS
    assert proposal.moves == ()
    assert "template_confidence" in proposal.evidence.rejection_reasons


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
