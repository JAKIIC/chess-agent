from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    QuarantinedStageCEventV1,
    QuarantineEventRecorder,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewIntegrityError,
    StageCReviewOutcome,
    StageCReviewService,
    StageCReviewStore,
    StageCReviewV1,
    legal_review_choices,
    load_stage_c_review,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.vision.occupancy import OccupancyEvidence

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
START = parse_fen(START_FEN)


def test_review_schema_enforces_label_specific_shapes() -> None:
    valid = _review()
    assert valid.label_kind is StageCLabelKind.VALID_TWO_PLY

    with pytest.raises(ValueError, match="exactly two"):
        replace(valid, moves_uci=("h2e2",))
    with pytest.raises(ValueError, match="no rejection scenario"):
        replace(valid, scenario=StageCScenario.OCCLUSION)
    with pytest.raises(ValueError, match="candidate confirmation or legal correction"):
        replace(valid, review_outcome=StageCReviewOutcome.EXPECTED_REJECTION)

    rejected = _review(
        review_id="review-rejected",
        label_kind=StageCLabelKind.EXPECTED_REJECTION,
        moves=(),
        final_id=None,
        scenario=StageCScenario.OCCLUSION,
        outcome=StageCReviewOutcome.EXPECTED_REJECTION,
    )
    with pytest.raises(ValueError, match="rejection scenario"):
        replace(rejected, scenario=StageCScenario.VALID_TWO_PLY)
    with pytest.raises(ValueError, match="must not contain a final"):
        replace(rejected, expected_final_position_id="0" * 32)
    with pytest.raises(ValueError, match="at most three"):
        replace(rejected, moves_uci=("h2e2", "h7e7", "e2e3", "a0a1"))

    discarded = _review(
        review_id="review-discarded",
        label_kind=StageCLabelKind.DISCARD,
        moves=(),
        final_id=None,
        scenario=None,
        outcome=StageCReviewOutcome.DISCARDED,
    )
    with pytest.raises(ValueError, match="discard"):
        replace(discarded, moves_uci=("h2e2",))


def test_review_schema_rejects_private_or_unsafe_values() -> None:
    with pytest.raises(ValueError, match="path-safe"):
        replace(_review(), review_id="../review")
    with pytest.raises(ValueError, match="64 lowercase"):
        replace(_review(), event_manifest_sha256="ABC")
    with pytest.raises(ValueError, match="UTC timestamp"):
        replace(_review(), created_at_utc="2026-09-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="four-character"):
        replace(_review(), moves_uci=("炮二平五", "h7e7"))
    with pytest.raises(ValueError, match="local_user"):
        replace(_review(), reviewer_kind="lenovo")


def test_review_draft_validates_combinations_before_storage() -> None:
    StageCReviewDraft(
        label_kind=StageCLabelKind.VALID_TWO_PLY,
        moves_uci=("h2e2", "h7e7"),
        scenario=None,
        review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
    )
    with pytest.raises(ValueError, match="discard"):
        StageCReviewDraft(
            label_kind=StageCLabelKind.DISCARD,
            moves_uci=(),
            scenario=StageCScenario.OCCLUSION,
            review_outcome=StageCReviewOutcome.DISCARDED,
        )


@pytest.mark.parametrize(
    "scenario",
    tuple(item for item in StageCScenario if item is not StageCScenario.VALID_TWO_PLY),
)
def test_review_draft_accepts_each_frozen_rejection_scenario(
    scenario: StageCScenario,
) -> None:
    draft = StageCReviewDraft(
        label_kind=StageCLabelKind.EXPECTED_REJECTION,
        moves_uci=(),
        scenario=scenario,
        review_outcome=StageCReviewOutcome.EXPECTED_REJECTION,
    )
    assert draft.scenario is scenario


def test_legal_review_choices_are_complete_sorted_and_rule_projected() -> None:
    choices = legal_review_choices(START)
    expected_moves = legal_moves(START)

    assert tuple((choice.chinese, choice.uci) for choice in choices) == tuple(
        sorted((choice.chinese, choice.uci) for choice in choices)
    )
    assert {choice.uci for choice in choices} == {move.uci for move in expected_moves}
    by_uci = {move.uci: move for move in expected_moves}
    assert all(
        choice.resulting_position_id
        == apply_move(START, by_uci[choice.uci]).position_id
        for choice in choices
    )

    first = next(choice for choice in choices if choice.uci == "h2e2")
    middle = apply_move(START, by_uci[first.uci])
    replies = legal_review_choices(middle)
    assert {choice.uci for choice in replies} == {
        move.uci for move in legal_moves(middle)
    }


def test_review_service_derives_hash_and_final_position_from_legal_chain(
    tmp_path: Path,
) -> None:
    event_dir = _record_event(tmp_path)
    store = StageCReviewStore(_review_root(tmp_path), enabled=True)
    path = StageCReviewService(store).submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.VALID_TWO_PLY,
            moves_uci=("h2e2", "h7e7"),
            scenario=None,
            review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
        ),
    )

    loaded = load_stage_c_review(path)
    assert loaded.event_manifest_sha256 == _sha256(event_dir / "manifest.json")
    assert loaded.expected_final_position_id == _two_ply_final(START).position_id
    assert loaded.event_id == "event-1"
    assert loaded.session_id == "session-1"
    assert store.active_review("session-1", "event-1") == loaded


def test_review_service_rejects_illegal_sequence_and_false_confirmation(
    tmp_path: Path,
) -> None:
    event_dir = _record_event(tmp_path)
    service = StageCReviewService(
        StageCReviewStore(_review_root(tmp_path), enabled=True)
    )
    with pytest.raises(ValueError, match="sequentially legal"):
        service.submit(
            event_dir,
            StageCReviewDraft(
                label_kind=StageCLabelKind.VALID_TWO_PLY,
                moves_uci=("a0a9", "h7e7"),
                scenario=None,
                review_outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
            ),
        )
    with pytest.raises(ValueError, match="prefilled candidate"):
        service.submit(
            event_dir,
            StageCReviewDraft(
                label_kind=StageCLabelKind.VALID_TWO_PLY,
                moves_uci=("b2b3", "b7b6"),
                scenario=None,
                review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
            ),
        )
    assert not _review_root(tmp_path).exists()


def test_review_service_accepts_legal_correction_and_rejection_prefix(
    tmp_path: Path,
) -> None:
    event_dir = _record_event(tmp_path)
    service = StageCReviewService(
        StageCReviewStore(_review_root(tmp_path), enabled=True)
    )
    correction = service.submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.VALID_TWO_PLY,
            moves_uci=("b2b3", "b7b6"),
            scenario=None,
            review_outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
        ),
    )
    active = load_stage_c_review(correction)

    rejection = service.submit(
        event_dir,
        StageCReviewDraft(
            label_kind=StageCLabelKind.EXPECTED_REJECTION,
            moves_uci=("h2e2", "h7e7"),
            scenario=StageCScenario.CONTINUOUS_ANIMATION,
            review_outcome=StageCReviewOutcome.EXPECTED_REJECTION,
            supersedes_review_id=active.review_id,
        ),
    )
    loaded = load_stage_c_review(rejection)
    assert loaded.expected_final_position_id is None
    assert loaded.supersedes_review_id == active.review_id


def test_review_store_is_disabled_by_default_and_enforces_quota(
    tmp_path: Path,
) -> None:
    with pytest.raises(DiagnosticsDisabledError, match="explicitly enabled"):
        StageCReviewStore(_review_root(tmp_path)).submit(_review())
    with pytest.raises(SampleQuotaExceededError, match="capacity"):
        StageCReviewStore(
            _review_root(tmp_path), enabled=True, max_bytes=10
        ).submit(_review())
    assert not _review_root(tmp_path).exists()


def test_review_store_never_overwrites_and_requires_linear_supersession(
    tmp_path: Path,
) -> None:
    store = StageCReviewStore(_review_root(tmp_path), enabled=True)
    first = _review()
    first_path = store.submit(first)
    original = first_path.read_bytes()

    with pytest.raises(FileExistsError):
        store.submit(first)
    with pytest.raises(StageCReviewIntegrityError, match="active review"):
        store.submit(replace(first, review_id="parallel-review"))
    with pytest.raises(StageCReviewIntegrityError, match="active review"):
        store.submit(
            replace(
                first,
                review_id="unknown-parent",
                supersedes_review_id="missing-review",
            )
        )

    second = replace(
        first,
        review_id="review-2",
        created_at_utc="2026-09-01T00:01:00Z",
        supersedes_review_id=first.review_id,
    )
    second_path = store.submit(second)
    assert first_path.read_bytes() == original
    assert first_path.exists() and second_path.exists()
    assert store.active_review("session-1", "event-1") == second


def test_review_store_detects_tampered_multiple_branches_and_cycles(
    tmp_path: Path,
) -> None:
    root = _review_root(tmp_path)
    store = StageCReviewStore(root, enabled=True)
    first = _review()
    store.submit(first)
    event_dir = root / "session-1" / "event-1"
    _write_review(event_dir / "parallel.json", replace(first, review_id="parallel"))

    with pytest.raises(StageCReviewIntegrityError, match="multiple active"):
        store.active_review("session-1", "event-1")

    cycle_dir = root / "session-cycle" / "event-cycle"
    left = replace(
        first,
        review_id="left",
        session_id="session-cycle",
        event_id="event-cycle",
        supersedes_review_id="right",
    )
    right = replace(left, review_id="right", supersedes_review_id="left")
    _write_review(cycle_dir / "left.json", left)
    _write_review(cycle_dir / "right.json", right)
    with pytest.raises(StageCReviewIntegrityError, match="cycle"):
        store.active_review("session-cycle", "event-cycle")


def test_review_loader_rejects_extra_fields_and_filename_mismatch(
    tmp_path: Path,
) -> None:
    path = StageCReviewStore(_review_root(tmp_path), enabled=True).submit(_review())
    payload = json.loads(path.read_text("utf-8"))
    payload["window_title"] = "private"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StageCReviewIntegrityError, match="unexpected"):
        load_stage_c_review(path)

    other = path.with_name("wrong-name.json")
    _write_review(other, replace(_review(), review_id="another-id"))
    other.rename(path.with_name("mismatch.json"))
    with pytest.raises(StageCReviewIntegrityError, match="filename"):
        load_stage_c_review(path.with_name("mismatch.json"))


def test_review_store_rejects_symlinked_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "stage-c-reviews"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")
    with pytest.raises(StageCReviewIntegrityError, match="symlink"):
        StageCReviewStore(alias, enabled=True).submit(_review())


def _review(
    *,
    review_id: str = "review-1",
    label_kind: StageCLabelKind = StageCLabelKind.VALID_TWO_PLY,
    moves: tuple[str, ...] = ("h2e2", "h7e7"),
    final_id: str | None = None,
    scenario: StageCScenario | None = None,
    outcome: StageCReviewOutcome = StageCReviewOutcome.CANDIDATE_CONFIRMED,
) -> StageCReviewV1:
    return StageCReviewV1(
        review_id=review_id,
        event_id="event-1",
        session_id="session-1",
        created_at_utc="2026-09-01T00:00:00Z",
        event_manifest_sha256="a" * 64,
        label_kind=label_kind,
        moves_uci=moves,
        expected_final_position_id=(
            _two_ply_final(START).position_id if final_id is None and moves else final_id
        ),
        scenario=scenario,
        review_outcome=outcome,
        supersedes_review_id=None,
    )


def _record_event(tmp_path: Path) -> Path:
    return QuarantineEventRecorder(_quarantine_root(tmp_path), enabled=True).record(
        _event(), _crops()
    )


def _event() -> QuarantinedStageCEventV1:
    final = _two_ply_final(START)
    return QuarantinedStageCEventV1(
        event_id="event-1",
        session_id="session-1",
        created_at_utc="2026-09-01T00:00:00Z",
        confirmed_fen=START_FEN,
        confirmed_position_id=START.position_id,
        observed_status=StageCObservedStatus.ACCEPTED,
        observed_moves_uci=("h2e2", "h7e7"),
        observed_final_position_id=final.position_id,
        side_to_move="w",
        orientation=Orientation.RED_BOTTOM,
        changed_points=(22, 25, 67, 70),
        local_differences=tuple(float(index + 1) / 10 for index in range(90)),
        candidates=(_candidate(final.position_id),),
        rejection_reasons=(),
        capture_context=CaptureContext(
            wgc_size=(216, 240),
            client_size=(216, 240),
            dpi_scale=1.0,
            geometry_revision="quad-v1",
            theme_fingerprint="theme-v1",
            generation_id=1,
        ),
        feature_version="two-ply-template-v1",
        threshold_profile_version="human-ai-two-ply-v1",
        decision_latency_ms=100.0,
        before_occupancy=_occupancy(START),
        after_occupancy=_occupancy(final),
    )


def _candidate(final_id: str) -> StageCCandidateRecord:
    return StageCCandidateRecord(
        moves_uci=("h2e2", "h7e7"),
        changed_points=(22, 25, 67, 70),
        expected_change_floor=20.0,
        unexpected_difference=1.0,
        maximum_template_distance=0.05,
        minimum_template_margin=0.1,
        minimum_template_confidence=0.9,
        score=20.0,
        final_position_id=final_id,
    )


def _crops() -> tuple[TransitionPointCrops, ...]:
    return tuple(
        TransitionPointCrops(
            point,
            np.full((48, 48, 4), (20, 20, 20, 255), dtype=np.uint8),
            np.full((48, 48, 4), (60, 60, 60, 255), dtype=np.uint8),
        )
        for point in (22, 25, 67, 70)
    )


def _occupancy(board: BoardState) -> OccupancyEvidence:
    return OccupancyEvidence(
        tuple(piece != "." for piece in board.pieces),
        (0.95,) * 90,
        "circular-occupancy-v1",
    )


def _two_ply_final(board: BoardState) -> BoardState:
    first = next(move for move in legal_moves(board) if move.uci == "h2e2")
    middle = apply_move(board, first)
    second = next(move for move in legal_moves(middle) if move.uci == "h7e7")
    return apply_move(middle, second)


def _write_review(path: Path, review: StageCReviewV1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(review)
    payload["label_kind"] = review.label_kind.value
    payload["scenario"] = None if review.scenario is None else review.scenario.value
    payload["review_outcome"] = review.review_outcome.value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


def _local_root(tmp_path: Path) -> Path:
    return tmp_path / ".local"


def _quarantine_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-quarantine"


def _review_root(tmp_path: Path) -> Path:
    return _local_root(tmp_path) / "stage-c-reviews"
