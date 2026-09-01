from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.unit.diagnostics.test_stage_c_samples import (
    FINAL_ID,
    START_ID,
    _candidate,
    _crops,
    _rejected_sample,
    _sample,
)
from xiangqi_agent.diagnostics.stage_c_replay import (
    HumanAiStageCReplayer,
    HumanAiStageCSampleLoader,
    StageCSampleIntegrityError,
)
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import ReviewedStageCSampleV2
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleRecorder,
    HumanAiStageCSampleV1,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.sync.sequence_gate import (
    SequenceDecisionGate,
    SequenceThresholdProfile,
)

PROFILE = SequenceThresholdProfile(
    min_local_difference=5.0,
    max_unexpected_difference=3.0,
    min_score=5.0,
    min_margin=5.0,
    max_template_distance=0.18,
    min_template_margin=0.02,
    min_template_confidence=0.8,
    profile_version="human-ai-two-ply-v1",
)


def _replayer(
    *,
    profile: SequenceThresholdProfile = PROFILE,
    feature_version: str = "two-ply-template-v1",
) -> HumanAiStageCReplayer:
    return HumanAiStageCReplayer(
        SequenceDecisionGate(profile),
        feature_version=feature_version,
    )


def _record(tmp_path: Path, *, sample=None, crops=None) -> Path:
    value = sample or _sample()
    point_crops = crops or _crops(value.changed_points)
    return HumanAiStageCSampleRecorder(tmp_path, enabled=True).record(value, point_crops)


def _rewrite_manifest(sample_dir: Path, **changes: object) -> None:
    path = sample_dir / "manifest.json"
    payload = json.loads(path.read_text("utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")


def test_same_frozen_event_replays_to_the_same_decision(tmp_path: Path) -> None:
    first_path = _record(tmp_path, sample=_sample(sample_id="first"))
    second_path = _record(tmp_path, sample=_sample(sample_id="second"))

    first = _replayer().replay(first_path)
    second = _replayer().replay(second_path)

    assert first.without_runtime_and_identity() == second.without_runtime_and_identity()
    assert first.correct_accept
    assert not first.false_accept
    assert first.replayed_moves_uci == ("h2e2", "h7e7")
    assert first.replayed_final_position_id == FINAL_ID


def test_wrong_recorded_observed_final_id_is_an_integrity_error(tmp_path: Path) -> None:
    path = _record(
        tmp_path,
        sample=_sample(observed_final_position_id="f" * 32),
    )

    with pytest.raises(StageCSampleIntegrityError, match="observed final position"):
        _replayer().replay(path)


def test_changed_crop_hash_is_an_integrity_error(tmp_path: Path) -> None:
    path = _record(tmp_path)
    (path / "point-22-before.png").write_bytes(b"changed")

    with pytest.raises(StageCSampleIntegrityError, match="hash mismatch"):
        _replayer().replay(path)


def test_extra_file_is_an_integrity_error(tmp_path: Path) -> None:
    path = _record(tmp_path)
    (path / "unexpected.txt").write_text("unexpected", "utf-8")

    with pytest.raises(StageCSampleIntegrityError, match="exactly the declared crops"):
        _replayer().replay(path)


def test_confirmed_fen_and_position_id_must_match(tmp_path: Path) -> None:
    path = _record(tmp_path)
    _rewrite_manifest(path, confirmed_position_id="f" * 32)

    with pytest.raises(StageCSampleIntegrityError, match="confirmed FEN"):
        _replayer().replay(path)


def test_serialized_candidate_must_be_a_real_legal_two_ply_chain(tmp_path: Path) -> None:
    illegal = _candidate(
        moves_uci=("a0a9", "a9a8"),
        final_position_id="e" * 32,
    )
    path = _record(tmp_path, sample=_sample(candidates=(illegal,)))

    with pytest.raises(StageCSampleIntegrityError, match="legal two-ply"):
        _replayer().replay(path)


def test_candidate_final_position_id_is_recomputed_from_rules(tmp_path: Path) -> None:
    path = _record(
        tmp_path,
        sample=_sample(candidates=(_candidate(final_position_id="e" * 32),)),
    )

    with pytest.raises(StageCSampleIntegrityError, match="candidate final position"):
        _replayer().replay(path)


def test_replay_requires_the_frozen_feature_and_threshold_versions(tmp_path: Path) -> None:
    wrong_feature = _record(
        tmp_path / "feature",
        sample=replace(_sample(), feature_version="two-ply-template-v2"),
    )
    wrong_profile = _record(
        tmp_path / "profile",
        sample=replace(_sample(), threshold_profile_version="other-profile"),
    )

    with pytest.raises(StageCSampleIntegrityError, match="feature version"):
        _replayer().replay(wrong_feature)
    with pytest.raises(StageCSampleIntegrityError, match="threshold profile"):
        _replayer().replay(wrong_profile)


def test_rejected_replay_never_returns_moves_or_a_new_position(tmp_path: Path) -> None:
    path = _record(tmp_path, sample=_rejected_sample(), crops=_crops((64,)))

    result = _replayer().replay(path)

    assert result.correct_reject
    assert not result.accepted
    assert not result.false_accept
    assert result.replayed_moves_uci == ()
    assert result.replayed_final_position_id is None
    assert result.recorded_observation_matches_replay


def test_wrong_accepted_chain_is_counted_as_a_false_accept(tmp_path: Path) -> None:
    wrong = _candidate(
        moves_uci=("b2b3", "b7b6"),
        final_position_id="82dab2674504ae6edc4a430afef9277e",
    )
    wrong = replace(wrong, changed_points=(19, 28, 55, 64))
    path = _record(
        tmp_path,
        sample=_rejected_sample(
            observed_status=StageCObservedStatus.ACCEPTED,
            observed_moves_uci=("b2b3", "b7b6"),
            observed_final_position_id="82dab2674504ae6edc4a430afef9277e",
            candidates=(wrong,),
            rejection_reasons=(),
        ),
        crops=_crops((64,)),
    )

    result = _replayer().replay(path)

    assert result.accepted
    assert result.false_accept
    assert not result.correct_reject
    assert result.replayed_final_position_id != START_ID


def test_loader_and_replayer_accept_equivalent_v1_and_reviewed_v2(
    tmp_path: Path,
) -> None:
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _record as record_event,
    )
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _review_valid,
        _reviewed_root,
    )
    from tests.unit.diagnostics.test_stage_c_quarantine import _event
    from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService

    v1 = _record(tmp_path / "legacy")
    event_dir = record_event(tmp_path / "reviewed", _event())
    review_path = _review_valid(tmp_path / "reviewed", event_dir)
    v2 = StageCPromotionService().promote(
        event_dir,
        review_path,
        _reviewed_root(tmp_path / "reviewed"),
    )

    legacy_loaded = HumanAiStageCSampleLoader().load(v1)
    reviewed_loaded = HumanAiStageCSampleLoader().load(v2)
    legacy_result = _replayer().replay(v1)
    reviewed_result = _replayer().replay(v2)

    assert isinstance(legacy_loaded.metadata, HumanAiStageCSampleV1)
    assert isinstance(reviewed_loaded.metadata, ReviewedStageCSampleV2)
    assert (
        legacy_result.without_runtime_and_identity()
        == reviewed_result.without_runtime_and_identity()
    )
    assert reviewed_result.review_outcome == "candidate_confirmed"
    assert legacy_result.review_outcome is None


@pytest.mark.parametrize(
    "filename",
    ("source-event-manifest.json", "review-manifest.json"),
)
def test_v2_provenance_sidecar_mutation_fails_before_replay(
    tmp_path: Path,
    filename: str,
) -> None:
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _record as record_event,
    )
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _review_valid,
        _reviewed_root,
    )
    from tests.unit.diagnostics.test_stage_c_quarantine import _event
    from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService

    event_dir = record_event(tmp_path, _event())
    review_path = _review_valid(tmp_path, event_dir)
    sample_dir = StageCPromotionService().promote(
        event_dir,
        review_path,
        _reviewed_root(tmp_path),
    )
    (sample_dir / filename).write_bytes((sample_dir / filename).read_bytes() + b" ")

    with pytest.raises(StageCSampleIntegrityError, match="provenance"):
        _replayer().replay(sample_dir)


def test_loader_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = _record(tmp_path)
    _rewrite_manifest(path, schema_version=99)

    with pytest.raises(StageCSampleIntegrityError, match="schema version"):
        HumanAiStageCSampleLoader().load(path)


def test_reviewed_resize_terminal_reason_is_consistent_with_safe_replay(
    tmp_path: Path,
) -> None:
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _record as record_event,
    )
    from tests.unit.diagnostics.test_stage_c_promotion import (
        _rejection_event,
        _review_rejection,
        _reviewed_root,
    )
    from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService

    event, moves = _rejection_event(StageCScenario.RESIZE)
    event_dir = record_event(tmp_path, event)
    review_path = _review_rejection(
        tmp_path,
        event_dir,
        StageCScenario.RESIZE,
        moves,
    )
    sample_dir = StageCPromotionService().promote(
        event_dir,
        review_path,
        _reviewed_root(tmp_path),
    )

    result = _replayer().replay(sample_dir)

    assert result.correct_reject
    assert result.recorded_observation_matches_replay
