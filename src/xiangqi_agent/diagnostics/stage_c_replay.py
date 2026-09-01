from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any, cast

import cv2
import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCReviewOutcome,
)
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import (
    ReviewedStageCSampleIntegrityError,
    ReviewedStageCSampleLoader,
    ReviewedStageCSampleV2,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    HumanAiStageCSampleV1,
    StageCCandidateRecord,
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState, Move, Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.committer import RuleStateCommitter, StateCommitter
from xiangqi_agent.sync.evidence import SequenceCandidateEvidence
from xiangqi_agent.sync.sequence_gate import SequenceDecisionGate

_CAPTURE_TERMINAL_REASONS = frozenset(
    {"frame_size_changed", "capture_context_invalid", "target_window_closed"}
)


class StageCSampleIntegrityError(ValueError):
    """A frozen Stage C sample is incomplete, changed, or contradictory."""


type StageCSampleMetadata = HumanAiStageCSampleV1 | ReviewedStageCSampleV2


@dataclass(frozen=True, slots=True)
class LoadedHumanAiStageCSample:
    metadata: StageCSampleMetadata
    crops: tuple[TransitionPointCrops, ...]
    directory: Path


@dataclass(frozen=True, slots=True)
class HumanAiStageCReplayResult:
    sample_id: str
    session_id: str
    scenario: StageCScenario
    expected_outcome: StageCExpectedOutcome
    ground_truth_moves_uci: tuple[str, ...]
    accepted: bool
    replayed_moves_uci: tuple[str, ...]
    replayed_final_position_id: str | None
    rejection_reasons: tuple[str, ...]
    correct_accept: bool
    false_accept: bool
    correct_reject: bool
    missed_valid: bool
    recorded_observation_matches_replay: bool
    decision_latency_ms: float
    feature_version: str
    threshold_profile_version: str
    runtime_ns: int
    review_outcome: StageCReviewOutcome | None = None
    label_source: str | None = None

    def without_runtime_and_identity(self) -> tuple[object, ...]:
        return (
            self.scenario,
            self.expected_outcome,
            self.ground_truth_moves_uci,
            self.accepted,
            self.replayed_moves_uci,
            self.replayed_final_position_id,
            self.rejection_reasons,
            self.correct_accept,
            self.false_accept,
            self.correct_reject,
            self.missed_valid,
            self.recorded_observation_matches_replay,
            self.decision_latency_ms,
            self.feature_version,
            self.threshold_profile_version,
        )


class HumanAiStageCSampleLoader:
    def load(self, sample_dir: Path) -> LoadedHumanAiStageCSample:
        if not isinstance(sample_dir, Path):
            raise TypeError("sample directory must be a Path")
        if not sample_dir.is_dir():
            raise StageCSampleIntegrityError("Stage C sample directory does not exist")

        manifest_path = sample_dir / "manifest.json"
        payload = _read_manifest(manifest_path)
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise StageCSampleIntegrityError(
                "Stage C manifest schema version must be an integer"
            )
        if schema_version == 2:
            try:
                reviewed = ReviewedStageCSampleLoader().load(sample_dir)
            except ReviewedStageCSampleIntegrityError as exc:
                raise StageCSampleIntegrityError(
                    "reviewed V2 provenance or sample integrity failed"
                ) from exc
            return LoadedHumanAiStageCSample(
                reviewed.metadata,
                reviewed.crops,
                reviewed.directory,
            )
        if schema_version != 1:
            raise StageCSampleIntegrityError(
                "Stage C manifest uses an unknown schema version"
            )
        metadata = _metadata_from_payload(payload)
        expected_crop_files = tuple(
            filename
            for point in metadata.changed_points
            for filename in (f"point-{point:02d}-before.png", f"point-{point:02d}-after.png")
        )
        expected_files = frozenset(("manifest.json", *expected_crop_files))
        actual_files = frozenset(path.name for path in sample_dir.iterdir() if path.is_file())
        if actual_files != expected_files:
            raise StageCSampleIntegrityError(
                "sample must contain exactly the declared crops and manifest"
            )

        hashes = _string_mapping(payload.get("crop_hashes"), "crop_hashes")
        if frozenset(hashes) != frozenset(expected_crop_files):
            raise StageCSampleIntegrityError("manifest crop hashes do not match declared points")
        encoded: dict[str, bytes] = {}
        for filename in expected_crop_files:
            contents = (sample_dir / filename).read_bytes()
            if _sha256(contents) != hashes[filename]:
                raise StageCSampleIntegrityError(f"Stage C crop hash mismatch: {filename}")
            encoded[filename] = contents
        crops = tuple(
            TransitionPointCrops(
                point,
                _decode_crop(encoded[f"point-{point:02d}-before.png"], point, "before"),
                _decode_crop(encoded[f"point-{point:02d}-after.png"], point, "after"),
            )
            for point in metadata.changed_points
        )

        board = _board_from_metadata(metadata)
        if board.position_id != metadata.confirmed_position_id:
            raise StageCSampleIntegrityError(
                "confirmed FEN does not match confirmed_position_id"
            )
        expected_board = _project_uci_chain(board, metadata.ground_truth_moves_uci)
        if metadata.ground_truth_moves_uci and expected_board is None:
            raise StageCSampleIntegrityError("ground-truth moves are not a legal sequence")
        if metadata.expected_outcome is StageCExpectedOutcome.ACCEPT and (
            expected_board is None
            or expected_board.position_id != metadata.expected_final_position_id
        ):
            raise StageCSampleIntegrityError(
                "expected final position does not match ground-truth moves"
            )

        if metadata.observed_status is StageCObservedStatus.ACCEPTED:
            observed_board = _project_uci_chain(board, metadata.observed_moves_uci)
            if (
                observed_board is None
                or observed_board.position_id != metadata.observed_final_position_id
            ):
                raise StageCSampleIntegrityError(
                    "observed final position does not match observed moves"
                )
        return LoadedHumanAiStageCSample(metadata, crops, sample_dir)


class HumanAiStageCReplayer:
    def __init__(
        self,
        gate: SequenceDecisionGate,
        *,
        feature_version: str,
        loader: HumanAiStageCSampleLoader | None = None,
        committer: StateCommitter | None = None,
    ) -> None:
        if not isinstance(gate, SequenceDecisionGate):
            raise TypeError("gate must be a SequenceDecisionGate")
        if not isinstance(feature_version, str) or not feature_version.strip():
            raise ValueError("feature_version must be a non-empty string")
        self._gate = gate
        self._feature_version = feature_version
        self._loader = loader or HumanAiStageCSampleLoader()
        self._committer = committer or RuleStateCommitter()

    def replay(self, sample_dir: Path) -> HumanAiStageCReplayResult:
        started_ns = perf_counter_ns()
        loaded = self._loader.load(sample_dir)
        sample = loaded.metadata
        if sample.feature_version != self._feature_version:
            raise StageCSampleIntegrityError("sample feature version is not frozen version")
        if sample.threshold_profile_version != self._gate.profile.profile_version:
            raise StageCSampleIntegrityError("sample threshold profile is not frozen profile")

        board = _board_from_metadata(sample)
        legal_chains = {
            (moves[0].uci, moves[1].uci): (moves, final)
            for moves, final in self._committer.project_two_ply(board)
        }
        candidates = tuple(
            _candidate_from_record(board, record, legal_chains)
            for record in sample.candidates
        )
        decision = self._gate.evaluate(
            candidates,
            template_unavailable="template_unavailable" in sample.rejection_reasons,
        )

        replayed_moves: tuple[str, ...] = ()
        replayed_final_id: str | None = None
        if decision.accepted:
            candidate = decision.candidate
            if candidate is None:
                raise StageCSampleIntegrityError(
                    "accepted replay decision did not expose a candidate"
                )
            final = self._committer.commit_sequence(board, candidate.moves)
            if final.position_id != candidate.final_position_id:
                raise StageCSampleIntegrityError(
                    "candidate final position changed during rule projection"
                )
            replayed_moves = tuple(move.uci for move in candidate.moves)
            replayed_final_id = final.position_id

        correct_accept = (
            sample.expected_outcome is StageCExpectedOutcome.ACCEPT
            and decision.accepted
            and replayed_moves == sample.ground_truth_moves_uci
            and replayed_final_id == sample.expected_final_position_id
        )
        false_accept = decision.accepted and not correct_accept
        correct_reject = (
            sample.expected_outcome is StageCExpectedOutcome.REJECT
            and not decision.accepted
        )
        missed_valid = (
            sample.expected_outcome is StageCExpectedOutcome.ACCEPT and not correct_accept
        )
        recorded_matches = _recorded_observation_matches(
            sample,
            accepted=decision.accepted,
            replayed_moves=replayed_moves,
            replayed_final_id=replayed_final_id,
            rejection_reasons=decision.rejection_reasons,
        )
        return HumanAiStageCReplayResult(
            sample_id=sample.sample_id,
            session_id=sample.session_id,
            scenario=sample.scenario,
            expected_outcome=sample.expected_outcome,
            ground_truth_moves_uci=sample.ground_truth_moves_uci,
            accepted=decision.accepted,
            replayed_moves_uci=replayed_moves,
            replayed_final_position_id=replayed_final_id,
            rejection_reasons=decision.rejection_reasons,
            correct_accept=correct_accept,
            false_accept=false_accept,
            correct_reject=correct_reject,
            missed_valid=missed_valid,
            recorded_observation_matches_replay=recorded_matches,
            decision_latency_ms=sample.decision_latency_ms,
            feature_version=sample.feature_version,
            threshold_profile_version=sample.threshold_profile_version,
            runtime_ns=perf_counter_ns() - started_ns,
            review_outcome=(
                sample.review_outcome
                if isinstance(sample, ReviewedStageCSampleV2)
                else None
            ),
            label_source=(
                sample.label_source
                if isinstance(sample, ReviewedStageCSampleV2)
                else None
            ),
        )


def _candidate_from_record(
    board: BoardState,
    record: StageCCandidateRecord,
    legal_chains: dict[tuple[str, str], tuple[tuple[Move, Move], BoardState]],
) -> SequenceCandidateEvidence:
    projected = legal_chains.get(record.moves_uci)
    if projected is None:
        raise StageCSampleIntegrityError("candidate is not a legal two-ply chain")
    moves, final = projected
    if final.position_id != record.final_position_id:
        raise StageCSampleIntegrityError("candidate final position id is not rule-grounded")
    changed_points = tuple(
        index
        for index, (before, after) in enumerate(zip(board.pieces, final.pieces, strict=True))
        if before != after
    )
    if changed_points != record.changed_points:
        raise StageCSampleIntegrityError("candidate changed points are not rule-grounded")
    return SequenceCandidateEvidence(
        moves=moves,
        changed_points=record.changed_points,
        expected_change_floor=record.expected_change_floor,
        unexpected_difference=record.unexpected_difference,
        maximum_template_distance=record.maximum_template_distance,
        minimum_template_margin=record.minimum_template_margin,
        minimum_template_confidence=record.minimum_template_confidence,
        score=record.score,
        final_position_id=record.final_position_id,
    )


def _recorded_observation_matches(
    sample: StageCSampleMetadata,
    *,
    accepted: bool,
    replayed_moves: tuple[str, ...],
    replayed_final_id: str | None,
    rejection_reasons: tuple[str, ...],
) -> bool:
    if accepted:
        return (
            sample.observed_status is StageCObservedStatus.ACCEPTED
            and sample.observed_moves_uci == replayed_moves
            and sample.observed_final_position_id == replayed_final_id
            and not sample.rejection_reasons
        )
    reasons_match = sample.rejection_reasons == rejection_reasons
    # Capture-layer terminal reasons cannot be reproduced by SequenceDecisionGate.
    # Reviewed V2 provenance authenticates them separately from the replay decision.
    if (
        isinstance(sample, ReviewedStageCSampleV2)
        and sample.scenario is StageCScenario.RESIZE
        and bool(set(sample.rejection_reasons) & _CAPTURE_TERMINAL_REASONS)
    ):
        reasons_match = True
    return (
        sample.observed_status is StageCObservedStatus.REJECTED
        and not sample.observed_moves_uci
        and sample.observed_final_position_id == sample.confirmed_position_id
        and reasons_match
    )


def _board_from_metadata(sample: StageCSampleMetadata) -> BoardState:
    board = parse_fen(sample.confirmed_fen)
    if board.side_to_move != sample.side_to_move:
        raise StageCSampleIntegrityError("confirmed FEN side does not match sample side")
    return replace(board, orientation=sample.orientation)


def _project_uci_chain(board: BoardState, moves_uci: tuple[str, ...]) -> BoardState | None:
    projected = board
    for uci in moves_uci:
        move = next((candidate for candidate in legal_moves(projected) if candidate.uci == uci), None)
        if move is None:
            return None
        projected = apply_move(projected, move)
    return projected


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageCSampleIntegrityError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StageCSampleIntegrityError("manifest root must be an object")
    return cast(dict[str, Any], value)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise StageCSampleIntegrityError(f"{name} must be a string mapping")
    return cast(dict[str, str], value)


def _metadata_from_payload(payload: dict[str, Any]) -> HumanAiStageCSampleV1:
    try:
        context_payload = cast(dict[str, Any], payload["capture_context"])
        context = CaptureContext(
            wgc_size=_size(context_payload["wgc_size"]),
            client_size=_size(context_payload["client_size"]),
            dpi_scale=float(context_payload["dpi_scale"]),
            geometry_revision=str(context_payload["geometry_revision"]),
            theme_fingerprint=str(context_payload["theme_fingerprint"]),
            generation_id=int(context_payload["generation_id"]),
        )
        candidate_payloads = cast(list[dict[str, object]], payload["candidates"])
        candidates = tuple(_candidate_metadata(item) for item in candidate_payloads)
        expected_final = payload["expected_final_position_id"]
        observed_final = payload["observed_final_position_id"]
        return HumanAiStageCSampleV1(
            sample_id=str(payload["sample_id"]),
            session_id=str(payload["session_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            confirmed_fen=str(payload["confirmed_fen"]),
            confirmed_position_id=str(payload["confirmed_position_id"]),
            expected_outcome=StageCExpectedOutcome(str(payload["expected_outcome"])),
            scenario=StageCScenario(str(payload["scenario"])),
            ground_truth_moves_uci=_moves(payload["ground_truth_moves_uci"]),
            expected_final_position_id=(
                None if expected_final is None else str(expected_final)
            ),
            observed_status=StageCObservedStatus(str(payload["observed_status"])),
            observed_moves_uci=_moves(payload["observed_moves_uci"]),
            observed_final_position_id=(
                None if observed_final is None else str(observed_final)
            ),
            side_to_move=cast(Any, payload["side_to_move"]),
            orientation=Orientation(str(payload["orientation"])),
            changed_points=_indices(payload["changed_points"]),
            local_differences=_floats(payload["local_differences"]),
            candidates=candidates,
            rejection_reasons=tuple(str(value) for value in payload["rejection_reasons"]),
            capture_context=context,
            feature_version=str(payload["feature_version"]),
            threshold_profile_version=str(payload["threshold_profile_version"]),
            decision_latency_ms=float(payload["decision_latency_ms"]),
            schema_version=int(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageCSampleIntegrityError("Stage C manifest metadata is invalid") from exc


def _candidate_metadata(payload: dict[str, object]) -> StageCCandidateRecord:
    return StageCCandidateRecord(
        moves_uci=cast(tuple[str, str], _moves(payload["moves_uci"])),
        changed_points=_indices(payload["changed_points"]),
        expected_change_floor=_float_value(payload["expected_change_floor"]),
        unexpected_difference=_float_value(payload["unexpected_difference"]),
        maximum_template_distance=_float_value(payload["maximum_template_distance"]),
        minimum_template_margin=_float_value(payload["minimum_template_margin"]),
        minimum_template_confidence=_float_value(
            payload["minimum_template_confidence"]
        ),
        score=_float_value(payload["score"]),
        final_position_id=str(payload["final_position_id"]),
    )


def _size(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("capture size must contain two integers")
    return int(value[0]), int(value[1])


def _moves(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError("moves must be a list")
    return tuple(str(item) for item in value)


def _indices(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("indices must be a list")
    return tuple(int(item) for item in value)


def _floats(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError("numeric evidence must be a list")
    return tuple(float(item) for item in value)


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError("numeric manifest value is invalid")
    return float(value)


def _decode_crop(contents: bytes, point: int, suffix: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (48, 48, 4):
        raise StageCSampleIntegrityError(
            f"Stage C crop is not a 48x48 BGRA PNG: point-{point:02d}-{suffix}"
        )
    return np.asarray(image, dtype=np.uint8)


def _sha256(contents: bytes) -> str:
    from hashlib import sha256

    return sha256(contents).hexdigest()
