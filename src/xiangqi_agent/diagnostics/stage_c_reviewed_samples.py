from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any, cast

import cv2
import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCReviewOutcome,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
    _validate_replay_fields,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import BoardState, Orientation, Side
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves

_CROP_SIZE = 48
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MANIFEST_FIELDS = {
    "sample_id",
    "session_id",
    "created_at_utc",
    "confirmed_fen",
    "confirmed_position_id",
    "expected_outcome",
    "scenario",
    "ground_truth_moves_uci",
    "expected_final_position_id",
    "observed_status",
    "observed_moves_uci",
    "observed_final_position_id",
    "side_to_move",
    "orientation",
    "changed_points",
    "local_differences",
    "candidates",
    "rejection_reasons",
    "capture_context",
    "feature_version",
    "threshold_profile_version",
    "decision_latency_ms",
    "source_event_manifest_sha256",
    "review_manifest_sha256",
    "label_source",
    "review_outcome",
    "occupancy_verifier_version",
    "promotion_verifier_version",
    "promoted_at_utc",
    "schema_version",
    "crop_hashes",
}
_CANDIDATE_FIELDS = {
    "moves_uci",
    "changed_points",
    "expected_change_floor",
    "unexpected_difference",
    "maximum_template_distance",
    "minimum_template_margin",
    "minimum_template_confidence",
    "score",
    "final_position_id",
}
_SOURCE_PROVENANCE_FIELDS = {
    "event_id",
    "session_id",
    "created_at_utc",
    "confirmed_fen",
    "confirmed_position_id",
    "observed_status",
    "observed_moves_uci",
    "observed_final_position_id",
    "side_to_move",
    "orientation",
    "changed_points",
    "local_differences",
    "candidates",
    "rejection_reasons",
    "capture_context",
    "feature_version",
    "threshold_profile_version",
    "decision_latency_ms",
    "before_occupancy",
    "after_occupancy",
    "schema_version",
    "crop_hashes",
}
_REVIEW_PROVENANCE_FIELDS = {
    "review_id",
    "event_id",
    "session_id",
    "created_at_utc",
    "event_manifest_sha256",
    "label_kind",
    "moves_uci",
    "expected_final_position_id",
    "scenario",
    "review_outcome",
    "supersedes_review_id",
    "reviewer_kind",
    "ui_version",
    "rules_version",
    "schema_version",
}
_CAPTURE_CONTEXT_FIELDS = {
    "wgc_size",
    "client_size",
    "dpi_scale",
    "geometry_revision",
    "theme_fingerprint",
    "generation_id",
}
_OCCUPANCY_FIELDS = {"occupied", "confidences", "algorithm_version"}


class ReviewedStageCSampleIntegrityError(ValueError):
    """A reviewed V2 sample is incomplete, changed, or path-unsafe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewedStageCSampleV2:
    sample_id: str
    session_id: str
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
    expected_outcome: StageCExpectedOutcome
    scenario: StageCScenario
    ground_truth_moves_uci: tuple[str, ...]
    expected_final_position_id: str | None
    observed_status: StageCObservedStatus
    observed_moves_uci: tuple[str, ...]
    observed_final_position_id: str | None
    side_to_move: Side
    orientation: Orientation
    changed_points: tuple[int, ...]
    local_differences: tuple[float, ...]
    candidates: tuple[StageCCandidateRecord, ...]
    rejection_reasons: tuple[str, ...]
    capture_context: CaptureContext
    feature_version: str
    threshold_profile_version: str
    decision_latency_ms: float
    source_event_manifest_sha256: str
    review_manifest_sha256: str
    review_outcome: StageCReviewOutcome
    occupancy_verifier_version: str
    promotion_verifier_version: str
    promoted_at_utc: str
    label_source: str = "post_event_local_user_review"
    schema_version: int = 2

    def __post_init__(self) -> None:
        _validate_replay_fields(self)
        for value in (
            self.source_event_manifest_sha256,
            self.review_manifest_sha256,
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(
                    "provenance hashes must contain 64 lowercase hexadecimal characters"
                )
        if self.label_source != "post_event_local_user_review":
            raise ValueError("label_source must be post_event_local_user_review")
        if not isinstance(self.review_outcome, StageCReviewOutcome):
            raise TypeError("review_outcome must be a StageCReviewOutcome")
        if self.expected_outcome is StageCExpectedOutcome.ACCEPT:
            if self.review_outcome not in (
                StageCReviewOutcome.CANDIDATE_CONFIRMED,
                StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
            ):
                raise ValueError("accept sample has an invalid review outcome")
        elif self.review_outcome is not StageCReviewOutcome.EXPECTED_REJECTION:
            raise ValueError("rejection sample has an invalid review outcome")
        if any(
            not isinstance(version, str) or not version.strip()
            for version in (
                self.occupancy_verifier_version,
                self.promotion_verifier_version,
            )
        ):
            raise ValueError("occupancy and promotion verifier versions must be non-empty")
        _parse_promoted_utc(self.promoted_at_utc)
        if self.schema_version != 2:
            raise ValueError("ReviewedStageCSampleV2 schema_version must be 2")


@dataclass(frozen=True, slots=True)
class LoadedReviewedStageCSample:
    metadata: ReviewedStageCSampleV2
    crops: tuple[TransitionPointCrops, ...]
    directory: Path
    source_event_manifest_bytes: bytes
    review_manifest_bytes: bytes


class ReviewedStageCSampleLoader:
    def load(self, sample_dir: Path) -> LoadedReviewedStageCSample:
        if not isinstance(sample_dir, Path):
            raise TypeError("sample directory must be a Path")
        if sample_dir.is_symlink() or not sample_dir.is_dir():
            raise ReviewedStageCSampleIntegrityError(
                "reviewed sample directory does not exist or is a symlink"
            )
        _validate_directory(sample_dir.parent, "reviewed session")
        manifest_path = sample_dir / "manifest.json"
        payload = _read_json_object(manifest_path, "reviewed manifest")
        if set(payload) != _MANIFEST_FIELDS:
            raise ReviewedStageCSampleIntegrityError(
                "reviewed manifest fields are incomplete or unexpected"
            )
        metadata = _metadata_from_payload(payload)
        if sample_dir.name != metadata.sample_id or sample_dir.parent.name != metadata.session_id:
            raise ReviewedStageCSampleIntegrityError(
                "reviewed sample path does not match its identifiers"
            )

        crop_files = tuple(
            filename
            for point in metadata.changed_points
            for filename in (
                f"point-{point:02d}-before.png",
                f"point-{point:02d}-after.png",
            )
        )
        expected_files = frozenset(
            (
                "manifest.json",
                "source-event-manifest.json",
                "review-manifest.json",
                *crop_files,
            )
        )
        entries = tuple(sample_dir.iterdir())
        if (
            any(path.is_symlink() or not path.is_file() for path in entries)
            or frozenset(path.name for path in entries) != expected_files
        ):
            raise ReviewedStageCSampleIntegrityError(
                "reviewed sample must contain exactly its provenance, crops, and manifest"
            )

        source_bytes = (sample_dir / "source-event-manifest.json").read_bytes()
        review_bytes = (sample_dir / "review-manifest.json").read_bytes()
        if sha256(source_bytes).hexdigest() != metadata.source_event_manifest_sha256:
            raise ReviewedStageCSampleIntegrityError("source event manifest hash mismatch")
        if sha256(review_bytes).hexdigest() != metadata.review_manifest_sha256:
            raise ReviewedStageCSampleIntegrityError("review manifest hash mismatch")

        crop_hashes = _string_mapping(payload["crop_hashes"], "crop_hashes")
        if frozenset(crop_hashes) != frozenset(crop_files) or any(
            _SHA256.fullmatch(value) is None for value in crop_hashes.values()
        ):
            raise ReviewedStageCSampleIntegrityError(
                "reviewed crop hashes do not match declared points"
            )
        _validate_provenance(metadata, source_bytes, review_bytes, crop_hashes)
        encoded: dict[str, bytes] = {}
        for filename in crop_files:
            contents = (sample_dir / filename).read_bytes()
            if sha256(contents).hexdigest() != crop_hashes[filename]:
                raise ReviewedStageCSampleIntegrityError(f"reviewed crop hash mismatch: {filename}")
            encoded[filename] = contents
        crops = tuple(
            TransitionPointCrops(
                point,
                _decode_crop(encoded[f"point-{point:02d}-before.png"], point, "before"),
                _decode_crop(encoded[f"point-{point:02d}-after.png"], point, "after"),
            )
            for point in metadata.changed_points
        )
        _validate_rule_projection(metadata)
        return LoadedReviewedStageCSample(
            metadata,
            crops,
            sample_dir,
            source_bytes,
            review_bytes,
        )


def reviewed_manifest_bytes(
    sample: ReviewedStageCSampleV2,
    crop_hashes: dict[str, str],
) -> bytes:
    if not isinstance(sample, ReviewedStageCSampleV2):
        raise TypeError("sample must be a ReviewedStageCSampleV2")
    if not isinstance(crop_hashes, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for key, value in crop_hashes.items()
    ):
        raise ValueError("crop_hashes must map filenames to SHA-256 values")
    payload = asdict(sample)
    payload["expected_outcome"] = sample.expected_outcome.value
    payload["scenario"] = sample.scenario.value
    payload["observed_status"] = sample.observed_status.value
    payload["orientation"] = sample.orientation.value
    payload["review_outcome"] = sample.review_outcome.value
    payload["crop_hashes"] = dict(sorted(crop_hashes.items()))
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def purge_expired_reviewed_samples(
    reviewed_root: Path,
    *,
    protected_relative_paths: frozenset[str],
    now_utc: datetime,
    retention_days: int = 30,
) -> tuple[Path, ...]:
    if not isinstance(reviewed_root, Path):
        raise TypeError("reviewed_root must be a Path")
    if not isinstance(protected_relative_paths, frozenset):
        raise TypeError("protected_relative_paths must be a frozenset")
    protected = frozenset(
        _validate_relative_sample_path(value) for value in protected_relative_paths
    )
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("cleanup clock must be a timezone-aware datetime")
    if (
        isinstance(retention_days, bool)
        or not isinstance(retention_days, int)
        or retention_days < 0
    ):
        raise ValueError("retention_days must be a non-negative integer")
    if reviewed_root.is_symlink():
        raise ReviewedStageCSampleIntegrityError("reviewed root must not be a symlink")
    if not reviewed_root.exists():
        return ()
    if not reviewed_root.is_dir():
        raise ReviewedStageCSampleIntegrityError("reviewed root must be a directory")

    loader = ReviewedStageCSampleLoader()
    validated: list[tuple[str, Path, ReviewedStageCSampleV2]] = []
    for session_dir in sorted(reviewed_root.iterdir()):
        _validate_directory(session_dir, "reviewed session")
        if _IDENTIFIER.fullmatch(session_dir.name) is None:
            raise ReviewedStageCSampleIntegrityError("reviewed session identifier is unsafe")
        for sample_dir in sorted(session_dir.iterdir()):
            loaded = loader.load(sample_dir)
            relative = f"{session_dir.name}/{sample_dir.name}"
            validated.append((relative, sample_dir, loaded.metadata))

    cutoff = now_utc.astimezone(UTC) - timedelta(days=retention_days)
    removed = tuple(
        path
        for relative, path, metadata in validated
        if relative not in protected and _parse_promoted_utc(metadata.promoted_at_utc) < cutoff
    )
    for path in removed:
        _assert_within_root(path, reviewed_root)
        shutil.rmtree(path)
    for session_dir in tuple(sorted(reviewed_root.iterdir())):
        if session_dir.is_dir() and not any(session_dir.iterdir()):
            session_dir.rmdir()
    return removed


def _metadata_from_payload(payload: dict[str, Any]) -> ReviewedStageCSampleV2:
    try:
        context_payload = _mapping(payload["capture_context"], "capture_context")
        _require_fields(
            context_payload,
            {
                "wgc_size",
                "client_size",
                "dpi_scale",
                "geometry_revision",
                "theme_fingerprint",
                "generation_id",
            },
            "capture_context",
        )
        candidates_value = payload["candidates"]
        if not isinstance(candidates_value, list):
            raise TypeError("candidates must be a list")
        candidates = tuple(
            _candidate_from_payload(_mapping(value, "candidate")) for value in candidates_value
        )
        expected_final = payload["expected_final_position_id"]
        observed_final = payload["observed_final_position_id"]
        return ReviewedStageCSampleV2(
            sample_id=_string(payload["sample_id"]),
            session_id=_string(payload["session_id"]),
            created_at_utc=_string(payload["created_at_utc"]),
            confirmed_fen=_string(payload["confirmed_fen"]),
            confirmed_position_id=_string(payload["confirmed_position_id"]),
            expected_outcome=StageCExpectedOutcome(_string(payload["expected_outcome"])),
            scenario=StageCScenario(_string(payload["scenario"])),
            ground_truth_moves_uci=_moves(payload["ground_truth_moves_uci"]),
            expected_final_position_id=(
                None if expected_final is None else _string(expected_final)
            ),
            observed_status=StageCObservedStatus(_string(payload["observed_status"])),
            observed_moves_uci=_moves(payload["observed_moves_uci"]),
            observed_final_position_id=(
                None if observed_final is None else _string(observed_final)
            ),
            side_to_move=cast(Side, _string(payload["side_to_move"])),
            orientation=Orientation(_string(payload["orientation"])),
            changed_points=_indices(payload["changed_points"]),
            local_differences=_floats(payload["local_differences"]),
            candidates=candidates,
            rejection_reasons=_strings(payload["rejection_reasons"]),
            capture_context=CaptureContext(
                wgc_size=_size(context_payload["wgc_size"]),
                client_size=_size(context_payload["client_size"]),
                dpi_scale=_float(context_payload["dpi_scale"]),
                geometry_revision=_string(context_payload["geometry_revision"]),
                theme_fingerprint=_string(context_payload["theme_fingerprint"]),
                generation_id=_integer(context_payload["generation_id"]),
            ),
            feature_version=_string(payload["feature_version"]),
            threshold_profile_version=_string(payload["threshold_profile_version"]),
            decision_latency_ms=_float(payload["decision_latency_ms"]),
            source_event_manifest_sha256=_string(payload["source_event_manifest_sha256"]),
            review_manifest_sha256=_string(payload["review_manifest_sha256"]),
            label_source=_string(payload["label_source"]),
            review_outcome=StageCReviewOutcome(_string(payload["review_outcome"])),
            occupancy_verifier_version=_string(payload["occupancy_verifier_version"]),
            promotion_verifier_version=_string(payload["promotion_verifier_version"]),
            promoted_at_utc=_string(payload["promoted_at_utc"]),
            schema_version=_integer(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReviewedStageCSampleIntegrityError):
            raise
        raise ReviewedStageCSampleIntegrityError("reviewed manifest metadata is invalid") from exc


def _candidate_from_payload(payload: dict[str, Any]) -> StageCCandidateRecord:
    _require_fields(payload, _CANDIDATE_FIELDS, "candidate")
    moves = _moves(payload["moves_uci"])
    if len(moves) != 2:
        raise ValueError("candidate must contain two moves")
    return StageCCandidateRecord(
        moves_uci=moves,
        changed_points=_indices(payload["changed_points"]),
        expected_change_floor=_float(payload["expected_change_floor"]),
        unexpected_difference=_float(payload["unexpected_difference"]),
        maximum_template_distance=_float(payload["maximum_template_distance"]),
        minimum_template_margin=_float(payload["minimum_template_margin"]),
        minimum_template_confidence=_float(payload["minimum_template_confidence"]),
        score=_float(payload["score"]),
        final_position_id=_string(payload["final_position_id"]),
    )


def _validate_rule_projection(sample: ReviewedStageCSampleV2) -> None:
    try:
        board = parse_fen(sample.confirmed_fen)
    except ValueError as exc:
        raise ReviewedStageCSampleIntegrityError("confirmed FEN is invalid") from exc
    if (
        board.position_id != sample.confirmed_position_id
        or board.side_to_move != sample.side_to_move
    ):
        raise ReviewedStageCSampleIntegrityError("confirmed FEN does not match reviewed metadata")
    expected = _project(board, sample.ground_truth_moves_uci)
    if expected is None:
        raise ReviewedStageCSampleIntegrityError("reviewed ground truth is not sequentially legal")
    if (
        sample.expected_outcome is StageCExpectedOutcome.ACCEPT
        and expected.position_id != sample.expected_final_position_id
    ):
        raise ReviewedStageCSampleIntegrityError(
            "reviewed final position does not match ground truth"
        )
    if sample.observed_status is StageCObservedStatus.ACCEPTED:
        observed = _project(board, sample.observed_moves_uci)
        if observed is None or observed.position_id != sample.observed_final_position_id:
            raise ReviewedStageCSampleIntegrityError("observed final position is not rule-grounded")


def _validate_provenance(
    sample: ReviewedStageCSampleV2,
    source_bytes: bytes,
    review_bytes: bytes,
    crop_hashes: dict[str, str],
) -> None:
    try:
        source_value = json.loads(source_bytes.decode("utf-8"))
        review_value = json.loads(review_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedStageCSampleIntegrityError(
            "reviewed provenance sidecars must be valid UTF-8 JSON"
        ) from exc
    if not isinstance(source_value, dict) or not isinstance(review_value, dict):
        raise ReviewedStageCSampleIntegrityError(
            "reviewed provenance sidecars must contain objects"
        )
    source = cast(dict[str, Any], source_value)
    review = cast(dict[str, Any], review_value)
    try:
        _require_fields(source, _SOURCE_PROVENANCE_FIELDS, "source provenance")
        _require_fields(review, _REVIEW_PROVENANCE_FIELDS, "review provenance")
        candidates_value = source["candidates"]
        if not isinstance(candidates_value, list):
            raise TypeError("source candidates must be a list")
        source_candidates = tuple(
            _candidate_from_payload(_mapping(value, "source candidate"))
            for value in candidates_value
        )
        context_payload = _mapping(source["capture_context"], "source capture context")
        _require_fields(
            context_payload,
            _CAPTURE_CONTEXT_FIELDS,
            "source capture context",
        )
        source_context = CaptureContext(
            wgc_size=_size(context_payload["wgc_size"]),
            client_size=_size(context_payload["client_size"]),
            dpi_scale=_float(context_payload["dpi_scale"]),
            geometry_revision=_string(context_payload["geometry_revision"]),
            theme_fingerprint=_string(context_payload["theme_fingerprint"]),
            generation_id=_integer(context_payload["generation_id"]),
        )
        before_version = _occupancy_version(source["before_occupancy"])
        after_version = _occupancy_version(source["after_occupancy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ReviewedStageCSampleIntegrityError(
            "reviewed provenance fields are invalid"
        ) from exc
    source_matches = (
        source.get("event_id") == sample.sample_id
        and source.get("session_id") == sample.session_id
        and source.get("created_at_utc") == sample.created_at_utc
        and source.get("confirmed_fen") == sample.confirmed_fen
        and source.get("confirmed_position_id") == sample.confirmed_position_id
        and source.get("observed_status") == sample.observed_status.value
        and source.get("observed_moves_uci") == list(sample.observed_moves_uci)
        and source.get("observed_final_position_id")
        == sample.observed_final_position_id
        and source.get("side_to_move") == sample.side_to_move
        and source.get("orientation") == sample.orientation.value
        and source.get("changed_points") == list(sample.changed_points)
        and source.get("local_differences") == list(sample.local_differences)
        and source.get("rejection_reasons") == list(sample.rejection_reasons)
        and source.get("feature_version") == sample.feature_version
        and source.get("threshold_profile_version")
        == sample.threshold_profile_version
        and source.get("decision_latency_ms") == sample.decision_latency_ms
        and source.get("crop_hashes") == crop_hashes
        and source.get("schema_version") == 1
        and source_candidates == sample.candidates
        and source_context == sample.capture_context
        and before_version == sample.occupancy_verifier_version
        and after_version == sample.occupancy_verifier_version
    )
    if not source_matches:
        raise ReviewedStageCSampleIntegrityError(
            "source event provenance does not match reviewed metadata"
        )

    accepted = sample.expected_outcome is StageCExpectedOutcome.ACCEPT
    expected_label = "valid_two_ply" if accepted else "expected_rejection"
    expected_scenario = None if accepted else sample.scenario.value
    review_matches = (
        review.get("event_id") == sample.sample_id
        and review.get("session_id") == sample.session_id
        and review.get("event_manifest_sha256")
        == sample.source_event_manifest_sha256
        and review.get("label_kind") == expected_label
        and review.get("moves_uci") == list(sample.ground_truth_moves_uci)
        and review.get("expected_final_position_id")
        == sample.expected_final_position_id
        and review.get("scenario") == expected_scenario
        and review.get("review_outcome") == sample.review_outcome.value
        and review.get("reviewer_kind") == "local_user"
        and review.get("ui_version") == "stage-c-review-v1"
        and review.get("rules_version") == "xiangqi-rules-v1"
        and review.get("schema_version") == 1
    )
    if not review_matches:
        raise ReviewedStageCSampleIntegrityError(
            "local review provenance does not match reviewed metadata"
        )


def _occupancy_version(value: object) -> str:
    payload = _mapping(value, "occupancy provenance")
    _require_fields(payload, _OCCUPANCY_FIELDS, "occupancy provenance")
    occupied = payload["occupied"]
    confidences = payload["confidences"]
    if (
        not isinstance(occupied, list)
        or len(occupied) != 90
        or any(not isinstance(item, bool) for item in occupied)
    ):
        raise ValueError("occupancy provenance must contain 90 booleans")
    if not isinstance(confidences, list) or len(confidences) != 90:
        raise ValueError("occupancy provenance must contain 90 confidences")
    parsed_confidences = tuple(_float(item) for item in confidences)
    if any(value < 0.0 or value > 1.0 for value in parsed_confidences):
        raise ValueError("occupancy provenance confidence is outside [0, 1]")
    version = _string(payload["algorithm_version"])
    if not version.strip():
        raise ValueError("occupancy provenance algorithm version is empty")
    return version


def _project(board: BoardState, moves_uci: tuple[str, ...]) -> BoardState | None:
    projected = board
    for uci in moves_uci:
        move = next(
            (candidate for candidate in legal_moves(projected) if candidate.uci == uci),
            None,
        )
        if move is None:
            return None
        projected = apply_move(projected, move)
    return projected


def _decode_crop(contents: bytes, point: int, suffix: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        image is None
        or image.dtype != np.uint8
        or image.shape
        != (
            _CROP_SIZE,
            _CROP_SIZE,
            4,
        )
    ):
        raise ReviewedStageCSampleIntegrityError(
            f"reviewed crop is not 48x48 BGRA: point-{point:02d}-{suffix}"
        )
    owned = np.array(image, dtype=np.uint8, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReviewedStageCSampleIntegrityError(f"{name} is missing or symlinked")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedStageCSampleIntegrityError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ReviewedStageCSampleIntegrityError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _parse_promoted_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("promoted_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("promoted_at_utc must be a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _validate_relative_sample_path(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("protected paths must be strings")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or any(_IDENTIFIER.fullmatch(part) is None for part in path.parts)
    ):
        raise ValueError("protected path must be a safe session/sample relative path")
    return path.as_posix()


def _validate_directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ReviewedStageCSampleIntegrityError(f"{name} must be a real directory, not a symlink")


def _assert_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ReviewedStageCSampleIntegrityError("reviewed path escapes its root") from exc


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _require_fields(payload: dict[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise ReviewedStageCSampleIntegrityError(f"{name} fields are incomplete or unexpected")


def _string_mapping(value: object, name: str) -> dict[str, str]:
    payload = _mapping(value, name)
    if any(not isinstance(key, str) or not isinstance(item, str) for key, item in payload.items()):
        raise TypeError(f"{name} must be a string mapping")
    return cast(dict[str, str], payload)


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    return value


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError("numeric value must be finite")
    return result


def _size(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError("size must contain two integers")
    return _integer(value[0]), _integer(value[1])


def _moves(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("moves must be a list of strings")
    return tuple(value)


def _indices(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError("indices must be a list")
    return tuple(_integer(item) for item in value)


def _floats(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError("numeric evidence must be a list")
    return tuple(_float(item) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("values must be a list of strings")
    return tuple(value)
