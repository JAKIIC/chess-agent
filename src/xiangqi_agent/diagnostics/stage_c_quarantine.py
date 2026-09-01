from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import cv2
import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCCandidateRecord,
    StageCObservedStatus,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import Orientation, Side
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.vision.occupancy import OccupancyEvidence

_CROP_SIZE = 48
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_RETENTION_DAYS = 7
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_POSITION_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UCI = re.compile(r"^[a-i][0-9][a-i][0-9]$")
_MANIFEST_FIELDS = {
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


class QuarantineEventIntegrityError(ValueError):
    """A quarantined event is incomplete, changed, or structurally unsafe."""


@dataclass(frozen=True, slots=True, kw_only=True)
class QuarantinedStageCEventV1:
    event_id: str
    session_id: str
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
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
    before_occupancy: OccupancyEvidence
    after_occupancy: OccupancyEvidence
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.event_id)
        _validate_identifier(self.session_id)
        _parse_utc(self.created_at_utc)
        if not isinstance(self.confirmed_fen, str) or not self.confirmed_fen.strip():
            raise ValueError("confirmed_fen must be non-empty")
        _validate_position_id(self.confirmed_position_id)
        if not isinstance(self.observed_status, StageCObservedStatus):
            raise TypeError("observed_status must be a StageCObservedStatus")
        _validate_moves(self.observed_moves_uci, maximum=2)
        if self.observed_status is StageCObservedStatus.ACCEPTED:
            if len(self.observed_moves_uci) != 2 or self.observed_final_position_id is None:
                raise ValueError(
                    "accepted observation must expose two moves and a final position"
                )
            _validate_position_id(self.observed_final_position_id)
            if self.rejection_reasons:
                raise ValueError("accepted observation must not contain rejection reasons")
        else:
            if self.observed_moves_uci:
                raise ValueError("rejected observation must not expose moves")
            if self.observed_final_position_id != self.confirmed_position_id:
                raise ValueError("rejected observation must keep the confirmed position")
            if not self.rejection_reasons:
                raise ValueError("rejected observation must contain a rejection reason")
        if self.side_to_move not in ("w", "b"):
            raise ValueError("side_to_move must be w or b")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation must be an Orientation")
        _validate_points(self.changed_points)
        if not isinstance(self.local_differences, tuple) or len(self.local_differences) != 90:
            raise ValueError("local_differences must contain exactly 90 values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            for value in self.local_differences
        ):
            raise ValueError("local differences must be finite non-negative values")
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, StageCCandidateRecord) for candidate in self.candidates
        ):
            raise TypeError("candidates must be a tuple of StageCCandidateRecord")
        if len(self.candidates) > 2:
            raise ValueError("quarantine evidence keeps at most two candidates")
        ranked = tuple(
            sorted(
                self.candidates,
                key=lambda candidate: (-candidate.score, candidate.moves_uci),
            )
        )
        if ranked != self.candidates:
            raise ValueError("quarantine candidates must use deterministic ranked order")
        if not isinstance(self.rejection_reasons, tuple) or any(
            not isinstance(reason, str) or not reason.strip()
            for reason in self.rejection_reasons
        ):
            raise TypeError("rejection_reasons must be a tuple of non-empty strings")
        if len(set(self.rejection_reasons)) != len(self.rejection_reasons):
            raise ValueError("rejection reasons must not contain duplicates")
        if not isinstance(self.capture_context, CaptureContext):
            raise TypeError("capture_context must be a CaptureContext")
        for version in (self.feature_version, self.threshold_profile_version):
            if not isinstance(version, str) or not version.strip():
                raise ValueError("feature and threshold profile versions must be non-empty")
        if (
            isinstance(self.decision_latency_ms, bool)
            or not isinstance(self.decision_latency_ms, (int, float))
            or not isfinite(self.decision_latency_ms)
            or self.decision_latency_ms < 0
        ):
            raise ValueError("decision_latency_ms must be finite and non-negative")
        if not isinstance(self.before_occupancy, OccupancyEvidence) or not isinstance(
            self.after_occupancy,
            OccupancyEvidence,
        ):
            raise TypeError("occupancy snapshots must be OccupancyEvidence values")
        if (
            self.before_occupancy.algorithm_version
            != self.after_occupancy.algorithm_version
        ):
            raise ValueError("occupancy algorithm versions must match")
        if self.schema_version != 1:
            raise ValueError("QuarantinedStageCEventV1 schema_version must be 1")


@dataclass(frozen=True, slots=True)
class LoadedQuarantinedStageCEvent:
    metadata: QuarantinedStageCEventV1
    crops: tuple[TransitionPointCrops, ...]
    directory: Path
    manifest_bytes: bytes


class QuarantineEventRecorder:
    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = False,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("quarantine root must be a Path")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(retention_days, bool)
            or not isinstance(retention_days, int)
            or retention_days < 0
        ):
            raise ValueError("retention_days must be a non-negative integer")
        self._root = root
        self._enabled = enabled
        self._max_bytes = max_bytes
        self._retention_days = retention_days

    def record(
        self,
        event: QuarantinedStageCEventV1,
        crops: tuple[TransitionPointCrops, ...],
    ) -> Path:
        if not self._enabled:
            raise DiagnosticsDisabledError("quarantine diagnostics must be explicitly enabled")
        if not isinstance(event, QuarantinedStageCEventV1):
            raise TypeError("event must be a QuarantinedStageCEventV1")
        if not isinstance(crops, tuple) or any(
            not isinstance(crop, TransitionPointCrops) for crop in crops
        ):
            raise TypeError("crops must be a tuple of TransitionPointCrops")
        if tuple(crop.point_index for crop in crops) != event.changed_points:
            raise ValueError("quarantine crops must match changed_points in stable order")
        encoded = _encode_crops(crops)
        crop_hashes = {
            filename: sha256(contents).hexdigest()
            for filename, contents in encoded.items()
        }
        manifest_bytes = (
            json.dumps(
                _manifest(event, crop_hashes),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        event_time = _parse_utc(event.created_at_utc)

        _validate_storage_directory(self._root, "quarantine root", allow_missing=True)
        session_dir = self._root / event.session_id
        _validate_storage_directory(session_dir, "quarantine session", allow_missing=True)
        final_dir = session_dir / event.event_id
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(f"quarantine event already exists: {event.event_id}")

        self.purge_expired(event_time)
        added_bytes = len(manifest_bytes) + sum(len(value) for value in encoded.values())
        if _tree_size(self._root) + added_bytes > self._max_bytes:
            raise SampleQuotaExceededError("quarantine event capacity would be exceeded")

        temporary_dir = session_dir / f".{event.event_id}.tmp-{uuid4().hex}"
        temporary_dir.mkdir(parents=True)
        try:
            _assert_within_root(temporary_dir, self._root)
            for filename, contents in encoded.items():
                (temporary_dir / filename).write_bytes(contents)
            (temporary_dir / "manifest.json").write_bytes(manifest_bytes)
            loaded = QuarantineEventLoader().load(temporary_dir)
            if loaded.metadata != event:
                raise QuarantineEventIntegrityError(
                    "temporary quarantine event changed during verification"
                )
            temporary_dir.rename(final_dir)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            _remove_empty_parents(session_dir, self._root)
            raise
        return final_dir

    def purge_expired(self, now: datetime) -> int:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("purge clock must be a timezone-aware datetime")
        _validate_storage_directory(self._root, "quarantine root", allow_missing=True)
        if not self._root.exists():
            return 0
        cutoff = now.astimezone(UTC) - timedelta(days=self._retention_days)
        removed = 0
        for session_dir in tuple(sorted(self._root.iterdir())):
            if not session_dir.is_dir() or session_dir.is_symlink():
                raise QuarantineEventIntegrityError(
                    "quarantine root contains an unsafe session entry"
                )
            for event_dir in tuple(sorted(session_dir.iterdir())):
                if event_dir.name.startswith("."):
                    continue
                if not event_dir.is_dir() or event_dir.is_symlink():
                    raise QuarantineEventIntegrityError(
                        "quarantine session contains an unsafe event entry"
                    )
                created_at = _event_created_at(event_dir)
                if created_at is not None and created_at < cutoff:
                    _assert_within_root(event_dir, self._root)
                    shutil.rmtree(event_dir)
                    removed += 1
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
        if self._root.exists() and not any(self._root.iterdir()):
            self._root.rmdir()
        return removed


class QuarantineEventLoader:
    def load(self, event_dir: Path) -> LoadedQuarantinedStageCEvent:
        if not isinstance(event_dir, Path):
            raise TypeError("event directory must be a Path")
        if event_dir.is_symlink() or not event_dir.is_dir():
            raise QuarantineEventIntegrityError("quarantine event directory does not exist")
        manifest_path = event_dir / "manifest.json"
        if manifest_path.is_symlink():
            raise QuarantineEventIntegrityError("quarantine manifest must not be a symlink")
        manifest_bytes, payload = _read_manifest(manifest_path)
        if set(payload) != _MANIFEST_FIELDS:
            raise QuarantineEventIntegrityError(
                "quarantine manifest fields are incomplete or unexpected"
            )
        metadata = _metadata_from_payload(payload)
        expected_crop_files = tuple(
            filename
            for point in metadata.changed_points
            for filename in (
                f"point-{point:02d}-before.png",
                f"point-{point:02d}-after.png",
            )
        )
        expected_files = frozenset(("manifest.json", *expected_crop_files))
        entries = tuple(event_dir.iterdir())
        if any(path.is_symlink() for path in entries):
            raise QuarantineEventIntegrityError("quarantine event files must not be symlinks")
        if any(not path.is_file() for path in entries) or frozenset(
            path.name for path in entries
        ) != expected_files:
            raise QuarantineEventIntegrityError(
                "quarantine event must contain exactly the declared crops and manifest"
            )

        hashes = _string_mapping(payload["crop_hashes"], "crop_hashes")
        if frozenset(hashes) != frozenset(expected_crop_files) or any(
            _SHA256.fullmatch(value) is None for value in hashes.values()
        ):
            raise QuarantineEventIntegrityError(
                "quarantine crop hashes do not match declared points"
            )
        encoded: dict[str, bytes] = {}
        for filename in expected_crop_files:
            contents = (event_dir / filename).read_bytes()
            if sha256(contents).hexdigest() != hashes[filename]:
                raise QuarantineEventIntegrityError(
                    f"quarantine crop hash mismatch: {filename}"
                )
            encoded[filename] = contents
        crops = tuple(
            TransitionPointCrops(
                point,
                _decode_crop(encoded[f"point-{point:02d}-before.png"], point, "before"),
                _decode_crop(encoded[f"point-{point:02d}-after.png"], point, "after"),
            )
            for point in metadata.changed_points
        )

        try:
            board = parse_fen(metadata.confirmed_fen)
        except ValueError as exc:
            raise QuarantineEventIntegrityError("confirmed FEN is invalid") from exc
        if (
            board.position_id != metadata.confirmed_position_id
            or board.side_to_move != metadata.side_to_move
        ):
            raise QuarantineEventIntegrityError(
                "confirmed FEN does not match its position metadata"
            )
        return LoadedQuarantinedStageCEvent(
            metadata,
            crops,
            event_dir,
            manifest_bytes,
        )


def _manifest(
    event: QuarantinedStageCEventV1,
    crop_hashes: dict[str, str],
) -> dict[str, Any]:
    payload = asdict(event)
    payload["observed_status"] = event.observed_status.value
    payload["orientation"] = event.orientation.value
    payload["crop_hashes"] = dict(sorted(crop_hashes.items()))
    return payload


def _metadata_from_payload(payload: dict[str, Any]) -> QuarantinedStageCEventV1:
    try:
        context_payload = _mapping(payload["capture_context"], "capture_context")
        _require_exact_fields(
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
        context = CaptureContext(
            wgc_size=_size(context_payload["wgc_size"]),
            client_size=_size(context_payload["client_size"]),
            dpi_scale=_float_value(context_payload["dpi_scale"]),
            geometry_revision=_string(context_payload["geometry_revision"]),
            theme_fingerprint=_string(context_payload["theme_fingerprint"]),
            generation_id=_integer(context_payload["generation_id"]),
        )
        candidates_value = payload["candidates"]
        if not isinstance(candidates_value, list):
            raise TypeError("candidates must be a list")
        candidates = tuple(
            _candidate_from_payload(_mapping(value, "candidate"))
            for value in candidates_value
        )
        observed_final_value = payload["observed_final_position_id"]
        observed_final = (
            None if observed_final_value is None else _string(observed_final_value)
        )
        return QuarantinedStageCEventV1(
            event_id=_string(payload["event_id"]),
            session_id=_string(payload["session_id"]),
            created_at_utc=_string(payload["created_at_utc"]),
            confirmed_fen=_string(payload["confirmed_fen"]),
            confirmed_position_id=_string(payload["confirmed_position_id"]),
            observed_status=StageCObservedStatus(_string(payload["observed_status"])),
            observed_moves_uci=_moves(payload["observed_moves_uci"]),
            observed_final_position_id=observed_final,
            side_to_move=cast(Side, _string(payload["side_to_move"])),
            orientation=Orientation(_string(payload["orientation"])),
            changed_points=_indices(payload["changed_points"]),
            local_differences=_floats(payload["local_differences"]),
            candidates=candidates,
            rejection_reasons=_strings(payload["rejection_reasons"]),
            capture_context=context,
            feature_version=_string(payload["feature_version"]),
            threshold_profile_version=_string(payload["threshold_profile_version"]),
            decision_latency_ms=_float_value(payload["decision_latency_ms"]),
            before_occupancy=_occupancy_from_payload(payload["before_occupancy"]),
            after_occupancy=_occupancy_from_payload(payload["after_occupancy"]),
            schema_version=_integer(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, QuarantineEventIntegrityError):
            raise
        raise QuarantineEventIntegrityError(
            "quarantine manifest metadata is invalid"
        ) from exc


def _candidate_from_payload(payload: dict[str, Any]) -> StageCCandidateRecord:
    _require_exact_fields(
        payload,
        {
            "moves_uci",
            "changed_points",
            "expected_change_floor",
            "unexpected_difference",
            "maximum_template_distance",
            "minimum_template_margin",
            "minimum_template_confidence",
            "score",
            "final_position_id",
        },
        "candidate",
    )
    moves = _moves(payload["moves_uci"])
    if len(moves) != 2:
        raise ValueError("candidate must contain two moves")
    return StageCCandidateRecord(
        moves_uci=moves,
        changed_points=_indices(payload["changed_points"]),
        expected_change_floor=_float_value(payload["expected_change_floor"]),
        unexpected_difference=_float_value(payload["unexpected_difference"]),
        maximum_template_distance=_float_value(payload["maximum_template_distance"]),
        minimum_template_margin=_float_value(payload["minimum_template_margin"]),
        minimum_template_confidence=_float_value(
            payload["minimum_template_confidence"]
        ),
        score=_float_value(payload["score"]),
        final_position_id=_string(payload["final_position_id"]),
    )


def _occupancy_from_payload(value: object) -> OccupancyEvidence:
    payload = _mapping(value, "occupancy")
    _require_exact_fields(
        payload,
        {"occupied", "confidences", "algorithm_version"},
        "occupancy",
    )
    occupied_value = payload["occupied"]
    confidences_value = payload["confidences"]
    if not isinstance(occupied_value, list) or not isinstance(confidences_value, list):
        raise TypeError("occupancy vectors must be lists")
    occupied = tuple(value for value in occupied_value if isinstance(value, bool))
    if len(occupied) != len(occupied_value):
        raise TypeError("occupancy values must be booleans")
    return OccupancyEvidence(
        occupied,
        tuple(_float_value(item) for item in confidences_value),
        _string(payload["algorithm_version"]),
    )


def _encode_crops(crops: tuple[TransitionPointCrops, ...]) -> dict[str, bytes]:
    encoded: dict[str, bytes] = {}
    for crop in crops:
        for suffix, value in (("before", crop.before), ("after", crop.after)):
            pixels = np.asarray(value)
            if pixels.dtype != np.uint8 or pixels.shape != (_CROP_SIZE, _CROP_SIZE, 4):
                raise ValueError(
                    "quarantine crops must be BGRA uint8 images of exactly 48x48 pixels"
                )
            success, buffer = cv2.imencode(
                ".png",
                pixels,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not success:
                raise RuntimeError(
                    f"failed to encode quarantine crop: {crop.point_index}-{suffix}"
                )
            encoded[f"point-{crop.point_index:02d}-{suffix}.png"] = buffer.tobytes()
    return encoded


def _decode_crop(contents: bytes, point: int, suffix: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (_CROP_SIZE, _CROP_SIZE, 4):
        raise QuarantineEventIntegrityError(
            f"quarantine crop is not a 48x48 BGRA PNG: point-{point:02d}-{suffix}"
        )
    owned = np.array(image, dtype=np.uint8, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_manifest(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        contents = path.read_bytes()
        value = json.loads(contents.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineEventIntegrityError(
            "quarantine manifest is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise QuarantineEventIntegrityError("quarantine manifest root must be an object")
    return contents, cast(dict[str, Any], value)


def _event_created_at(event_dir: Path) -> datetime | None:
    try:
        _, payload = _read_manifest(event_dir / "manifest.json")
        return _parse_utc(_string(payload["created_at_utc"]))
    except (KeyError, QuarantineEventIntegrityError, TypeError, ValueError):
        return None


def _validate_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("event and session identifier must be path-safe")


def _validate_position_id(value: str) -> None:
    if not isinstance(value, str) or _POSITION_ID.fullmatch(value) is None:
        raise ValueError("position id must be 32 lowercase hex characters")


def _validate_moves(values: tuple[str, ...], *, maximum: int) -> None:
    if not isinstance(values, tuple) or len(values) > maximum:
        raise ValueError(f"moves must contain at most {maximum} entries")
    if any(not isinstance(value, str) or _UCI.fullmatch(value) is None for value in values):
        raise ValueError("moves must use four-character Xiangqi UCI coordinates")


def _validate_points(values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple) or not 1 <= len(values) <= 4:
        raise ValueError("changed_points must contain one through four points")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 90
        for value in values
    ):
        raise ValueError("changed_points must contain board indices from zero through 89")
    if tuple(sorted(set(values))) != values:
        raise ValueError("changed_points must use unique stable ascending order")


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("created_at_utc must be a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _validate_storage_directory(path: Path, name: str, *, allow_missing: bool) -> None:
    if path.is_symlink():
        raise QuarantineEventIntegrityError(f"{name} must not be a symlink")
    if path.exists() and not path.is_dir():
        raise QuarantineEventIntegrityError(f"{name} must be a directory")
    if not allow_missing and not path.exists():
        raise QuarantineEventIntegrityError(f"{name} does not exist")


def _assert_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise QuarantineEventIntegrityError("quarantine path escapes its root") from exc


def _remove_empty_parents(session_dir: Path, root: Path) -> None:
    if session_dir.exists() and not any(session_dir.iterdir()):
        session_dir.rmdir()
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def _tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def _require_exact_fields(
    payload: dict[str, Any],
    expected: set[str],
    name: str,
) -> None:
    if set(payload) != expected:
        raise QuarantineEventIntegrityError(
            f"{name} fields are incomplete or unexpected"
        )


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


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be numeric")
    return float(value)


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
    return tuple(_float_value(item) for item in value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("values must be a list of strings")
    return tuple(value)
