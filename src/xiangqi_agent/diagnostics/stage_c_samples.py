from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import cv2
import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.transition_samples import TransitionPointCrops
from xiangqi_agent.domain.board import Orientation, Side

_CROP_SIZE = 48
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_RETENTION_DAYS = 7
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_POSITION_ID = re.compile(r"^[0-9a-f]{32}$")
_UCI = re.compile(r"^[a-i][0-9][a-i][0-9]$")


class StageCExpectedOutcome(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class StageCObservedStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class StageCScenario(StrEnum):
    VALID_TWO_PLY = "valid_two_ply"
    MULTIPLE_CANDIDATES = "multiple_candidates"
    SELECTION_HIGHLIGHT = "selection_highlight"
    CONTINUOUS_ANIMATION = "continuous_animation"
    OCCLUSION = "occlusion"
    RESIZE = "resize"
    THREE_PLY = "three_ply"


@dataclass(frozen=True, slots=True, kw_only=True)
class StageCCandidateRecord:
    moves_uci: tuple[str, str]
    changed_points: tuple[int, ...]
    expected_change_floor: float
    unexpected_difference: float
    maximum_template_distance: float
    minimum_template_margin: float
    minimum_template_confidence: float
    score: float
    final_position_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.moves_uci, tuple) or len(self.moves_uci) != 2:
            raise ValueError("candidate must contain exactly two moves")
        for move in self.moves_uci:
            _validate_uci(move)
        _validate_points(self.changed_points, minimum=2)
        non_negative = (
            self.expected_change_floor,
            self.unexpected_difference,
            self.maximum_template_distance,
        )
        if any(not _is_number(value) or not isfinite(value) or value < 0 for value in non_negative):
            raise ValueError("candidate distances must be finite and non-negative")
        if not _is_number(self.minimum_template_margin) or not isfinite(
            self.minimum_template_margin
        ):
            raise ValueError("candidate template margin must be finite")
        if (
            not _is_number(self.minimum_template_confidence)
            or not isfinite(self.minimum_template_confidence)
            or not 0 <= self.minimum_template_confidence <= 1
        ):
            raise ValueError("candidate template evidence score must be between zero and one")
        if not _is_number(self.score) or not isfinite(self.score):
            raise ValueError("candidate score must be finite")
        _validate_position_id(self.final_position_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class HumanAiStageCSampleV1:
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
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_replay_fields(self)
        if self.schema_version != 1:
            raise ValueError("HumanAiStageCSampleV1 schema_version must be 1")


class _StageCReplayFields(Protocol):
    @property
    def sample_id(self) -> str: ...

    @property
    def session_id(self) -> str: ...

    @property
    def created_at_utc(self) -> str: ...

    @property
    def confirmed_fen(self) -> str: ...

    @property
    def confirmed_position_id(self) -> str: ...

    @property
    def expected_outcome(self) -> StageCExpectedOutcome: ...

    @property
    def scenario(self) -> StageCScenario: ...

    @property
    def ground_truth_moves_uci(self) -> tuple[str, ...]: ...

    @property
    def expected_final_position_id(self) -> str | None: ...

    @property
    def observed_status(self) -> StageCObservedStatus: ...

    @property
    def observed_moves_uci(self) -> tuple[str, ...]: ...

    @property
    def observed_final_position_id(self) -> str | None: ...

    @property
    def side_to_move(self) -> Side: ...

    @property
    def orientation(self) -> Orientation: ...

    @property
    def changed_points(self) -> tuple[int, ...]: ...

    @property
    def local_differences(self) -> tuple[float, ...]: ...

    @property
    def candidates(self) -> tuple[StageCCandidateRecord, ...]: ...

    @property
    def rejection_reasons(self) -> tuple[str, ...]: ...

    @property
    def capture_context(self) -> CaptureContext: ...

    @property
    def feature_version(self) -> str: ...

    @property
    def threshold_profile_version(self) -> str: ...

    @property
    def decision_latency_ms(self) -> float: ...


def _validate_replay_fields(sample: _StageCReplayFields) -> None:
    _validate_identifier(sample.sample_id)
    _validate_identifier(sample.session_id)
    _parse_utc(sample.created_at_utc)
    if not isinstance(sample.confirmed_fen, str) or not sample.confirmed_fen.strip():
        raise ValueError("confirmed_fen must be non-empty")
    _validate_position_id(sample.confirmed_position_id)
    if not isinstance(sample.expected_outcome, StageCExpectedOutcome):
        raise TypeError("expected_outcome must be a StageCExpectedOutcome")
    if not isinstance(sample.scenario, StageCScenario):
        raise TypeError("scenario must be a StageCScenario")
    _validate_move_tuple(sample.ground_truth_moves_uci, maximum=3)
    if sample.expected_outcome is StageCExpectedOutcome.ACCEPT:
        if len(sample.ground_truth_moves_uci) != 2:
            raise ValueError("accepted event must contain exactly two ground-truth moves")
        if sample.scenario is not StageCScenario.VALID_TWO_PLY:
            raise ValueError("accepted event must use the valid_two_ply scenario")
        if sample.expected_final_position_id is None:
            raise ValueError("accepted event must contain an expected final position")
        _validate_position_id(sample.expected_final_position_id)
    elif sample.expected_final_position_id is not None:
        raise ValueError("rejection event must not claim an expected final position")
    elif sample.scenario is StageCScenario.VALID_TWO_PLY:
        raise ValueError("rejection event must use a rejection scenario")

    if not isinstance(sample.observed_status, StageCObservedStatus):
        raise TypeError("observed_status must be a StageCObservedStatus")
    _validate_move_tuple(sample.observed_moves_uci, maximum=2)
    if sample.observed_status is StageCObservedStatus.ACCEPTED:
        if len(sample.observed_moves_uci) != 2 or sample.observed_final_position_id is None:
            raise ValueError("accepted observation must expose two moves and a final position")
        _validate_position_id(sample.observed_final_position_id)
        if sample.rejection_reasons:
            raise ValueError("accepted observation must not contain rejection reasons")
    else:
        if sample.observed_moves_uci:
            raise ValueError("rejected observation must not expose moves")
        if sample.observed_final_position_id != sample.confirmed_position_id:
            raise ValueError("rejected observation must keep the confirmed position")
        if not sample.rejection_reasons:
            raise ValueError("rejected observation must contain a rejection reason")

    if sample.side_to_move not in ("w", "b"):
        raise ValueError("side_to_move must be w or b")
    if not isinstance(sample.orientation, Orientation):
        raise TypeError("orientation must be an Orientation")
    _validate_points(sample.changed_points, minimum=1)
    if not isinstance(sample.local_differences, tuple) or len(sample.local_differences) != 90:
        raise ValueError("local_differences must contain exactly 90 values")
    if any(
        not _is_number(value) or not isfinite(value) or value < 0
        for value in sample.local_differences
    ):
        raise ValueError("local differences must be finite non-negative values")
    if not isinstance(sample.candidates, tuple) or any(
        not isinstance(candidate, StageCCandidateRecord) for candidate in sample.candidates
    ):
        raise TypeError("candidates must be a tuple of StageCCandidateRecord")
    if len(sample.candidates) > 2:
        raise ValueError("Stage C evidence keeps at most two candidates")
    ranked = tuple(
        sorted(
            sample.candidates,
            key=lambda candidate: (-candidate.score, candidate.moves_uci),
        )
    )
    if ranked != sample.candidates:
        raise ValueError("Stage C candidates must use deterministic ranked order")
    if not isinstance(sample.rejection_reasons, tuple) or any(
        not isinstance(reason, str) or not reason.strip() for reason in sample.rejection_reasons
    ):
        raise TypeError("rejection_reasons must be a tuple of non-empty strings")
    if len(set(sample.rejection_reasons)) != len(sample.rejection_reasons):
        raise ValueError("rejection reasons must not contain duplicates")
    if not isinstance(sample.capture_context, CaptureContext):
        raise TypeError("capture_context must be a CaptureContext")
    for version in (sample.feature_version, sample.threshold_profile_version):
        if not isinstance(version, str) or not version.strip():
            raise ValueError("feature and threshold profile versions must be non-empty")
    if (
        not _is_number(sample.decision_latency_ms)
        or not isfinite(sample.decision_latency_ms)
        or sample.decision_latency_ms < 0
    ):
        raise ValueError("decision_latency_ms must be finite and non-negative")


class HumanAiStageCSampleRecorder:
    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = False,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("sample root must be a Path")
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
        sample: HumanAiStageCSampleV1,
        crops: tuple[TransitionPointCrops, ...],
    ) -> Path:
        if not self._enabled:
            raise DiagnosticsDisabledError("Stage C diagnostics must be explicitly enabled")
        if not isinstance(sample, HumanAiStageCSampleV1):
            raise TypeError("sample must be a HumanAiStageCSampleV1")
        if not isinstance(crops, tuple) or any(
            not isinstance(crop, TransitionPointCrops) for crop in crops
        ):
            raise TypeError("crops must be a tuple of TransitionPointCrops")
        if tuple(crop.point_index for crop in crops) != sample.changed_points:
            raise ValueError("Stage C crops must match changed_points in stable order")
        final_dir = self._root / sample.session_id / sample.sample_id
        if final_dir.exists():
            raise FileExistsError(f"Stage C sample already exists: {sample.sample_id}")

        sample_time = _parse_utc(sample.created_at_utc)
        encoded = _encode_crops(crops)
        crop_hashes = {
            filename: sha256(contents).hexdigest() for filename, contents in encoded.items()
        }
        manifest_bytes = json.dumps(
            _manifest(sample, crop_hashes),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

        self.purge_expired(sample_time)
        added_bytes = len(manifest_bytes) + sum(len(contents) for contents in encoded.values())
        if _tree_size(self._root) + added_bytes > self._max_bytes:
            raise SampleQuotaExceededError("Stage C sample capacity would be exceeded")

        session_dir = final_dir.parent
        temporary_dir = session_dir / f".{sample.sample_id}.tmp-{uuid4().hex}"
        temporary_dir.mkdir(parents=True)
        try:
            for filename, contents in encoded.items():
                (temporary_dir / filename).write_bytes(contents)
            (temporary_dir / "manifest.json").write_bytes(manifest_bytes)
            temporary_dir.rename(final_dir)
        except BaseException:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
            if self._root.exists() and not any(self._root.iterdir()):
                self._root.rmdir()
            raise
        return final_dir

    def purge_expired(self, now: datetime) -> int:
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("purge clock must be a timezone-aware datetime")
        cutoff = now.astimezone(UTC) - timedelta(days=self._retention_days)
        if not self._root.exists():
            return 0
        removed = 0
        for session_dir in tuple(path for path in self._root.iterdir() if path.is_dir()):
            for sample_dir in tuple(path for path in session_dir.iterdir() if path.is_dir()):
                created_at = _sample_created_at(sample_dir)
                if created_at is not None and created_at < cutoff:
                    shutil.rmtree(sample_dir)
                    removed += 1
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
        return removed


def _validate_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("sample and session identifier must be path-safe")


def _validate_position_id(value: str) -> None:
    if not isinstance(value, str) or _POSITION_ID.fullmatch(value) is None:
        raise ValueError("position ids must be 32 lowercase hex characters")


def _validate_uci(value: str) -> None:
    if not isinstance(value, str) or _UCI.fullmatch(value) is None:
        raise ValueError("move must use four-character Xiangqi UCI coordinates")


def _validate_move_tuple(value: tuple[str, ...], *, maximum: int) -> None:
    if not isinstance(value, tuple) or len(value) > maximum:
        raise ValueError(f"move sequence must contain at most {maximum} moves")
    for move in value:
        _validate_uci(move)


def _validate_points(points: tuple[int, ...], *, minimum: int) -> None:
    if not isinstance(points, tuple) or not minimum <= len(points) <= 4:
        label = "one" if minimum == 1 else "two"
        raise ValueError(f"changed_points must contain {label} through four points")
    if any(
        isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < 90
        for point in points
    ):
        raise ValueError("changed_points must contain board indices from 0 through 89")
    if tuple(sorted(set(points))) != points:
        raise ValueError("changed_points must use unique stable ascending order")


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("created_at_utc must be a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _encode_crops(crops: tuple[TransitionPointCrops, ...]) -> dict[str, bytes]:
    encoded: dict[str, bytes] = {}
    for crop in crops:
        for suffix, value in (("before", crop.before), ("after", crop.after)):
            pixels = np.asarray(value)
            if pixels.dtype != np.uint8 or pixels.shape != (_CROP_SIZE, _CROP_SIZE, 4):
                raise ValueError("Stage C crops must be BGRA uint8 images of exactly 48x48 pixels")
            success, buffer = cv2.imencode(
                ".png",
                pixels,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not success:
                raise RuntimeError(f"failed to encode Stage C crop: {crop.point_index}-{suffix}")
            encoded[f"point-{crop.point_index:02d}-{suffix}.png"] = buffer.tobytes()
    return encoded


def _manifest(
    sample: HumanAiStageCSampleV1,
    crop_hashes: dict[str, str],
) -> dict[str, Any]:
    payload = asdict(sample)
    payload["expected_outcome"] = sample.expected_outcome.value
    payload["scenario"] = sample.scenario.value
    payload["observed_status"] = sample.observed_status.value
    payload["orientation"] = sample.orientation.value
    payload["crop_hashes"] = dict(sorted(crop_hashes.items()))
    return payload


def _tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _sample_created_at(sample_dir: Path) -> datetime | None:
    manifest_path = sample_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        value = payload["created_at_utc"]
        if not isinstance(value, str):
            return None
        return _parse_utc(value)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
