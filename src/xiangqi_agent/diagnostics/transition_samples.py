from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.domain.board import Orientation, Side

_CROP_SIZE = 48
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_RETENTION_DAYS = 7
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_POSITION_ID = re.compile(r"^[0-9a-f]{32}$")
_UCI = re.compile(r"^[a-i][0-9][a-i][0-9]$")


@dataclass(frozen=True, slots=True)
class TransitionPointCrops:
    point_index: int
    before: NDArray[np.uint8]
    after: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if (
            isinstance(self.point_index, bool)
            or not isinstance(self.point_index, int)
            or not 0 <= self.point_index < 90
        ):
            raise ValueError("point_index must be an integer from 0 through 89")


@dataclass(frozen=True, slots=True, kw_only=True)
class TransitionSampleV2:
    sample_id: str
    session_id: str
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
    final_position_id: str
    moves_uci: tuple[str, str]
    side_to_move: Side
    orientation: Orientation
    changed_points: tuple[int, ...]
    capture_context: CaptureContext
    feature_version: str
    threshold_profile_version: str
    rejection_reasons: tuple[str, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id)
        _validate_identifier(self.session_id)
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
        if not isinstance(self.confirmed_fen, str) or not self.confirmed_fen.strip():
            raise ValueError("confirmed_fen must be non-empty")
        for position_id in (self.confirmed_position_id, self.final_position_id):
            if not isinstance(position_id, str) or _POSITION_ID.fullmatch(position_id) is None:
                raise ValueError("position ids must be 32 lowercase hex characters")
        if not isinstance(self.moves_uci, tuple) or len(self.moves_uci) != 2:
            raise ValueError("moves_uci must contain exactly two moves")
        for move in self.moves_uci:
            _validate_uci(move)
        if self.side_to_move not in ("w", "b"):
            raise ValueError("side_to_move must be w or b")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation must be an Orientation")
        if not isinstance(self.changed_points, tuple) or not 2 <= len(self.changed_points) <= 4:
            raise ValueError("changed_points must contain two through four points")
        if any(
            isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < 90
            for point in self.changed_points
        ):
            raise ValueError("changed_points must contain board indices from 0 through 89")
        if tuple(sorted(set(self.changed_points))) != self.changed_points:
            raise ValueError("changed_points must use unique stable ascending order")
        if not isinstance(self.capture_context, CaptureContext):
            raise TypeError("capture_context must be a CaptureContext")
        if not isinstance(self.rejection_reasons, tuple):
            raise TypeError("rejection_reasons must be a tuple")
        if not self.feature_version.strip() or not self.threshold_profile_version.strip():
            raise ValueError("feature and threshold profile versions must be non-empty")
        if self.schema_version != 2:
            raise ValueError("TransitionSampleV2 schema_version must be 2")


class TransitionSampleRecorder:
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
        sample: TransitionSampleV2,
        crops: tuple[TransitionPointCrops, ...],
    ) -> Path:
        if not self._enabled:
            raise DiagnosticsDisabledError("transition diagnostics must be explicitly enabled")
        if not isinstance(sample, TransitionSampleV2):
            raise TypeError("sample must be a TransitionSampleV2")
        if not isinstance(crops, tuple) or any(
            not isinstance(crop, TransitionPointCrops) for crop in crops
        ):
            raise TypeError("crops must be a tuple of TransitionPointCrops")
        if tuple(crop.point_index for crop in crops) != sample.changed_points:
            raise ValueError("transition crops must match changed_points in stable order")
        sample_time = datetime.fromisoformat(
            f"{sample.created_at_utc[:-1]}+00:00"
        ).astimezone(UTC)
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
            raise SampleQuotaExceededError("transition sample capacity would be exceeded")

        session_dir = self._root / sample.session_id
        final_dir = session_dir / sample.sample_id
        if final_dir.exists():
            raise FileExistsError(f"transition sample already exists: {sample.sample_id}")
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


def _validate_uci(value: str) -> None:
    if not isinstance(value, str) or _UCI.fullmatch(value) is None:
        raise ValueError("move must use four-character Xiangqi UCI coordinates")


def _encode_crops(crops: tuple[TransitionPointCrops, ...]) -> dict[str, bytes]:
    encoded: dict[str, bytes] = {}
    for crop in crops:
        for suffix, value in (("before", crop.before), ("after", crop.after)):
            pixels = np.asarray(value)
            if pixels.dtype != np.uint8 or pixels.shape != (_CROP_SIZE, _CROP_SIZE, 4):
                raise ValueError(
                    "transition crops must be BGRA uint8 images of exactly 48x48 pixels"
                )
            success, buffer = cv2.imencode(
                ".png",
                pixels,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            )
            if not success:
                raise RuntimeError(
                    f"failed to encode transition crop: {crop.point_index}-{suffix}"
                )
            encoded[f"point-{crop.point_index:02d}-{suffix}.png"] = buffer.tobytes()
    return encoded


def _manifest(sample: TransitionSampleV2, crop_hashes: dict[str, str]) -> dict[str, Any]:
    payload = asdict(sample)
    payload["orientation"] = sample.orientation.value
    payload["crop_hashes"] = crop_hashes
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
        if not isinstance(value, str) or not value.endswith("Z"):
            return None
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return parsed.astimezone(UTC)
