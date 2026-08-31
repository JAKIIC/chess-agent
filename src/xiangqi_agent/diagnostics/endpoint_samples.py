from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.domain.board import Orientation, Side

_CROP_SIZE = 48
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_RETENTION_DAYS = 7
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UCI = re.compile(r"^[a-i][0-9][a-i][0-9]$")
_CROP_NAMES = (
    "source_before",
    "source_after",
    "target_before",
    "target_after",
)


class DiagnosticsDisabledError(RuntimeError):
    """Endpoint persistence was not explicitly enabled."""


class SampleQuotaExceededError(RuntimeError):
    """Writing another sample would exceed the configured capacity."""


class SampleKind(StrEnum):
    MOVE = "move"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EndpointCrops:
    source_before: NDArray[np.uint8]
    source_after: NDArray[np.uint8]
    target_before: NDArray[np.uint8]
    target_after: NDArray[np.uint8]


@dataclass(frozen=True, slots=True, kw_only=True)
class EndpointSampleV1:
    sample_id: str
    session_id: str
    sample_kind: SampleKind
    created_at_utc: str
    confirmed_fen: str
    confirmed_position_id: str
    actual_uci: str | None
    probe_uci: str
    side_to_move: Side
    orientation: Orientation
    source_index: int
    target_index: int
    top_k_candidates: tuple[dict[str, str | float], ...]
    rejection_reasons: tuple[str, ...]
    capture_context: CaptureContext
    feature_version: str
    threshold_profile_version: str
    change_scores: dict[str, float]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id)
        _validate_identifier(self.session_id)
        if not isinstance(self.sample_kind, SampleKind):
            raise TypeError("sample_kind must be a SampleKind")
        if not isinstance(self.created_at_utc, str) or not self.created_at_utc.endswith("Z"):
            raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
        if not isinstance(self.confirmed_fen, str) or not self.confirmed_fen.strip():
            raise ValueError("confirmed_fen must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{32}", self.confirmed_position_id):
            raise ValueError("confirmed_position_id must be 32 lowercase hex characters")
        _validate_uci(self.probe_uci)
        if self.actual_uci is not None:
            _validate_uci(self.actual_uci)
        if self.side_to_move not in ("w", "b"):
            raise ValueError("side_to_move must be w or b")
        if not isinstance(self.orientation, Orientation):
            raise TypeError("orientation must be an Orientation")
        for index in (self.source_index, self.target_index):
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 90:
                raise ValueError("endpoint indices must be integers from 0 through 89")
        if self.source_index == self.target_index:
            raise ValueError("source and target indices must differ")
        if not isinstance(self.top_k_candidates, tuple):
            raise TypeError("top_k_candidates must be a tuple")
        if not isinstance(self.rejection_reasons, tuple):
            raise TypeError("rejection_reasons must be a tuple")
        if not isinstance(self.capture_context, CaptureContext):
            raise TypeError("capture_context must be a CaptureContext")
        if not self.feature_version.strip() or not self.threshold_profile_version.strip():
            raise ValueError("feature and threshold profile versions must be non-empty")
        if self.schema_version != 1:
            raise ValueError("EndpointSampleV1 schema_version must be 1")


class EndpointSampleRecorder:
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

    def record(self, sample: EndpointSampleV1, crops: EndpointCrops) -> Path:
        if not self._enabled:
            raise DiagnosticsDisabledError("endpoint diagnostics must be explicitly enabled")
        if not isinstance(sample, EndpointSampleV1):
            raise TypeError("sample must be an EndpointSampleV1")
        if not isinstance(crops, EndpointCrops):
            raise TypeError("crops must be EndpointCrops")
        sample_time = datetime.fromisoformat(
            f"{sample.created_at_utc[:-1]}+00:00"
        ).astimezone(UTC)

        encoded = _encode_crops(crops)
        crop_hashes = {
            f"{name}.png": sha256(contents).hexdigest() for name, contents in encoded.items()
        }
        manifest = _manifest(sample, crop_hashes)
        manifest_bytes = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        self.purge_expired(sample_time)
        added_bytes = len(manifest_bytes) + sum(len(contents) for contents in encoded.values())
        if _tree_size(self._root) + added_bytes > self._max_bytes:
            raise SampleQuotaExceededError("endpoint sample capacity would be exceeded")

        session_dir = self._root / sample.session_id
        final_dir = session_dir / sample.sample_id
        if final_dir.exists():
            raise FileExistsError(f"endpoint sample already exists: {sample.sample_id}")
        temporary_dir = session_dir / f".{sample.sample_id}.tmp-{uuid4().hex}"
        temporary_dir.mkdir(parents=True)
        try:
            for name, contents in encoded.items():
                (temporary_dir / f"{name}.png").write_bytes(contents)
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

    def delete_session(self, session_id: str) -> int:
        _validate_identifier(session_id)
        session_dir = self._root / session_id
        removed = _delete_sample_directories(session_dir)
        if session_dir.exists() and not any(session_dir.iterdir()):
            session_dir.rmdir()
        return removed

    def delete_all(self) -> int:
        if not self._root.exists():
            return 0
        removed = 0
        for session_dir in tuple(path for path in self._root.iterdir() if path.is_dir()):
            removed += _delete_sample_directories(session_dir)
            if session_dir.exists() and not any(session_dir.iterdir()):
                session_dir.rmdir()
        return removed

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


def _encode_crops(crops: EndpointCrops) -> dict[str, bytes]:
    encoded: dict[str, bytes] = {}
    for name in _CROP_NAMES:
        pixels = np.asarray(getattr(crops, name))
        if pixels.dtype != np.uint8 or pixels.shape != (_CROP_SIZE, _CROP_SIZE, 4):
            raise ValueError("endpoint crops must be BGRA uint8 images of exactly 48x48 pixels")
        success, buffer = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not success:
            raise RuntimeError(f"failed to encode endpoint crop: {name}")
        encoded[name] = buffer.tobytes()
    return encoded


def _manifest(sample: EndpointSampleV1, crop_hashes: dict[str, str]) -> dict[str, Any]:
    payload = asdict(sample)
    payload["sample_kind"] = sample.sample_kind.value
    payload["orientation"] = sample.orientation.value
    payload["crop_hashes"] = crop_hashes
    return payload


def _tree_size(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _delete_sample_directories(session_dir: Path) -> int:
    if not session_dir.exists() or not session_dir.is_dir():
        return 0
    samples = tuple(
        path
        for path in session_dir.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    for sample_dir in samples:
        shutil.rmtree(sample_dir)
    return len(samples)


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
