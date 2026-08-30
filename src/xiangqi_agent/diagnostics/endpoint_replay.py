from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Protocol, cast

import cv2
import numpy as np

from xiangqi_agent.capture.context import CaptureContext
from xiangqi_agent.diagnostics.endpoint_samples import (
    EndpointCrops,
    EndpointSampleV1,
    SampleKind,
)
from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.vision.endpoint_features import EndpointFeatureExtractor, EndpointFeatures

_CROP_FILES = (
    "source_before.png",
    "source_after.png",
    "target_before.png",
    "target_after.png",
)
_EXPECTED_FILES = frozenset((*_CROP_FILES, "manifest.json"))


class SampleIntegrityError(ValueError):
    """A persisted endpoint sample is incomplete, changed, or malformed."""


@dataclass(frozen=True, slots=True)
class LoadedEndpointSample:
    metadata: EndpointSampleV1
    crops: EndpointCrops
    directory: Path


@dataclass(frozen=True, slots=True)
class ReplayThresholds:
    max_instance_distance: float
    min_source_change: float
    min_target_change: float
    profile_version: str

    def __post_init__(self) -> None:
        values = (
            self.max_instance_distance,
            self.min_source_change,
            self.min_target_change,
        )
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError("replay thresholds must be finite and non-negative")
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise ValueError("profile_version must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ReplayDecision:
    accepted: bool
    rejection_reasons: tuple[str, ...]


class ReplayGate(Protocol):
    @property
    def profile_version(self) -> str: ...

    def evaluate(self, features: EndpointFeatures) -> ReplayDecision: ...


class EndpointReplayGate:
    def __init__(self, thresholds: ReplayThresholds) -> None:
        self._thresholds = thresholds

    @property
    def profile_version(self) -> str:
        return self._thresholds.profile_version

    def evaluate(self, features: EndpointFeatures) -> ReplayDecision:
        reasons: list[str] = []
        if features.instance_distance > self._thresholds.max_instance_distance:
            reasons.append("instance_distance")
        if features.source_change_distance < self._thresholds.min_source_change:
            reasons.append("source_change")
        if features.target_change_distance < self._thresholds.min_target_change:
            reasons.append("target_change")
        return ReplayDecision(not reasons, tuple(reasons))


@dataclass(frozen=True, slots=True)
class EndpointReplayResult:
    sample_id: str
    actual_uci: str | None
    probe_uci: str
    feature_version: str
    threshold_profile_version: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    features: EndpointFeatures
    result_fen: str | None
    runtime_ns: int

    def without_runtime(self) -> tuple[object, ...]:
        return (
            self.sample_id,
            self.actual_uci,
            self.probe_uci,
            self.feature_version,
            self.threshold_profile_version,
            self.accepted,
            self.rejection_reasons,
            self.features,
            self.result_fen,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "actual_uci": self.actual_uci,
            "probe_uci": self.probe_uci,
            "feature_version": self.feature_version,
            "threshold_profile_version": self.threshold_profile_version,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "features": asdict(self.features),
            "result_fen": self.result_fen,
            "runtime_ns": self.runtime_ns,
        }


class EndpointSampleLoader:
    def load(self, sample_dir: Path) -> LoadedEndpointSample:
        if not isinstance(sample_dir, Path):
            raise TypeError("sample directory must be a Path")
        if not sample_dir.is_dir():
            raise SampleIntegrityError("endpoint sample directory does not exist")
        actual_files = frozenset(path.name for path in sample_dir.iterdir() if path.is_file())
        if actual_files != _EXPECTED_FILES:
            raise SampleIntegrityError("sample must contain exactly four endpoint crops and a manifest")

        payload = _read_manifest(sample_dir / "manifest.json")
        hashes = _string_mapping(payload.get("crop_hashes"), "crop_hashes")
        if frozenset(hashes) != frozenset(_CROP_FILES):
            raise SampleIntegrityError("manifest crop hash names do not match endpoint crops")
        encoded: dict[str, bytes] = {}
        for filename in _CROP_FILES:
            contents = (sample_dir / filename).read_bytes()
            if sha256(contents).hexdigest() != hashes[filename]:
                raise SampleIntegrityError(f"endpoint crop hash mismatch: {filename}")
            encoded[filename] = contents

        crops = EndpointCrops(
            source_before=_decode_crop(encoded["source_before.png"], "source_before.png"),
            source_after=_decode_crop(encoded["source_after.png"], "source_after.png"),
            target_before=_decode_crop(encoded["target_before.png"], "target_before.png"),
            target_after=_decode_crop(encoded["target_after.png"], "target_after.png"),
        )
        metadata = _metadata_from_payload(payload)
        if parse_fen(metadata.confirmed_fen).position_id != metadata.confirmed_position_id:
            raise SampleIntegrityError("confirmed FEN does not match confirmed_position_id")
        return LoadedEndpointSample(metadata, crops, sample_dir)


class EndpointReplayer:
    def __init__(
        self,
        extractor: EndpointFeatureExtractor,
        gate: ReplayGate,
        *,
        loader: EndpointSampleLoader | None = None,
    ) -> None:
        self._extractor = extractor
        self._gate = gate
        self._loader = loader or EndpointSampleLoader()

    def replay(self, sample_dir: Path) -> EndpointReplayResult:
        started_ns = perf_counter_ns()
        loaded = self._loader.load(sample_dir)
        features = self._extractor.extract(loaded.crops)
        decision = self._gate.evaluate(features)
        accepted = decision.accepted
        reasons = list(decision.rejection_reasons)
        result_fen: str | None = None
        if accepted:
            board = parse_fen(loaded.metadata.confirmed_fen)
            move = next(
                (move for move in legal_moves(board) if move.uci == loaded.metadata.probe_uci),
                None,
            )
            if move is None:
                accepted = False
                reasons.append("probe_move_not_legal")
            else:
                result_fen = apply_move(board, move).fen
        return EndpointReplayResult(
            sample_id=loaded.metadata.sample_id,
            actual_uci=loaded.metadata.actual_uci,
            probe_uci=loaded.metadata.probe_uci,
            feature_version=features.feature_version,
            threshold_profile_version=self._gate.profile_version,
            accepted=accepted,
            rejection_reasons=tuple(reasons),
            features=features,
            result_fen=result_fen,
            runtime_ns=perf_counter_ns() - started_ns,
        )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SampleIntegrityError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SampleIntegrityError("manifest root must be an object")
    return cast(dict[str, Any], value)


def _string_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise SampleIntegrityError(f"{name} must be a string mapping")
    return cast(dict[str, str], value)


def _decode_crop(contents: bytes, filename: str) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8 or image.shape != (48, 48, 4):
        raise SampleIntegrityError(f"endpoint crop is not a 48x48 BGRA PNG: {filename}")
    return np.asarray(image, dtype=np.uint8)


def _metadata_from_payload(payload: dict[str, Any]) -> EndpointSampleV1:
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
        top_k = tuple(
            cast(dict[str, str | float], item)
            for item in cast(list[object], payload["top_k_candidates"])
            if isinstance(item, dict)
        )
        changes = {
            str(key): _float_value(value)
            for key, value in cast(dict[object, object], payload["change_scores"]).items()
        }
        actual_value = payload["actual_uci"]
        return EndpointSampleV1(
            sample_id=str(payload["sample_id"]),
            session_id=str(payload["session_id"]),
            sample_kind=SampleKind(str(payload["sample_kind"])),
            created_at_utc=str(payload["created_at_utc"]),
            confirmed_fen=str(payload["confirmed_fen"]),
            confirmed_position_id=str(payload["confirmed_position_id"]),
            actual_uci=None if actual_value is None else str(actual_value),
            probe_uci=str(payload["probe_uci"]),
            side_to_move=cast(Any, payload["side_to_move"]),
            orientation=Orientation(str(payload["orientation"])),
            source_index=int(payload["source_index"]),
            target_index=int(payload["target_index"]),
            top_k_candidates=top_k,
            rejection_reasons=tuple(str(value) for value in payload["rejection_reasons"]),
            capture_context=context,
            feature_version=str(payload["feature_version"]),
            threshold_profile_version=str(payload["threshold_profile_version"]),
            change_scores=changes,
            schema_version=int(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleIntegrityError("manifest metadata is invalid") from exc


def _size(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise SampleIntegrityError("capture size must contain two integers")
    return int(value[0]), int(value[1])


def _float_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise SampleIntegrityError("numeric manifest value is invalid")
    return float(value)
