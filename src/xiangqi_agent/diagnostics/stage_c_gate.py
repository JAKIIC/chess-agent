from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path, PurePosixPath
from typing import Any, cast

from xiangqi_agent.diagnostics.stage_c_replay import (
    HumanAiStageCReplayer,
    HumanAiStageCReplayResult,
    HumanAiStageCSampleLoader,
    StageCSampleIntegrityError,
)
from xiangqi_agent.diagnostics.stage_c_review import StageCReviewOutcome
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import (
    ReviewedStageCSampleIntegrityError,
    ReviewedStageCSampleLoader,
    ReviewedStageCSampleV2,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCExpectedOutcome,
    StageCScenario,
)
from xiangqi_agent.sync.sequence_gate import (
    SequenceDecisionGate,
    SequenceThresholdProfile,
)

_MIN_VALID_SAMPLES = 30
_MIN_DISTINCT_VALID_SESSIONS = 30
_MIN_REJECTION_SAMPLES = 30
_MIN_COVERAGE = 0.80
_MAX_DECISION_P95_MS = 500.0
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_REJECTION_SCENARIOS = (
    StageCScenario.MULTIPLE_CANDIDATES,
    StageCScenario.SELECTION_HIGHLIGHT,
    StageCScenario.CONTINUOUS_ANIMATION,
    StageCScenario.OCCLUSION,
    StageCScenario.RESIZE,
    StageCScenario.THREE_PLY,
)

DEFAULT_STAGE_C_FEATURE_VERSION = "two-ply-template-v3"
DEFAULT_STAGE_C_THRESHOLD_PROFILE = SequenceThresholdProfile(
    min_local_difference=5.0,
    max_unexpected_difference=3.0,
    min_score=5.0,
    min_margin=5.0,
    max_template_distance=0.18,
    min_template_margin=0.02,
    min_template_confidence=0.8,
    profile_version="human-ai-two-ply-v1",
)


class StageCGateIntegrityError(ValueError):
    """Frozen Stage C input is malformed, changed, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class FrozenStageCSampleV1:
    sample_id: str
    relative_path: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or _IDENTIFIER.fullmatch(self.sample_id) is None:
            raise StageCGateIntegrityError("frozen sample id is not path-safe")
        _validate_relative_sample_path(self.relative_path)
        if (
            not isinstance(self.manifest_sha256, str)
            or _SHA256.fullmatch(self.manifest_sha256) is None
        ):
            raise StageCGateIntegrityError("frozen sample manifest hash is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "manifest_sha256": self.manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class FrozenStageCManifestV1:
    created_at_utc: str
    feature_version: str
    threshold_profile: SequenceThresholdProfile
    samples: tuple[FrozenStageCSampleV1, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _parse_utc(self.created_at_utc)
        if not isinstance(self.feature_version, str) or not self.feature_version.strip():
            raise StageCGateIntegrityError("frozen feature version must be non-empty")
        if not isinstance(self.threshold_profile, SequenceThresholdProfile):
            raise TypeError("threshold_profile must be a SequenceThresholdProfile")
        if not isinstance(self.samples, tuple) or not self.samples:
            raise StageCGateIntegrityError("frozen Stage C manifest must contain samples")
        if any(not isinstance(sample, FrozenStageCSampleV1) for sample in self.samples):
            raise TypeError("frozen samples must be FrozenStageCSampleV1 values")
        paths = tuple(sample.relative_path for sample in self.samples)
        identifiers = tuple(sample.sample_id for sample in self.samples)
        if paths != tuple(sorted(paths)):
            raise StageCGateIntegrityError("frozen sample paths must use sorted stable order")
        if len(set(paths)) != len(paths):
            raise StageCGateIntegrityError("frozen sample paths must be unique")
        if len(set(identifiers)) != len(identifiers):
            raise StageCGateIntegrityError("frozen sample ids must be unique")
        if self.schema_version != 1:
            raise StageCGateIntegrityError("FrozenStageCManifestV1 schema_version must be 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "created_at_utc": self.created_at_utc,
            "feature_version": self.feature_version,
            "threshold_profile": asdict(self.threshold_profile),
            "samples": [sample.to_dict() for sample in self.samples],
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class HumanAiStageCMetrics:
    total_samples: int
    valid_samples: int
    rejection_samples: int
    distinct_valid_sessions: int
    scenario_counts: tuple[tuple[str, int], ...]
    review_outcome_counts: tuple[tuple[str, int], ...]
    accepted_samples: int
    correct_accepts: int
    false_accepts: int
    correct_rejects: int
    missed_valid: int
    recorded_consistency_failures: int
    coverage: float
    accepted_precision: float
    final_position_accuracy: float
    p95_decision_latency_ms: float
    p95_replay_runtime_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "rejection_samples": self.rejection_samples,
            "distinct_valid_sessions": self.distinct_valid_sessions,
            "scenario_counts": dict(self.scenario_counts),
            "review_outcome_counts": dict(self.review_outcome_counts),
            "accepted_samples": self.accepted_samples,
            "correct_accepts": self.correct_accepts,
            "false_accepts": self.false_accepts,
            "correct_rejects": self.correct_rejects,
            "missed_valid": self.missed_valid,
            "recorded_consistency_failures": self.recorded_consistency_failures,
            "coverage": self.coverage,
            "accepted_precision": self.accepted_precision,
            "final_position_accuracy": self.final_position_accuracy,
            "p95_decision_latency_ms": self.p95_decision_latency_ms,
            "p95_replay_runtime_ms": self.p95_replay_runtime_ms,
        }


@dataclass(frozen=True, slots=True)
class HumanAiStageCReport:
    release_pass: bool
    reasons: tuple[str, ...]
    metrics: HumanAiStageCMetrics
    feature_version: str
    threshold_profile_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "release_pass": self.release_pass,
            "reasons": list(self.reasons),
            "metrics": self.metrics.to_dict(),
            "feature_version": self.feature_version,
            "threshold_profile_version": self.threshold_profile_version,
        }


class HumanAiStageCGate:
    def evaluate(self, manifest_path: Path) -> HumanAiStageCReport:
        if not isinstance(manifest_path, Path):
            raise TypeError("frozen manifest path must be a Path")
        manifest = load_frozen_stage_c_manifest(manifest_path)
        sample_root = manifest_path.parent.resolve()
        replayer = HumanAiStageCReplayer(
            SequenceDecisionGate(manifest.threshold_profile),
            feature_version=manifest.feature_version,
        )
        results: list[HumanAiStageCReplayResult] = []
        for entry in manifest.samples:
            sample_dir = _resolve_sample_dir(sample_root, entry.relative_path)
            sample_manifest_path = sample_dir / "manifest.json"
            try:
                contents = sample_manifest_path.read_bytes()
            except OSError as exc:
                raise StageCGateIntegrityError(
                    f"frozen sample manifest cannot be read: {entry.sample_id}"
                ) from exc
            if sha256(contents).hexdigest() != entry.manifest_sha256:
                raise StageCGateIntegrityError(
                    f"frozen sample manifest hash changed: {entry.sample_id}"
                )
            try:
                result = replayer.replay(sample_dir)
            except StageCSampleIntegrityError as exc:
                raise StageCGateIntegrityError(
                    f"frozen sample failed integrity checks: {entry.sample_id}"
                ) from exc
            if result.sample_id != entry.sample_id:
                raise StageCGateIntegrityError("frozen sample id does not match its manifest")
            if PurePosixPath(entry.relative_path).parts != (
                result.session_id,
                result.sample_id,
            ):
                raise StageCGateIntegrityError(
                    "frozen sample path does not match its anonymous ids"
                )
            results.append(result)

        valid = tuple(
            result
            for result in results
            if result.expected_outcome is StageCExpectedOutcome.ACCEPT
        )
        rejected = tuple(
            result
            for result in results
            if result.expected_outcome is StageCExpectedOutcome.REJECT
        )
        report = evaluate_stage_c_results(valid, rejected)
        if (
            report.feature_version != manifest.feature_version
            or report.threshold_profile_version
            != manifest.threshold_profile.profile_version
        ):
            raise StageCGateIntegrityError("replay report versions changed from frozen manifest")
        return report


def freeze_human_ai_stage_c(
    sample_root: Path,
    output_name: str,
    *,
    feature_version: str = DEFAULT_STAGE_C_FEATURE_VERSION,
    threshold_profile: SequenceThresholdProfile = DEFAULT_STAGE_C_THRESHOLD_PROFILE,
    created_at_utc: str | None = None,
) -> Path:
    if not isinstance(sample_root, Path):
        raise TypeError("sample_root must be a Path")
    if not sample_root.is_dir():
        raise StageCGateIntegrityError("Stage C sample root does not exist")
    if not isinstance(output_name, str):
        raise TypeError("output_name must be a string")
    _validate_output_name(output_name)
    if not isinstance(feature_version, str) or not feature_version.strip():
        raise StageCGateIntegrityError("feature version must be non-empty")
    if not isinstance(threshold_profile, SequenceThresholdProfile):
        raise TypeError("threshold_profile must be a SequenceThresholdProfile")

    output_path = sample_root / output_name
    if output_path.exists():
        raise StageCGateIntegrityError("frozen manifest output already exists")
    root = sample_root.resolve()
    loader = HumanAiStageCSampleLoader()
    entries: list[FrozenStageCSampleV1] = []
    manifest_paths = tuple(sorted(sample_root.rglob("manifest.json")))
    if not manifest_paths:
        raise StageCGateIntegrityError("Stage C sample root contains no samples")
    for sample_manifest_path in manifest_paths:
        sample_dir = sample_manifest_path.parent.resolve()
        try:
            relative = sample_dir.relative_to(root)
        except ValueError as exc:
            raise StageCGateIntegrityError("sample path escapes Stage C sample root") from exc
        relative_path = PurePosixPath(*relative.parts).as_posix()
        _validate_relative_sample_path(relative_path)
        try:
            loaded = loader.load(sample_dir)
        except StageCSampleIntegrityError as exc:
            raise StageCGateIntegrityError("sample failed integrity checks before freeze") from exc
        metadata = loaded.metadata
        if relative.parts != (metadata.session_id, metadata.sample_id):
            raise StageCGateIntegrityError(
                "sample directory does not match its anonymous session and sample ids"
            )
        if metadata.feature_version != feature_version:
            raise StageCGateIntegrityError("samples mix or differ from the frozen feature version")
        if metadata.threshold_profile_version != threshold_profile.profile_version:
            raise StageCGateIntegrityError(
                "samples mix or differ from the frozen threshold profile"
            )
        entries.append(
            FrozenStageCSampleV1(
                sample_id=metadata.sample_id,
                relative_path=relative_path,
                manifest_sha256=sha256(sample_manifest_path.read_bytes()).hexdigest(),
            )
        )

    entries.sort(key=lambda entry: entry.relative_path)
    frozen = FrozenStageCManifestV1(
        created_at_utc=created_at_utc or _utc_now(),
        feature_version=feature_version,
        threshold_profile=threshold_profile,
        samples=tuple(entries),
    )
    payload = json.dumps(
        frozen.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
    except FileExistsError as exc:
        raise StageCGateIntegrityError("frozen manifest output already exists") from exc
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def freeze_reviewed_human_ai_stage_c(
    reviewed_root: Path,
    output_name: str,
    *,
    feature_version: str = DEFAULT_STAGE_C_FEATURE_VERSION,
    threshold_profile: SequenceThresholdProfile = DEFAULT_STAGE_C_THRESHOLD_PROFILE,
    created_at_utc: str | None = None,
) -> Path:
    if not isinstance(reviewed_root, Path):
        raise TypeError("reviewed_root must be a Path")
    if (
        reviewed_root.name != "stage-c-reviewed"
        or reviewed_root.is_symlink()
        or not reviewed_root.is_dir()
    ):
        raise StageCGateIntegrityError(
            "reviewed-only freeze requires a real stage-c-reviewed root"
        )
    if not isinstance(output_name, str):
        raise TypeError("output_name must be a string")
    _validate_output_name(output_name)
    if not isinstance(feature_version, str) or not feature_version.strip():
        raise StageCGateIntegrityError("feature version must be non-empty")
    if not isinstance(threshold_profile, SequenceThresholdProfile):
        raise TypeError("threshold_profile must be a SequenceThresholdProfile")
    output_path = reviewed_root / output_name
    if output_path.exists() or output_path.is_symlink():
        raise StageCGateIntegrityError("frozen manifest output already exists")

    root = reviewed_root.resolve()
    manifest_paths = tuple(sorted(reviewed_root.rglob("manifest.json")))
    if not manifest_paths:
        raise StageCGateIntegrityError("stage-c-reviewed contains no V2 samples")
    for path in manifest_paths:
        try:
            relative = path.resolve().relative_to(root)
        except ValueError as exc:
            raise StageCGateIntegrityError(
                "reviewed manifest path escapes the reviewed root"
            ) from exc
        if len(relative.parts) != 3 or relative.name != "manifest.json":
            raise StageCGateIntegrityError(
                "reviewed manifest has an unknown placement"
            )
    root_entries = tuple(reviewed_root.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in root_entries):
        raise StageCGateIntegrityError(
            "stage-c-reviewed root has an unknown layout entry"
        )
    for session_dir in root_entries:
        sample_dirs = tuple(session_dir.iterdir())
        if not sample_dirs or any(
            path.is_symlink() or not path.is_dir() for path in sample_dirs
        ):
            raise StageCGateIntegrityError(
                "stage-c-reviewed root has an unknown layout entry"
            )
        if any(
            (sample_dir / "manifest.json").is_symlink()
            or not (sample_dir / "manifest.json").is_file()
            for sample_dir in sample_dirs
        ):
            raise StageCGateIntegrityError(
                "stage-c-reviewed root has an unknown layout entry"
            )

    loader = ReviewedStageCSampleLoader()
    replayer = HumanAiStageCReplayer(
        SequenceDecisionGate(threshold_profile),
        feature_version=feature_version,
    )
    entries: list[FrozenStageCSampleV1] = []
    for manifest_path in manifest_paths:
        sample_dir = manifest_path.parent
        try:
            loaded = loader.load(sample_dir)
            replayed = replayer.replay(sample_dir)
        except (ReviewedStageCSampleIntegrityError, StageCSampleIntegrityError) as exc:
            raise StageCGateIntegrityError(
                "reviewed V2 provenance or deterministic replay failed before freeze"
            ) from exc
        metadata = loaded.metadata
        if not isinstance(metadata, ReviewedStageCSampleV2):
            raise StageCGateIntegrityError("reviewed-only freeze accepts only V2 samples")
        relative = sample_dir.resolve().relative_to(root)
        if relative.parts != (metadata.session_id, metadata.sample_id):
            raise StageCGateIntegrityError(
                "reviewed V2 directory does not match its anonymous ids"
            )
        if replayed.sample_id != metadata.sample_id:
            raise StageCGateIntegrityError("reviewed V2 replay changed its sample id")
        if metadata.feature_version != feature_version:
            raise StageCGateIntegrityError(
                "reviewed V2 samples mix or differ from the frozen feature version"
            )
        if metadata.threshold_profile_version != threshold_profile.profile_version:
            raise StageCGateIntegrityError(
                "reviewed V2 samples mix or differ from the frozen threshold profile"
            )
        relative_path = PurePosixPath(*relative.parts).as_posix()
        entries.append(
            FrozenStageCSampleV1(
                metadata.sample_id,
                relative_path,
                sha256(manifest_path.read_bytes()).hexdigest(),
            )
        )

    entries.sort(key=lambda entry: entry.relative_path)
    frozen = FrozenStageCManifestV1(
        created_at_utc=created_at_utc or _utc_now(),
        feature_version=feature_version,
        threshold_profile=threshold_profile,
        samples=tuple(entries),
    )
    payload = json.dumps(
        frozen.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
    except FileExistsError as exc:
        raise StageCGateIntegrityError("frozen manifest output already exists") from exc
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return output_path


def load_frozen_stage_c_manifest(path: Path) -> FrozenStageCManifestV1:
    if not isinstance(path, Path):
        raise TypeError("frozen manifest path must be a Path")
    try:
        value = json.loads(path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageCGateIntegrityError("frozen manifest is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StageCGateIntegrityError("frozen manifest root must be an object")
    payload = cast(dict[str, Any], value)
    required = {
        "created_at_utc",
        "feature_version",
        "threshold_profile",
        "samples",
        "schema_version",
    }
    if set(payload) != required:
        raise StageCGateIntegrityError("frozen manifest fields are incomplete or unexpected")
    try:
        profile = _threshold_profile(payload["threshold_profile"])
        raw_samples = payload["samples"]
        if not isinstance(raw_samples, list):
            raise TypeError("samples must be a list")
        samples = tuple(_frozen_sample(item) for item in raw_samples)
        return FrozenStageCManifestV1(
            created_at_utc=str(payload["created_at_utc"]),
            feature_version=str(payload["feature_version"]),
            threshold_profile=profile,
            samples=samples,
            schema_version=_integer(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, StageCGateIntegrityError):
            raise
        raise StageCGateIntegrityError("frozen manifest metadata is invalid") from exc


def evaluate_stage_c_results(
    valid_results: tuple[HumanAiStageCReplayResult, ...],
    rejection_results: tuple[HumanAiStageCReplayResult, ...] = (),
) -> HumanAiStageCReport:
    results = (*valid_results, *rejection_results)
    if any(not isinstance(result, HumanAiStageCReplayResult) for result in results):
        raise TypeError("Stage C results must be HumanAiStageCReplayResult values")

    feature_versions = {result.feature_version for result in results}
    profile_versions = {result.threshold_profile_version for result in results}
    if len(feature_versions) > 1 or len(profile_versions) > 1:
        raise StageCGateIntegrityError("Stage C replay results mix frozen versions")
    if any(
        not isfinite(result.decision_latency_ms)
        or result.decision_latency_ms < 0
        or isinstance(result.runtime_ns, bool)
        or not isinstance(result.runtime_ns, int)
        or result.runtime_ns < 0
        for result in results
    ):
        raise StageCGateIntegrityError("Stage C replay timings must be finite and non-negative")

    valid = tuple(
        result
        for result in results
        if result.expected_outcome is StageCExpectedOutcome.ACCEPT
    )
    rejected = tuple(
        result
        for result in results
        if result.expected_outcome is StageCExpectedOutcome.REJECT
    )
    if len(valid) + len(rejected) != len(results):
        raise StageCGateIntegrityError("Stage C replay contains an unknown expected outcome")

    accepted = tuple(result for result in results if result.accepted)
    accepted_valid = tuple(result for result in valid if result.accepted)
    correct_accepts = sum(result.correct_accept for result in valid)
    false_accepts = sum(result.accepted and not result.correct_accept for result in results)
    correct_rejects = sum(not result.accepted for result in rejected)
    missed_valid = len(valid) - correct_accepts
    consistency_failures = sum(
        not result.recorded_observation_matches_replay for result in results
    )
    scenario_counts = tuple(
        (scenario.value, sum(result.scenario is scenario for result in rejected))
        for scenario in _REQUIRED_REJECTION_SCENARIOS
    )
    review_outcome_counts = tuple(
        (
            outcome.value,
            sum(result.review_outcome is outcome for result in results),
        )
        for outcome in (
            StageCReviewOutcome.CANDIDATE_CONFIRMED,
            StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
            StageCReviewOutcome.EXPECTED_REJECTION,
        )
    )
    metrics = HumanAiStageCMetrics(
        total_samples=len(results),
        valid_samples=len(valid),
        rejection_samples=len(rejected),
        distinct_valid_sessions=len({result.session_id for result in valid}),
        scenario_counts=scenario_counts,
        review_outcome_counts=review_outcome_counts,
        accepted_samples=len(accepted),
        correct_accepts=correct_accepts,
        false_accepts=false_accepts,
        correct_rejects=correct_rejects,
        missed_valid=missed_valid,
        recorded_consistency_failures=consistency_failures,
        coverage=_ratio(correct_accepts, len(valid)),
        accepted_precision=_ratio(correct_accepts, len(accepted)),
        final_position_accuracy=_ratio(correct_accepts, len(accepted_valid)),
        p95_decision_latency_ms=_p95(
            tuple(result.decision_latency_ms for result in results)
        ),
        p95_replay_runtime_ms=_p95(
            tuple(result.runtime_ns / 1_000_000 for result in results)
        ),
    )

    reasons: list[str] = []
    if metrics.valid_samples < _MIN_VALID_SAMPLES:
        reasons.append("minimum_valid_events")
    if metrics.distinct_valid_sessions < _MIN_DISTINCT_VALID_SESSIONS:
        reasons.append("minimum_distinct_valid_sessions")
    if metrics.rejection_samples < _MIN_REJECTION_SAMPLES:
        reasons.append("minimum_rejection_events")
    counts = dict(metrics.scenario_counts)
    for scenario in _REQUIRED_REJECTION_SCENARIOS:
        if counts[scenario.value] == 0:
            reasons.append(f"missing_rejection_scenario:{scenario.value}")
    if metrics.false_accepts != 0:
        reasons.append("zero_false_accepts")
    if metrics.coverage < _MIN_COVERAGE:
        reasons.append("minimum_valid_coverage")
    if metrics.p95_decision_latency_ms > _MAX_DECISION_P95_MS:
        reasons.append("maximum_decision_latency_p95")
    if metrics.recorded_consistency_failures != 0:
        reasons.append("recorded_replay_consistency")

    return HumanAiStageCReport(
        release_pass=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        feature_version=next(iter(feature_versions), ""),
        threshold_profile_version=next(iter(profile_versions), ""),
    )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _p95(values: tuple[float, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * 0.95) - 1)]


def _frozen_sample(value: object) -> FrozenStageCSampleV1:
    if not isinstance(value, dict):
        raise StageCGateIntegrityError("frozen sample entry must be an object")
    payload = cast(dict[str, object], value)
    if set(payload) != {"sample_id", "relative_path", "manifest_sha256"}:
        raise StageCGateIntegrityError("frozen sample fields are incomplete or unexpected")
    return FrozenStageCSampleV1(
        sample_id=str(payload["sample_id"]),
        relative_path=str(payload["relative_path"]),
        manifest_sha256=str(payload["manifest_sha256"]),
    )


def _threshold_profile(value: object) -> SequenceThresholdProfile:
    if not isinstance(value, dict):
        raise StageCGateIntegrityError("frozen threshold profile must be an object")
    payload = cast(dict[str, object], value)
    fields = {
        "min_local_difference",
        "max_unexpected_difference",
        "min_score",
        "min_margin",
        "max_template_distance",
        "min_template_margin",
        "min_template_confidence",
        "profile_version",
    }
    if set(payload) != fields:
        raise StageCGateIntegrityError("frozen threshold profile fields are incomplete")
    return SequenceThresholdProfile(
        min_local_difference=_number(payload["min_local_difference"]),
        max_unexpected_difference=_number(payload["max_unexpected_difference"]),
        min_score=_number(payload["min_score"]),
        min_margin=_number(payload["min_margin"]),
        max_template_distance=_number(payload["max_template_distance"]),
        min_template_margin=_number(payload["min_template_margin"]),
        min_template_confidence=_number(payload["min_template_confidence"]),
        profile_version=str(payload["profile_version"]),
    )


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageCGateIntegrityError("frozen threshold value must be numeric")
    number = float(value)
    if not isfinite(number):
        raise StageCGateIntegrityError("frozen threshold value must be finite")
    return number


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageCGateIntegrityError("frozen schema version must be an integer")
    return value


def _validate_relative_sample_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StageCGateIntegrityError("frozen sample path must be portable and relative")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise StageCGateIntegrityError("frozen sample path must be portable and relative")


def _validate_output_name(value: str) -> None:
    if not value or "\\" in value:
        raise StageCGateIntegrityError("frozen output must be one portable relative filename")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in ("", ".", ".."):
        raise StageCGateIntegrityError("frozen output must be one portable relative filename")


def _resolve_sample_dir(root: Path, relative_path: str) -> Path:
    _validate_relative_sample_path(relative_path)
    parts = PurePosixPath(relative_path).parts
    candidate = (root / Path(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StageCGateIntegrityError("frozen sample path escapes sample root") from exc
    if not candidate.is_dir():
        raise StageCGateIntegrityError("frozen sample directory does not exist")
    return candidate


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise StageCGateIntegrityError("frozen timestamp must be UTC and end in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise StageCGateIntegrityError("frozen timestamp is invalid") from exc
    return parsed.astimezone(UTC)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
