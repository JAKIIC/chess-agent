from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops
from xiangqi_agent.diagnostics.stage_c_quarantine import (
    LoadedQuarantinedStageCEvent,
    QuarantineEventLoader,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewStore,
    StageCReviewV1,
    load_stage_c_review,
)
from xiangqi_agent.diagnostics.stage_c_reviewed_samples import (
    ReviewedStageCSampleLoader,
    ReviewedStageCSampleV2,
    reviewed_manifest_bytes,
)
from xiangqi_agent.diagnostics.stage_c_samples import (
    StageCExpectedOutcome,
    StageCObservedStatus,
    StageCScenario,
)
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.sync.two_ply_profile import (
    TWO_PLY_FEATURE_VERSION,
    TWO_PLY_INSTANCE_TRANSFER_MAX_SHIFT,
    TWO_PLY_MINIMUM_SEMANTIC_CONFIDENCE,
)
from xiangqi_agent.vision.endpoint_features import InstanceTransferExtractor

_MINIMUM_OCCUPANCY_CONFIDENCE = 0.65
_MACHINE_TERMINAL_REASONS = frozenset(
    {
        "frame_size_changed",
        "capture_context_invalid",
        "target_window_closed",
    }
)


class PromotionStatus(StrEnum):
    PROMOTABLE = "promotable"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    status: PromotionStatus
    reason_codes: tuple[str, ...]
    projected_final_position_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PromotionStatus):
            raise TypeError("status must be a PromotionStatus")
        if not isinstance(self.reason_codes, tuple) or any(
            not isinstance(reason, str) or not reason for reason in self.reason_codes
        ):
            raise TypeError("reason_codes must contain non-empty strings")
        if len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason_codes must not contain duplicates")
        if self.status is PromotionStatus.PROMOTABLE and self.reason_codes:
            raise ValueError("promotable decision cannot contain failure reasons")


class StageCPromotionBlockedError(RuntimeError):
    def __init__(self, decision: PromotionDecision) -> None:
        self.decision = decision
        reasons = ",".join(decision.reason_codes) or decision.status.value
        super().__init__(f"Stage C promotion blocked: {reasons}")


@dataclass(frozen=True, slots=True)
class _VerifiedInputs:
    event: LoadedQuarantinedStageCEvent
    review: StageCReviewV1
    review_bytes: bytes
    projected: BoardState


@dataclass(frozen=True, slots=True)
class _Inspection:
    decision: PromotionDecision
    inputs: _VerifiedInputs | None = None


class StageCPromotionVerifier:
    verifier_version = "stage-c-promotion-v2"
    occupancy_verifier_version = "occupancy-evidence-v1"

    def verify(self, event_dir: Path, review_path: Path) -> PromotionDecision:
        return self._inspect(event_dir, review_path).decision

    def _inspect(self, event_dir: Path, review_path: Path) -> _Inspection:
        if not isinstance(event_dir, Path) or not isinstance(review_path, Path):
            return _rejected("invalid_path_type")
        try:
            event = QuarantineEventLoader().load(event_dir)
        except (OSError, RuntimeError, TypeError, ValueError):
            return _rejected("event_integrity")
        if (
            event_dir.name != event.metadata.event_id
            or event_dir.parent.name != event.metadata.session_id
        ):
            return _rejected("identifier_mismatch")

        try:
            review_bytes = review_path.read_bytes()
            review = load_stage_c_review(review_path)
            if review_path.read_bytes() != review_bytes:
                return _rejected("review_integrity")
        except (OSError, RuntimeError, TypeError, ValueError):
            return _rejected("review_integrity")
        if (
            review.event_id != event.metadata.event_id
            or review.session_id != event.metadata.session_id
        ):
            return _rejected("identifier_mismatch")
        if sha256(event.manifest_bytes).hexdigest() != review.event_manifest_sha256:
            return _rejected("review_event_hash_mismatch")
        if _parse_utc(review.created_at_utc) < _parse_utc(event.metadata.created_at_utc):
            return _rejected("review_predates_event")

        try:
            review_root = review_path.parents[2]
            active = StageCReviewStore(
                review_root,
                enabled=True,
            ).active_review(review.session_id, review.event_id)
        except (IndexError, OSError, RuntimeError, TypeError, ValueError):
            return _rejected("review_integrity")
        if active is None or active.review_id != review.review_id:
            return _rejected("inactive_review")

        try:
            board = parse_fen(event.metadata.confirmed_fen)
        except ValueError:
            return _rejected("event_integrity")
        if (
            board.position_id != event.metadata.confirmed_position_id
            or board.side_to_move != event.metadata.side_to_move
        ):
            return _rejected("event_integrity")

        candidate_finals: list[BoardState] = []
        for candidate in event.metadata.candidates:
            projected_candidate = _project(board, candidate.moves_uci)
            if (
                projected_candidate is None
                or projected_candidate.position_id != candidate.final_position_id
                or _changed_points(board, projected_candidate) != candidate.changed_points
            ):
                return _rejected("candidate_integrity")
            candidate_finals.append(projected_candidate)
        if event.metadata.observed_status is StageCObservedStatus.ACCEPTED:
            observed = _project(board, event.metadata.observed_moves_uci)
            if (
                observed is None
                or observed.position_id != event.metadata.observed_final_position_id
            ):
                return _rejected("observation_integrity")

        baseline = event.metadata.before_occupancy
        if any(confidence < _MINIMUM_OCCUPANCY_CONFIDENCE for confidence in baseline.confidences):
            return _needs_review("baseline_confidence")
        expected_baseline = tuple(piece != "." for piece in board.pieces)
        if baseline.occupied != expected_baseline:
            return _rejected("baseline_occupancy_mismatch")

        projected = _project(board, review.moves_uci)
        if projected is None:
            return _rejected("illegal_move_chain")
        if review.label_kind is StageCLabelKind.DISCARD:
            return _rejected("discarded")
        if review.label_kind is StageCLabelKind.VALID_TWO_PLY:
            if (
                len(review.moves_uci) != 2
                or review.expected_final_position_id != projected.position_id
            ):
                return _rejected("final_position_mismatch")
            evidence = _verify_projected_evidence(
                event,
                board,
                projected,
                review.moves_uci,
            )
            if evidence is not None:
                return evidence
        else:
            scenario_result = _verify_rejection_scenario(
                event,
                review,
                board,
                tuple(candidate_finals),
            )
            if scenario_result is not None:
                return scenario_result
            evidence = _verify_projected_evidence(
                event,
                board,
                projected,
                review.moves_uci,
            )
            if evidence is not None:
                return evidence

        decision = PromotionDecision(
            PromotionStatus.PROMOTABLE,
            (),
            projected.position_id,
        )
        return _Inspection(
            decision,
            _VerifiedInputs(event, review, review_bytes, projected),
        )


class StageCPromotionService:
    def __init__(
        self,
        verifier: StageCPromotionVerifier | None = None,
        *,
        now_utc: Callable[[], datetime] | None = None,
        before_copy: Callable[[], None] | None = None,
    ) -> None:
        self._verifier = verifier or StageCPromotionVerifier()
        if not isinstance(self._verifier, StageCPromotionVerifier):
            raise TypeError("verifier must be a StageCPromotionVerifier")
        self._now_utc = now_utc or (lambda: datetime.now(UTC))
        self._before_copy = before_copy

    def promote(
        self,
        event_dir: Path,
        review_path: Path,
        reviewed_root: Path,
    ) -> Path:
        if not all(isinstance(path, Path) for path in (event_dir, review_path, reviewed_root)):
            raise TypeError("promotion paths must be Paths")
        quarantine_root = event_dir.parent.parent
        try:
            review_root = review_path.parents[2]
        except IndexError as exc:
            raise ValueError("review path does not have the required layout") from exc
        _validate_non_overlapping_roots(
            quarantine_root,
            review_root,
            reviewed_root,
        )
        if (
            quarantine_root.name != "stage-c-quarantine"
            or review_root.name != "stage-c-reviews"
            or reviewed_root.name != "stage-c-reviewed"
        ):
            raise ValueError("promotion requires the three fixed Stage C roots")
        if reviewed_root.is_symlink() or (reviewed_root.exists() and not reviewed_root.is_dir()):
            raise ValueError("reviewed root must be a real directory, not a symlink")
        _validate_source_paths(event_dir, review_path, quarantine_root, review_root)

        inspection = self._verifier._inspect(event_dir, review_path)
        if inspection.decision.status is not PromotionStatus.PROMOTABLE:
            raise StageCPromotionBlockedError(inspection.decision)
        inputs = inspection.inputs
        if inputs is None:
            raise RuntimeError("promotable inspection did not preserve verified inputs")
        target = reviewed_root / inputs.event.metadata.session_id / inputs.event.metadata.event_id
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"reviewed Stage C sample already exists: {inputs.event.metadata.event_id}"
            )

        if self._before_copy is not None:
            self._before_copy()
        source_snapshot = _read_source_snapshot(event_dir, review_path, inputs)
        now = self._now_utc()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("promotion clock must return a timezone-aware datetime")
        sample = _reviewed_sample(inputs, promoted_at=now)
        crop_hashes = {
            filename: sha256(contents).hexdigest()
            for filename, contents in source_snapshot.crops.items()
        }
        manifest_bytes = reviewed_manifest_bytes(sample, crop_hashes)

        temporary_root = reviewed_root / f".promotion-{uuid4().hex}"
        temporary_sample = temporary_root / sample.session_id / sample.sample_id
        temporary_sample.mkdir(parents=True)
        try:
            (temporary_sample / "source-event-manifest.json").write_bytes(
                source_snapshot.event_manifest
            )
            (temporary_sample / "review-manifest.json").write_bytes(source_snapshot.review_manifest)
            for filename, contents in source_snapshot.crops.items():
                (temporary_sample / filename).write_bytes(contents)
            (temporary_sample / "manifest.json").write_bytes(manifest_bytes)
            loaded = ReviewedStageCSampleLoader().load(temporary_sample)
            if loaded.metadata != sample:
                raise RuntimeError("temporary reviewed sample changed during reload")
            _assert_sources_unchanged(event_dir, review_path, source_snapshot)

            target.parent.mkdir(parents=True, exist_ok=True)
            _validate_real_directory(reviewed_root, "reviewed root")
            _validate_real_directory(target.parent, "reviewed session")
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"reviewed Stage C sample already exists: {sample.sample_id}")
            temporary_sample.rename(target)
        except BaseException:
            shutil.rmtree(temporary_root, ignore_errors=True)
            if target.parent.exists() and not any(target.parent.iterdir()):
                target.parent.rmdir()
            if reviewed_root.exists() and not any(reviewed_root.iterdir()):
                reviewed_root.rmdir()
            raise
        shutil.rmtree(temporary_root, ignore_errors=True)
        return target


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    event_manifest: bytes
    review_manifest: bytes
    crops: dict[str, bytes]


def _read_source_snapshot(
    event_dir: Path,
    review_path: Path,
    inputs: _VerifiedInputs,
) -> _SourceSnapshot:
    event_manifest = (event_dir / "manifest.json").read_bytes()
    review_manifest = review_path.read_bytes()
    if event_manifest != inputs.event.manifest_bytes or review_manifest != inputs.review_bytes:
        raise StageCPromotionBlockedError(
            PromotionDecision(PromotionStatus.REJECTED, ("source_mutated",), None)
        )
    try:
        payload = json.loads(event_manifest.decode("utf-8"))
        hashes = cast(dict[str, str], cast(dict[str, Any], payload)["crop_hashes"])
    except (KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageCPromotionBlockedError(
            PromotionDecision(PromotionStatus.REJECTED, ("source_mutated",), None)
        ) from exc
    crop_names = tuple(
        filename
        for point in inputs.event.metadata.changed_points
        for filename in (
            f"point-{point:02d}-before.png",
            f"point-{point:02d}-after.png",
        )
    )
    crops: dict[str, bytes] = {}
    for filename in crop_names:
        contents = (event_dir / filename).read_bytes()
        if sha256(contents).hexdigest() != hashes.get(filename):
            raise StageCPromotionBlockedError(
                PromotionDecision(
                    PromotionStatus.REJECTED,
                    ("source_mutated",),
                    None,
                )
            )
        crops[filename] = contents
    return _SourceSnapshot(event_manifest, review_manifest, crops)


def _assert_sources_unchanged(
    event_dir: Path,
    review_path: Path,
    snapshot: _SourceSnapshot,
) -> None:
    if (
        (event_dir / "manifest.json").read_bytes() != snapshot.event_manifest
        or review_path.read_bytes() != snapshot.review_manifest
        or any(
            (event_dir / filename).read_bytes() != contents
            for filename, contents in snapshot.crops.items()
        )
    ):
        raise StageCPromotionBlockedError(
            PromotionDecision(PromotionStatus.REJECTED, ("source_mutated",), None)
        )


def _reviewed_sample(
    inputs: _VerifiedInputs,
    *,
    promoted_at: datetime,
) -> ReviewedStageCSampleV2:
    event = inputs.event.metadata
    review = inputs.review
    accepted = review.label_kind is StageCLabelKind.VALID_TWO_PLY
    scenario = StageCScenario.VALID_TWO_PLY if accepted else review.scenario
    if scenario is None:
        raise RuntimeError("verified rejection review lost its scenario")
    return ReviewedStageCSampleV2(
        sample_id=event.event_id,
        session_id=event.session_id,
        created_at_utc=event.created_at_utc,
        confirmed_fen=event.confirmed_fen,
        confirmed_position_id=event.confirmed_position_id,
        expected_outcome=(
            StageCExpectedOutcome.ACCEPT if accepted else StageCExpectedOutcome.REJECT
        ),
        scenario=scenario,
        ground_truth_moves_uci=review.moves_uci,
        expected_final_position_id=(inputs.projected.position_id if accepted else None),
        observed_status=event.observed_status,
        observed_moves_uci=event.observed_moves_uci,
        observed_final_position_id=event.observed_final_position_id,
        side_to_move=event.side_to_move,
        orientation=event.orientation,
        changed_points=event.changed_points,
        local_differences=event.local_differences,
        candidates=event.candidates,
        rejection_reasons=event.rejection_reasons,
        capture_context=event.capture_context,
        feature_version=event.feature_version,
        threshold_profile_version=event.threshold_profile_version,
        decision_latency_ms=event.decision_latency_ms,
        source_event_manifest_sha256=sha256(inputs.event.manifest_bytes).hexdigest(),
        review_manifest_sha256=sha256(inputs.review_bytes).hexdigest(),
        review_outcome=review.review_outcome,
        occupancy_verifier_version=event.before_occupancy.algorithm_version,
        promotion_verifier_version=StageCPromotionVerifier.verifier_version,
        promoted_at_utc=promoted_at.astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
    )


def _verify_projected_evidence(
    event: LoadedQuarantinedStageCEvent,
    before: BoardState,
    after: BoardState,
    moves_uci: tuple[str, ...],
) -> _Inspection | None:
    metadata = event.metadata
    expected_after = tuple(piece != "." for piece in after.pieces)
    high_confidence_mismatch = any(
        confidence >= _MINIMUM_OCCUPANCY_CONFIDENCE and observed != expected
        for observed, confidence, expected in zip(
            metadata.after_occupancy.occupied,
            metadata.after_occupancy.confidences,
            expected_after,
            strict=True,
        )
    )
    if high_confidence_mismatch:
        return _rejected("final_occupancy_mismatch")
    changed = _changed_points(before, after)
    if not set(changed) <= set(metadata.changed_points):
        return _rejected("missing_changed_point_crop")
    if any(metadata.local_differences[index] <= 0 for index in changed):
        return _rejected("missing_local_change")
    if any(
        metadata.after_occupancy.confidences[index] < _MINIMUM_OCCUPANCY_CONFIDENCE
        for index in changed
    ) and not _piece_transfer_verifies_changed_points(
        event,
        before,
        after,
        moves_uci,
    ):
        return _needs_review("changed_point_confidence")
    return None


def _piece_transfer_verifies_changed_points(
    event: LoadedQuarantinedStageCEvent,
    before: BoardState,
    after: BoardState,
    moves_uci: tuple[str, ...],
) -> bool:
    metadata = event.metadata
    if (
        metadata.feature_version != TWO_PLY_FEATURE_VERSION
        or metadata.observed_status is not StageCObservedStatus.ACCEPTED
        or metadata.observed_moves_uci != moves_uci
        or len(moves_uci) != 2
        or metadata.after_occupancy.occupied
        != tuple(piece != "." for piece in after.pieces)
    ):
        return False
    candidate = next(
        (
            value
            for value in metadata.candidates
            if value.moves_uci == moves_uci
            and value.final_position_id == after.position_id
        ),
        None,
    )
    if (
        candidate is None
        or candidate.minimum_template_confidence
        < TWO_PLY_MINIMUM_SEMANTIC_CONFIDENCE
    ):
        return False
    moves = _projected_moves(before, moves_uci)
    if moves is None or len(moves) != 2:
        return False
    first, second = moves
    surviving_moves = (first, second) if second.to_index != first.to_index else (second,)
    crops = {crop.point_index: crop for crop in event.crops}
    extractor = InstanceTransferExtractor(
        max_shift=TWO_PLY_INSTANCE_TRANSFER_MAX_SHIFT
    )
    for move in surviving_moves:
        source = crops.get(move.from_index)
        target = crops.get(move.to_index)
        if source is None or target is None:
            return False
        score = extractor.extract(
            EndpointCrops(
                source_before=source.before,
                source_after=source.after,
                target_before=target.before,
                target_after=target.after,
            )
        ).instance_evidence_score
        if score < TWO_PLY_MINIMUM_SEMANTIC_CONFIDENCE:
            return False
    return True


def _projected_moves(
    board: BoardState,
    moves_uci: tuple[str, ...],
) -> tuple[Move, ...] | None:
    projected = board
    moves: list[Move] = []
    for uci in moves_uci:
        move = next(
            (candidate for candidate in legal_moves(projected) if candidate.uci == uci),
            None,
        )
        if move is None:
            return None
        moves.append(move)
        projected = apply_move(projected, move)
    return tuple(moves)


def _verify_rejection_scenario(
    event: LoadedQuarantinedStageCEvent,
    review: StageCReviewV1,
    board: BoardState,
    candidate_finals: tuple[BoardState, ...],
) -> _Inspection | None:
    scenario = review.scenario
    if scenario is StageCScenario.THREE_PLY:
        if len(review.moves_uci) != 3:
            return _rejected("three_ply_count")
        return None
    if scenario is StageCScenario.MULTIPLE_CANDIDATES:
        distinct_candidates = {candidate.moves_uci for candidate in event.metadata.candidates}
        if len(candidate_finals) >= 2 and len(distinct_candidates) >= 2:
            return None
        if _compatible_two_ply_count(event, board) < 2:
            return _rejected("multiple_candidate_evidence")
        return None
    if scenario in (
        StageCScenario.SELECTION_HIGHLIGHT,
        StageCScenario.CONTINUOUS_ANIMATION,
        StageCScenario.OCCLUSION,
    ):
        if event.metadata.observed_status is not StageCObservedStatus.REJECTED:
            return _rejected("scenario_evidence_mismatch")
        return None
    if scenario is StageCScenario.RESIZE:
        if (
            event.metadata.observed_status is not StageCObservedStatus.REJECTED
            or not set(event.metadata.rejection_reasons) & _MACHINE_TERMINAL_REASONS
        ):
            return _rejected("scenario_evidence_mismatch")
        return None
    return _rejected("scenario_evidence_mismatch")


def _compatible_two_ply_count(
    event: LoadedQuarantinedStageCEvent,
    board: BoardState,
) -> int:
    count = 0
    occupancy = event.metadata.after_occupancy
    for first in legal_moves(board):
        middle = apply_move(board, first)
        for second in legal_moves(middle):
            final = apply_move(middle, second)
            expected = tuple(piece != "." for piece in final.pieces)
            if all(
                confidence < _MINIMUM_OCCUPANCY_CONFIDENCE or observed == target
                for observed, confidence, target in zip(
                    occupancy.occupied,
                    occupancy.confidences,
                    expected,
                    strict=True,
                )
            ):
                changed = _changed_points(board, final)
                if all(event.metadata.local_differences[index] > 0 for index in changed):
                    count += 1
                    if count >= 2:
                        return count
    return count


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


def _changed_points(before: BoardState, after: BoardState) -> tuple[int, ...]:
    return tuple(
        index
        for index, (left, right) in enumerate(zip(before.pieces, after.pieces, strict=True))
        if left != right
    )


def _rejected(reason: str) -> _Inspection:
    return _Inspection(PromotionDecision(PromotionStatus.REJECTED, (reason,), None))


def _needs_review(reason: str) -> _Inspection:
    return _Inspection(PromotionDecision(PromotionStatus.NEEDS_REVIEW, (reason,), None))


def _validate_non_overlapping_roots(*roots: Path) -> None:
    resolved = tuple(root.resolve() for root in roots)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if _contains(left, right) or _contains(right, left):
                raise ValueError("Stage C storage roots must not overlap")


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_source_paths(
    event_dir: Path,
    review_path: Path,
    quarantine_root: Path,
    review_root: Path,
) -> None:
    paths = (
        quarantine_root,
        event_dir.parent,
        event_dir,
        review_root,
        review_path.parent.parent,
        review_path.parent,
        review_path,
    )
    if any(path.is_symlink() for path in paths):
        raise ValueError("promotion source paths must not contain symlinks")
    try:
        event_relative = event_dir.resolve().relative_to(quarantine_root.resolve())
        review_relative = review_path.resolve().relative_to(review_root.resolve())
    except ValueError as exc:
        raise ValueError("promotion source path escapes its fixed root") from exc
    if len(event_relative.parts) != 2 or len(review_relative.parts) != 3:
        raise ValueError("promotion source paths have an invalid layout")


def _validate_real_directory(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{name} must be a real directory")


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(f"{value[:-1]}+00:00").astimezone(UTC)
