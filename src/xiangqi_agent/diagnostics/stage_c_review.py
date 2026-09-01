from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from xiangqi_agent.diagnostics.endpoint_samples import (
    DiagnosticsDisabledError,
    SampleQuotaExceededError,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import QuarantineEventLoader
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import apply_move, legal_moves

_DEFAULT_MAX_BYTES = 8 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_POSITION_ID = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UCI = re.compile(r"^[a-i][0-9][a-i][0-9]$")
_REVIEW_FIELDS = {
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
_REJECTION_SCENARIOS = frozenset(StageCScenario) - {
    StageCScenario.VALID_TWO_PLY
}


class StageCReviewIntegrityError(ValueError):
    """An immutable review tree or review file is inconsistent or unsafe."""


class StageCLabelKind(StrEnum):
    VALID_TWO_PLY = "valid_two_ply"
    EXPECTED_REJECTION = "expected_rejection"
    DISCARD = "discard"


class StageCReviewOutcome(StrEnum):
    CANDIDATE_CONFIRMED = "candidate_confirmed"
    LEGAL_MOVE_CORRECTION = "legal_move_correction"
    EXPECTED_REJECTION = "expected_rejection"
    DISCARDED = "discarded"


@dataclass(frozen=True, slots=True, kw_only=True)
class StageCReviewV1:
    review_id: str
    event_id: str
    session_id: str
    created_at_utc: str
    event_manifest_sha256: str
    label_kind: StageCLabelKind
    moves_uci: tuple[str, ...]
    expected_final_position_id: str | None
    scenario: StageCScenario | None
    review_outcome: StageCReviewOutcome
    supersedes_review_id: str | None
    reviewer_kind: str = "local_user"
    ui_version: str = "stage-c-review-v1"
    rules_version: str = "xiangqi-rules-v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.review_id)
        _validate_identifier(self.event_id)
        _validate_identifier(self.session_id)
        _parse_utc(self.created_at_utc)
        if (
            not isinstance(self.event_manifest_sha256, str)
            or _SHA256.fullmatch(self.event_manifest_sha256) is None
        ):
            raise ValueError(
                "event manifest hash must contain 64 lowercase hexadecimal characters"
            )
        _validate_label_values(
            label_kind=self.label_kind,
            moves_uci=self.moves_uci,
            expected_final_position_id=self.expected_final_position_id,
            scenario=self.scenario,
            review_outcome=self.review_outcome,
        )
        if self.supersedes_review_id is not None:
            _validate_identifier(self.supersedes_review_id)
            if self.supersedes_review_id == self.review_id:
                raise ValueError("a review cannot supersede itself")
        if self.reviewer_kind != "local_user":
            raise ValueError("reviewer_kind must remain anonymous local_user")
        if self.ui_version != "stage-c-review-v1":
            raise ValueError("ui_version must be stage-c-review-v1")
        if self.rules_version != "xiangqi-rules-v1":
            raise ValueError("rules_version must be xiangqi-rules-v1")
        if self.schema_version != 1:
            raise ValueError("StageCReviewV1 schema_version must be 1")


@dataclass(frozen=True, slots=True)
class ReviewMoveChoice:
    uci: str
    chinese: str
    resulting_position_id: str

    def __post_init__(self) -> None:
        _validate_move(self.uci)
        if not isinstance(self.chinese, str) or not self.chinese.strip():
            raise ValueError("Chinese notation must be non-empty")
        _validate_position_id(self.resulting_position_id)


def legal_review_choices(board: BoardState) -> tuple[ReviewMoveChoice, ...]:
    if not isinstance(board, BoardState):
        raise TypeError("board must be a BoardState")
    choices = tuple(
        ReviewMoveChoice(
            move.uci,
            to_chinese(board, move),
            apply_move(board, move).position_id,
        )
        for move in legal_moves(board)
    )
    return tuple(sorted(choices, key=lambda choice: (choice.chinese, choice.uci)))


@dataclass(frozen=True, slots=True, kw_only=True)
class StageCReviewDraft:
    label_kind: StageCLabelKind
    moves_uci: tuple[str, ...]
    scenario: StageCScenario | None
    review_outcome: StageCReviewOutcome
    supersedes_review_id: str | None = None

    def __post_init__(self) -> None:
        _validate_label_values(
            label_kind=self.label_kind,
            moves_uci=self.moves_uci,
            expected_final_position_id=(
                "0" * 32
                if self.label_kind is StageCLabelKind.VALID_TWO_PLY
                else None
            ),
            scenario=self.scenario,
            review_outcome=self.review_outcome,
        )
        if self.supersedes_review_id is not None:
            _validate_identifier(self.supersedes_review_id)


class StageCReviewStore:
    def __init__(
        self,
        root: Path,
        *,
        enabled: bool = False,
        max_bytes: int = _DEFAULT_MAX_BYTES,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("review root must be a Path")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        self._root = root
        self._enabled = enabled
        self._max_bytes = max_bytes

    @property
    def root(self) -> Path:
        return self._root

    def submit(self, review: StageCReviewV1) -> Path:
        if not self._enabled:
            raise DiagnosticsDisabledError(
                "Stage C review storage must be explicitly enabled"
            )
        if not isinstance(review, StageCReviewV1):
            raise TypeError("review must be a StageCReviewV1")
        _validate_directory(self._root, "review root", allow_missing=True)
        session_dir = self._root / review.session_id
        event_dir = session_dir / review.event_id
        _validate_directory(session_dir, "review session", allow_missing=True)
        _validate_directory(event_dir, "review event", allow_missing=True)
        final_path = event_dir / f"{review.review_id}.json"
        if final_path.exists() or final_path.is_symlink():
            raise FileExistsError(f"review already exists: {review.review_id}")

        existing = _load_event_reviews(event_dir, review.session_id, review.event_id)
        active = _active_review(existing)
        if active is None:
            if existing:
                raise StageCReviewIntegrityError("review history has no active review")
            if review.supersedes_review_id is not None:
                raise StageCReviewIntegrityError(
                    "first review cannot name an active review to supersede"
                )
        else:
            if review.supersedes_review_id != active.review_id:
                raise StageCReviewIntegrityError(
                    "new review must supersede the current active review"
                )
            if review.event_manifest_sha256 != active.event_manifest_sha256:
                raise StageCReviewIntegrityError(
                    "review chain must retain the same event manifest hash"
                )
            if _parse_utc(review.created_at_utc) < _parse_utc(active.created_at_utc):
                raise StageCReviewIntegrityError(
                    "review correction cannot predate the active review"
                )

        contents = _review_bytes(review)
        if _tree_size(self._root) + len(contents) > self._max_bytes:
            raise SampleQuotaExceededError("Stage C review capacity would be exceeded")

        event_dir.mkdir(parents=True, exist_ok=True)
        _validate_directory(self._root, "review root", allow_missing=False)
        _validate_directory(session_dir, "review session", allow_missing=False)
        _validate_directory(event_dir, "review event", allow_missing=False)
        _assert_within_root(final_path, self._root)
        created = False
        try:
            with final_path.open("xb") as stream:
                created = True
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            if load_stage_c_review(final_path) != review:
                raise StageCReviewIntegrityError(
                    "stored review changed during verification"
                )
            if self.active_review(review.session_id, review.event_id) != review:
                raise StageCReviewIntegrityError(
                    "stored review did not become the unique active review"
                )
        except BaseException:
            if created:
                final_path.unlink(missing_ok=True)
                _remove_empty_parents(event_dir, session_dir, self._root)
            raise
        return final_path

    def active_review(
        self,
        session_id: str,
        event_id: str,
    ) -> StageCReviewV1 | None:
        _validate_identifier(session_id)
        _validate_identifier(event_id)
        _validate_directory(self._root, "review root", allow_missing=True)
        if not self._root.exists():
            return None
        session_dir = self._root / session_id
        _validate_directory(session_dir, "review session", allow_missing=True)
        if not session_dir.exists():
            return None
        event_dir = session_dir / event_id
        _validate_directory(event_dir, "review event", allow_missing=True)
        if not event_dir.exists():
            return None
        return _active_review(_load_event_reviews(event_dir, session_id, event_id))


class StageCReviewService:
    def __init__(
        self,
        store: StageCReviewStore,
        *,
        now_utc: Callable[[], datetime] | None = None,
        review_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(store, StageCReviewStore):
            raise TypeError("store must be a StageCReviewStore")
        self._store = store
        self._now_utc = now_utc or (lambda: datetime.now(UTC))
        self._review_id_factory = review_id_factory or (lambda: uuid4().hex)

    def submit(self, event_dir: Path, draft: StageCReviewDraft) -> Path:
        if not isinstance(event_dir, Path):
            raise TypeError("event_dir must be a Path")
        if not isinstance(draft, StageCReviewDraft):
            raise TypeError("draft must be a StageCReviewDraft")
        loaded = QuarantineEventLoader().load(event_dir)
        if (
            event_dir.name != loaded.metadata.event_id
            or event_dir.parent.name != loaded.metadata.session_id
        ):
            raise StageCReviewIntegrityError(
                "quarantine path does not match its session and event ids"
            )
        board = parse_fen(loaded.metadata.confirmed_fen)
        projected = _project_legal_sequence(board, draft.moves_uci)

        prefilled = (
            loaded.metadata.candidates[0].moves_uci
            if loaded.metadata.candidates
            else None
        )
        if draft.review_outcome is StageCReviewOutcome.CANDIDATE_CONFIRMED:
            if prefilled is None or draft.moves_uci != prefilled:
                raise ValueError(
                    "candidate confirmation must match the prefilled candidate"
                )
        elif (
            draft.review_outcome is StageCReviewOutcome.LEGAL_MOVE_CORRECTION
            and prefilled is not None
            and draft.moves_uci == prefilled
        ):
            raise ValueError("legal correction must differ from the prefilled candidate")

        expected_final = (
            projected.position_id
            if draft.label_kind is StageCLabelKind.VALID_TWO_PLY
            else None
        )
        review_id = self._review_id_factory()
        _validate_identifier(review_id)
        created_at = self._now_utc()
        if (
            not isinstance(created_at, datetime)
            or created_at.tzinfo is None
            or created_at.utcoffset() is None
        ):
            raise ValueError("review clock must return a timezone-aware datetime")
        review = StageCReviewV1(
            review_id=review_id,
            event_id=loaded.metadata.event_id,
            session_id=loaded.metadata.session_id,
            created_at_utc=created_at.astimezone(UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            event_manifest_sha256=sha256(loaded.manifest_bytes).hexdigest(),
            label_kind=draft.label_kind,
            moves_uci=draft.moves_uci,
            expected_final_position_id=expected_final,
            scenario=draft.scenario,
            review_outcome=draft.review_outcome,
            supersedes_review_id=draft.supersedes_review_id,
        )
        return self._store.submit(review)


def load_stage_c_review(path: Path) -> StageCReviewV1:
    if not isinstance(path, Path):
        raise TypeError("review path must be a Path")
    if path.is_symlink() or not path.is_file():
        raise StageCReviewIntegrityError("review file does not exist or is a symlink")
    _validate_directory(path.parent, "review event", allow_missing=False)
    _validate_directory(path.parent.parent, "review session", allow_missing=False)
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageCReviewIntegrityError("review file is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise StageCReviewIntegrityError("review root must be an object")
    payload = cast(dict[str, Any], value)
    if set(payload) != _REVIEW_FIELDS:
        raise StageCReviewIntegrityError(
            "review fields are incomplete or unexpected"
        )
    try:
        scenario_value = payload["scenario"]
        final_value = payload["expected_final_position_id"]
        supersedes_value = payload["supersedes_review_id"]
        review = StageCReviewV1(
            review_id=_string(payload["review_id"]),
            event_id=_string(payload["event_id"]),
            session_id=_string(payload["session_id"]),
            created_at_utc=_string(payload["created_at_utc"]),
            event_manifest_sha256=_string(payload["event_manifest_sha256"]),
            label_kind=StageCLabelKind(_string(payload["label_kind"])),
            moves_uci=_moves(payload["moves_uci"]),
            expected_final_position_id=(
                None if final_value is None else _string(final_value)
            ),
            scenario=(
                None if scenario_value is None else StageCScenario(_string(scenario_value))
            ),
            review_outcome=StageCReviewOutcome(
                _string(payload["review_outcome"])
            ),
            supersedes_review_id=(
                None if supersedes_value is None else _string(supersedes_value)
            ),
            reviewer_kind=_string(payload["reviewer_kind"]),
            ui_version=_string(payload["ui_version"]),
            rules_version=_string(payload["rules_version"]),
            schema_version=_integer(payload["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StageCReviewIntegrityError("review metadata is invalid") from exc
    if path.stem != review.review_id:
        raise StageCReviewIntegrityError("review filename does not match review_id")
    if path.parent.name != review.event_id or path.parent.parent.name != review.session_id:
        raise StageCReviewIntegrityError("review path does not match session and event ids")
    return review


def project_review_prefix(board: BoardState, moves_uci: tuple[str, ...]) -> BoardState:
    """Apply a user-selected legal prefix for the CLI and Qt legal selectors."""
    if not isinstance(board, BoardState):
        raise TypeError("board must be a BoardState")
    _validate_moves(moves_uci, maximum=3)
    return _project_legal_sequence(board, moves_uci)


def _project_legal_sequence(
    board: BoardState,
    moves_uci: tuple[str, ...],
) -> BoardState:
    projected = board
    for uci in moves_uci:
        move = _legal_move_by_uci(projected, uci)
        if move is None:
            raise ValueError("review moves must form a sequentially legal chain")
        projected = apply_move(projected, move)
    return projected


def _legal_move_by_uci(board: BoardState, uci: str) -> Move | None:
    return next((move for move in legal_moves(board) if move.uci == uci), None)


def _validate_label_values(
    *,
    label_kind: StageCLabelKind,
    moves_uci: tuple[str, ...],
    expected_final_position_id: str | None,
    scenario: StageCScenario | None,
    review_outcome: StageCReviewOutcome,
) -> None:
    if not isinstance(label_kind, StageCLabelKind):
        raise TypeError("label_kind must be a StageCLabelKind")
    if not isinstance(review_outcome, StageCReviewOutcome):
        raise TypeError("review_outcome must be a StageCReviewOutcome")
    if scenario is not None and not isinstance(scenario, StageCScenario):
        raise TypeError("scenario must be a StageCScenario or None")
    _validate_moves(moves_uci, maximum=3)
    if label_kind is StageCLabelKind.VALID_TWO_PLY:
        if len(moves_uci) != 2:
            raise ValueError("valid_two_ply review must contain exactly two moves")
        if expected_final_position_id is None:
            raise ValueError("valid_two_ply review must contain a final position")
        _validate_position_id(expected_final_position_id)
        if scenario is not None:
            raise ValueError("valid_two_ply review has no rejection scenario")
        if review_outcome not in (
            StageCReviewOutcome.CANDIDATE_CONFIRMED,
            StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
        ):
            raise ValueError(
                "valid_two_ply requires candidate confirmation or legal correction"
            )
    elif label_kind is StageCLabelKind.EXPECTED_REJECTION:
        if scenario not in _REJECTION_SCENARIOS:
            raise ValueError("expected_rejection requires a frozen rejection scenario")
        if expected_final_position_id is not None:
            raise ValueError("expected_rejection must not contain a final position")
        if review_outcome is not StageCReviewOutcome.EXPECTED_REJECTION:
            raise ValueError("expected_rejection must use its matching review outcome")
    else:
        if moves_uci or expected_final_position_id is not None or scenario is not None:
            raise ValueError("discard review cannot contain moves, final position, or scenario")
        if review_outcome is not StageCReviewOutcome.DISCARDED:
            raise ValueError("discard review must use the discarded outcome")


def _review_bytes(review: StageCReviewV1) -> bytes:
    payload = asdict(review)
    payload["label_kind"] = review.label_kind.value
    payload["scenario"] = None if review.scenario is None else review.scenario.value
    payload["review_outcome"] = review.review_outcome.value
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_event_reviews(
    event_dir: Path,
    session_id: str,
    event_id: str,
) -> tuple[StageCReviewV1, ...]:
    if not event_dir.exists():
        return ()
    _validate_directory(event_dir, "review event", allow_missing=False)
    entries = tuple(sorted(event_dir.iterdir()))
    if any(path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in entries):
        raise StageCReviewIntegrityError(
            "review event directory may contain only JSON review files"
        )
    reviews = tuple(load_stage_c_review(path) for path in entries)
    if any(
        review.session_id != session_id or review.event_id != event_id
        for review in reviews
    ):
        raise StageCReviewIntegrityError("review path contains mismatched identifiers")
    if len({review.review_id for review in reviews}) != len(reviews):
        raise StageCReviewIntegrityError("review identifiers must be unique")
    if len({review.event_manifest_sha256 for review in reviews}) > 1:
        raise StageCReviewIntegrityError("review chain contains inconsistent event hashes")
    return reviews


def _active_review(reviews: tuple[StageCReviewV1, ...]) -> StageCReviewV1 | None:
    if not reviews:
        return None
    by_id = {review.review_id: review for review in reviews}
    for review in reviews:
        parent = review.supersedes_review_id
        if parent is not None and parent not in by_id:
            raise StageCReviewIntegrityError("review supersedes an unknown review")

    states: dict[str, int] = {}

    def visit(review_id: str) -> None:
        state = states.get(review_id, 0)
        if state == 1:
            raise StageCReviewIntegrityError("review history contains a cycle")
        if state == 2:
            return
        states[review_id] = 1
        parent = by_id[review_id].supersedes_review_id
        if parent is not None:
            visit(parent)
        states[review_id] = 2

    for review_id in by_id:
        visit(review_id)
    superseded = {
        review.supersedes_review_id
        for review in reviews
        if review.supersedes_review_id is not None
    }
    active_ids = tuple(sorted(set(by_id) - superseded))
    if len(active_ids) != 1:
        raise StageCReviewIntegrityError("review history has multiple active reviews")
    return by_id[active_ids[0]]


def _validate_identifier(value: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("review, event, and session identifiers must be path-safe")


def _validate_position_id(value: str) -> None:
    if not isinstance(value, str) or _POSITION_ID.fullmatch(value) is None:
        raise ValueError("position id must contain 32 lowercase hexadecimal characters")


def _validate_move(value: str) -> None:
    if not isinstance(value, str) or _UCI.fullmatch(value) is None:
        raise ValueError("move must use four-character Xiangqi UCI coordinates")


def _validate_moves(values: tuple[str, ...], *, maximum: int) -> None:
    if not isinstance(values, tuple) or len(values) > maximum:
        limit = "three" if maximum == 3 else str(maximum)
        raise ValueError(f"review moves must contain at most {limit} entries")
    for value in values:
        _validate_move(value)


def _parse_utc(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("created_at_utc must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as exc:
        raise ValueError("created_at_utc must be a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _validate_directory(path: Path, name: str, *, allow_missing: bool) -> None:
    if path.is_symlink():
        raise StageCReviewIntegrityError(f"{name} must not be a symlink")
    if path.exists() and not path.is_dir():
        raise StageCReviewIntegrityError(f"{name} must be a directory")
    if not allow_missing and not path.exists():
        raise StageCReviewIntegrityError(f"{name} does not exist")


def _assert_within_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise StageCReviewIntegrityError("review path escapes its root") from exc


def _tree_size(root: Path) -> int:
    if not root.exists():
        return 0

    def size(directory: Path) -> int:
        total = 0
        for entry in directory.iterdir():
            if entry.is_symlink():
                raise StageCReviewIntegrityError("review tree must not contain symlinks")
            if entry.is_dir():
                total += size(entry)
            elif entry.is_file():
                total += entry.stat().st_size
            else:
                raise StageCReviewIntegrityError("review tree contains an unsafe entry")
        return total

    return size(root)


def _remove_empty_parents(event_dir: Path, session_dir: Path, root: Path) -> None:
    if event_dir.exists() and not any(event_dir.iterdir()):
        event_dir.rmdir()
    if session_dir.exists() and not any(session_dir.iterdir()):
        session_dir.rmdir()
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("value must be an integer")
    return value


def _moves(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError("moves must be a list of strings")
    return tuple(value)
