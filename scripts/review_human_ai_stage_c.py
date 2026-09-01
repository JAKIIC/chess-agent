from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from xiangqi_agent.diagnostics.stage_c_quarantine import QuarantineEventLoader
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewOutcome,
    StageCReviewService,
    StageCReviewStore,
    legal_review_choices,
    project_review_prefix,
)
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario
from xiangqi_agent.domain.fen import parse_fen

_DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / ".local"


def main(
    argv: Sequence[str] | None = None,
    *,
    allowed_local_root: Path | None = None,
) -> int:
    args = _parse_args(argv)
    local_root = allowed_local_root or _DEFAULT_LOCAL_ROOT
    try:
        _validate_paths(args.event_dir, args.review_root, local_root)
        loaded = QuarantineEventLoader().load(args.event_dir)
        if args.list_legal:
            if any(
                value is not None
                for value in (
                    args.label,
                    args.outcome,
                    args.scenario,
                    args.supersedes_review_id,
                )
            ):
                raise ValueError("legal listing cannot also submit review fields")
            if len(args.moves_uci) > 1:
                raise ValueError("legal selector accepts at most one move prefix")
            board = project_review_prefix(
                parse_fen(loaded.metadata.confirmed_fen),
                args.moves_uci,
            )
            choices = legal_review_choices(board)
            print(
                json.dumps(
                    {
                        "status": "legal_choices",
                        "event_id": loaded.metadata.event_id,
                        "prefix_length": len(args.moves_uci),
                        "choices": [
                            {"chinese": choice.chinese, "uci": choice.uci}
                            for choice in choices
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        draft = _draft_from_args(args)
        output = StageCReviewService(
            StageCReviewStore(args.review_root, enabled=True)
        ).submit(args.event_dir, draft)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "review_error", "code": _error_code(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "reviewed",
                "event_id": output.parent.name,
                "review_id": output.stem,
            },
            sort_keys=True,
        )
    )
    return 0


def _draft_from_args(args: argparse.Namespace) -> StageCReviewDraft:
    if args.label is None:
        raise ValueError("--label is required when submitting a review")
    label = StageCLabelKind(args.label)
    if label is StageCLabelKind.VALID_TWO_PLY:
        if args.outcome is None:
            raise ValueError("valid_two_ply requires --outcome")
        outcome = StageCReviewOutcome(args.outcome)
    elif label is StageCLabelKind.EXPECTED_REJECTION:
        if args.outcome is not None:
            raise ValueError("expected_rejection derives its review outcome")
        outcome = StageCReviewOutcome.EXPECTED_REJECTION
    else:
        if args.outcome is not None:
            raise ValueError("discard derives its review outcome")
        outcome = StageCReviewOutcome.DISCARDED
    scenario = None if args.scenario is None else StageCScenario(args.scenario)
    return StageCReviewDraft(
        label_kind=label,
        moves_uci=args.moves_uci,
        scenario=scenario,
        review_outcome=outcome,
        supersedes_review_id=args.supersedes_review_id,
    )


def _validate_paths(event_dir: Path, review_root: Path, local_root: Path) -> None:
    if not all(isinstance(path, Path) for path in (event_dir, review_root, local_root)):
        raise TypeError("event, review, and local roots must be Paths")
    if local_root.name != ".local":
        raise ValueError("allowed local root must be named .local")
    quarantine_root = local_root / "stage-c-quarantine"
    expected_review_root = local_root / "stage-c-reviews"
    if any(
        path.is_symlink()
        for path in (
            local_root,
            quarantine_root,
            event_dir.parent,
            expected_review_root,
        )
    ):
        raise ValueError("Stage C local paths must not contain symlinked roots")
    if review_root.resolve() != expected_review_root.resolve():
        raise ValueError("review root must be the fixed stage-c-reviews directory")
    if event_dir.is_symlink():
        raise ValueError("event directory must not be a symlink")
    try:
        relative = event_dir.resolve().relative_to(quarantine_root.resolve())
    except ValueError as exc:
        raise ValueError("event must be under the fixed quarantine root") from exc
    if len(relative.parts) != 2:
        raise ValueError("event path must contain one session and one event id")


def _error_code(error: BaseException) -> str:
    if isinstance(error, FileExistsError):
        return "immutable_conflict"
    if "integrity" in type(error).__name__.lower():
        return "integrity_failure"
    return "invalid_review"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review one quarantined human-vs-AI event locally"
    )
    parser.add_argument("--event-dir", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--list-legal", action="store_true")
    parser.add_argument("--label", choices=tuple(item.value for item in StageCLabelKind))
    parser.add_argument("--moves", default="")
    parser.add_argument(
        "--outcome",
        choices=(
            StageCReviewOutcome.CANDIDATE_CONFIRMED.value,
            StageCReviewOutcome.LEGAL_MOVE_CORRECTION.value,
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(item.value for item in StageCScenario),
    )
    parser.add_argument("--supersedes-review-id")
    args = parser.parse_args(argv)
    if args.moves:
        parts = tuple(args.moves.split(","))
        if any(not part for part in parts):
            parser.error("--moves must be a comma-separated sequence without empty items")
        args.moves_uci = parts
    else:
        args.moves_uci = ()
    return args


if __name__ == "__main__":
    raise SystemExit(main())
