from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from xiangqi_agent.diagnostics.stage_c_promotion import (
    PromotionStatus,
    StageCPromotionBlockedError,
    StageCPromotionService,
    StageCPromotionVerifier,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    load_stage_c_review,
)

_DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parents[1] / ".local"


def main(
    argv: Sequence[str] | None = None,
    *,
    allowed_local_root: Path | None = None,
) -> int:
    args = _parse_args(argv)
    local_root = allowed_local_root or _DEFAULT_LOCAL_ROOT
    try:
        _validate_paths(
            args.event_dir,
            args.review_path,
            args.reviewed_root,
            local_root,
        )
        review = load_stage_c_review(args.review_path)
        if review.label_kind is StageCLabelKind.DISCARD:
            _print_error("discarded")
            return 2
        verifier = StageCPromotionVerifier()
        decision = verifier.verify(args.event_dir, args.review_path)
        if decision.status is PromotionStatus.NEEDS_REVIEW:
            _print_error(decision.reason_codes[0])
            return 1
        if decision.status is PromotionStatus.REJECTED:
            _print_error(decision.reason_codes[0])
            return 2
        output = StageCPromotionService(verifier).promote(
            args.event_dir,
            args.review_path,
            args.reviewed_root,
        )
    except StageCPromotionBlockedError as exc:
        _print_error(exc.decision.reason_codes[0])
        return 1 if exc.decision.status is PromotionStatus.NEEDS_REVIEW else 2
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _print_error(_error_code(exc))
        return 2
    print(output.name)
    return 0


def _validate_paths(
    event_dir: Path,
    review_path: Path,
    reviewed_root: Path,
    local_root: Path,
) -> None:
    if not all(
        isinstance(path, Path) for path in (event_dir, review_path, reviewed_root, local_root)
    ):
        raise TypeError("promotion paths must be Paths")
    if local_root.name != ".local" or local_root.is_symlink():
        raise ValueError("allowed local root must be a real .local directory")
    quarantine_root = local_root / "stage-c-quarantine"
    review_root = local_root / "stage-c-reviews"
    expected_reviewed = local_root / "stage-c-reviewed"
    if reviewed_root.resolve() != expected_reviewed.resolve():
        raise ValueError("reviewed root must be the fixed stage-c-reviewed directory")
    for path, root, depth in (
        (event_dir, quarantine_root, 2),
        (review_path, review_root, 3),
    ):
        if path.is_symlink() or root.is_symlink():
            raise ValueError("promotion sources must not be symlinked")
        try:
            relative = path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("promotion source escapes its fixed root") from exc
        if len(relative.parts) != depth:
            raise ValueError("promotion source has an invalid path layout")


def _print_error(code: str) -> None:
    print(
        json.dumps({"status": "promotion_error", "code": code}, sort_keys=True),
        file=sys.stderr,
    )


def _error_code(error: BaseException) -> str:
    if isinstance(error, FileExistsError):
        return "target_exists"
    if "integrity" in type(error).__name__.lower():
        return "integrity_failure"
    return "invalid_configuration"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote one verified Stage C review into a self-contained V2 sample"
    )
    parser.add_argument("--event-dir", required=True, type=Path)
    parser.add_argument("--review-path", required=True, type=Path)
    parser.add_argument("--reviewed-root", required=True, type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
