from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from xiangqi_agent.diagnostics.stage_c_gate import (
    HumanAiStageCGate,
    HumanAiStageCReport,
    StageCGateIntegrityError,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.output.resolve() == args.frozen_manifest.resolve():
            raise StageCGateIntegrityError(
                "report output must not overwrite the frozen manifest"
            )
        report = HumanAiStageCGate().evaluate(args.frozen_manifest)
        _write_report_atomic(args.output, report)
    except (OSError, StageCGateIntegrityError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "integrity_error", "message": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.release_pass else 1


def _write_report_atomic(path: Path, report: HumanAiStageCReport) -> None:
    if not isinstance(path, Path):
        raise TypeError("report output must be a Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    payload = json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one frozen human-vs-AI Stage C blind set"
    )
    parser.add_argument("frozen_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
