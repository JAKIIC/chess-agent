from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from xiangqi_agent.diagnostics.stage_c_gate import (
    StageCGateIntegrityError,
    freeze_reviewed_human_ai_stage_c,
    load_frozen_stage_c_manifest,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output = freeze_reviewed_human_ai_stage_c(args.sample_root, args.output)
        manifest = load_frozen_stage_c_manifest(output)
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
    print(
        json.dumps(
            {
                "status": "frozen",
                "sample_count": len(manifest.samples),
                "feature_version": manifest.feature_version,
                "threshold_profile_version": manifest.threshold_profile.profile_version,
                "output": output.name,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a privacy-safe human-vs-AI Stage C blind set"
    )
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--output", required=True, help="one new filename inside sample_root")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
