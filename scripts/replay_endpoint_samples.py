from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from xiangqi_agent.diagnostics.endpoint_replay import (
    EndpointReplayer,
    EndpointReplayGate,
    ReplayThresholds,
)
from xiangqi_agent.vision.endpoint_features import InstanceTransferExtractor


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    replayer = EndpointReplayer(
        InstanceTransferExtractor(max_shift=args.max_shift),
        EndpointReplayGate(
            ReplayThresholds(
                max_instance_distance=args.max_instance_distance,
                min_source_change=args.min_source_change,
                min_target_change=args.min_target_change,
                profile_version=args.profile_version,
            )
        ),
    )
    result = replayer.replay(args.sample_dir)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one privacy-safe endpoint sample")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--max-instance-distance", type=float, required=True)
    parser.add_argument("--min-source-change", type=float, required=True)
    parser.add_argument("--min-target-change", type=float, required=True)
    parser.add_argument("--profile-version", required=True)
    parser.add_argument("--max-shift", type=int, default=3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
