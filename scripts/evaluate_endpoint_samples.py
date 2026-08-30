from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path

from xiangqi_agent.diagnostics.endpoint_replay import (
    EndpointReplayer,
    EndpointReplayGate,
    EndpointReplayResult,
    ReplayThresholds,
)
from xiangqi_agent.vision.endpoint_features import InstanceTransferExtractor


@dataclass(frozen=True, slots=True)
class DatasetEvaluationReport:
    total_samples: int
    positive_samples: int
    negative_samples: int
    top1_correct: int
    top1_accuracy: float
    accepted_samples: int
    correct_accepts: int
    false_accepts: int
    accepted_precision: float
    coverage: float
    rejection_counts: dict[str, int]
    p95_runtime_ns: int


def evaluate_results(results: tuple[EndpointReplayResult, ...]) -> DatasetEvaluationReport:
    positives = tuple(result for result in results if result.actual_uci is not None)
    negatives = tuple(result for result in results if result.actual_uci is None)
    top1_correct = sum(result.probe_uci == result.actual_uci for result in positives)
    accepted = tuple(result for result in results if result.accepted)
    correct_accepts = sum(
        result.actual_uci is not None and result.probe_uci == result.actual_uci
        for result in accepted
    )
    false_accepts = len(accepted) - correct_accepts
    rejection_counts = Counter(
        reason for result in results for reason in result.rejection_reasons
    )
    runtimes = sorted(result.runtime_ns for result in results)
    p95_runtime_ns = runtimes[max(0, ceil(len(runtimes) * 0.95) - 1)] if runtimes else 0
    return DatasetEvaluationReport(
        total_samples=len(results),
        positive_samples=len(positives),
        negative_samples=len(negatives),
        top1_correct=top1_correct,
        top1_accuracy=_ratio(top1_correct, len(positives)),
        accepted_samples=len(accepted),
        correct_accepts=correct_accepts,
        false_accepts=false_accepts,
        accepted_precision=_ratio(correct_accepts, len(accepted)),
        coverage=_ratio(correct_accepts, len(positives)),
        rejection_counts=dict(sorted(rejection_counts.items())),
        p95_runtime_ns=p95_runtime_ns,
    )


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
    sample_dirs = tuple(sorted(path.parent for path in args.sample_root.rglob("manifest.json")))
    results = tuple(replayer.replay(path) for path in sample_dirs)
    print(json.dumps(asdict(evaluate_results(results)), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a labeled endpoint sample directory")
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--max-instance-distance", type=float, required=True)
    parser.add_argument("--min-source-change", type=float, required=True)
    parser.add_argument("--min-target-change", type=float, required=True)
    parser.add_argument("--profile-version", required=True)
    parser.add_argument("--max-shift", type=int, default=3)
    return parser.parse_args(argv)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
