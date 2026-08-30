from __future__ import annotations

import argparse
import json
from pathlib import Path

from xiangqi_agent.engine.installer import install_pikafish


def main(argv: list[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download, verify, extract, and probe the pinned official Pikafish release."
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=repository / "assets" / "pikafish-2026-01-02.json",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=repository / ".local" / "pikafish",
        help="Gitignored local installation root.",
    )
    arguments = parser.parse_args(argv)
    installed = install_pikafish(arguments.lock.resolve(), arguments.root.resolve())
    print(
        json.dumps(
            {
                "tag": installed.tag,
                "commit": installed.commit,
                "executable": str(installed.executable),
                "eval_file": str(installed.eval_file),
                "asset_sha256": installed.asset_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
