from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv or [])
    if args == ["--check"]:
        return 0
    from xiangqi_agent.bootstrap import run

    return run()


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
