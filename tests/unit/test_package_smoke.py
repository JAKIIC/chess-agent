from xiangqi_agent import __version__
from xiangqi_agent.__main__ import main


def test_package_has_version_and_entrypoint() -> None:
    assert __version__ == "0.1.0"
    assert main(["--check"]) == 0
