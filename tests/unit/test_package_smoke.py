from xiangqi_agent import __version__
from xiangqi_agent.__main__ import main


def test_package_has_version_and_entrypoint() -> None:
    assert __version__ == "0.1.0"
    assert main(["--check"]) == 0


def test_pyside6_qt_runtime_loads() -> None:
    from PySide6 import QtCore

    assert QtCore.qVersion().split(".")[0] == "6"
