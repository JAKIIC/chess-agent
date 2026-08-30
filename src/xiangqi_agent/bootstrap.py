from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from xiangqi_agent.ui.main_window import MainWindow


def run() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()
