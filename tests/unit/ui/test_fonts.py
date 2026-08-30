from __future__ import annotations

from PySide6.QtGui import QFont, QFontMetrics

from xiangqi_agent.ui.fonts import ensure_cjk_font


def test_ensure_cjk_font_can_render_chinese_in_headless_windows_qt(qapp: object) -> None:
    family = ensure_cjk_font()
    metrics = QFontMetrics(QFont(family))

    assert metrics.inFontUcs4(ord("棋"))
