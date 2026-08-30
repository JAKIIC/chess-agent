from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFontDatabase

_PREFERRED_FAMILIES = ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei")
_WINDOWS_FONT_FILES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
)


def ensure_cjk_font() -> str:
    available = set(QFontDatabase.families())
    for family in _PREFERRED_FAMILIES:
        if family in available:
            return family
    for path in _WINDOWS_FONT_FILES:
        if not path.is_file():
            continue
        font_id = QFontDatabase.addApplicationFont(str(path))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            return families[-1] if "UI" in families[-1] else families[0]
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
