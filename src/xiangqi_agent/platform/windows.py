from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WindowSelectionError(RuntimeError):
    """No usable top-level target window was selected."""


@dataclass(frozen=True, slots=True)
class WindowInfo:
    hwnd: int
    title: str
    process_name: str
    client_size: tuple[int, int]
    is_minimized: bool = False


def filter_target_windows(windows: tuple[WindowInfo, ...]) -> tuple[WindowInfo, ...]:
    filtered = (
        item
        for item in windows
        if item.hwnd > 0
        and not item.is_minimized
        and item.title.strip() == "天天象棋"
        and item.client_size[0] > 0
        and item.client_size[1] > 0
    )
    return tuple(filtered)


def select_window(candidates: tuple[WindowInfo, ...], hwnd: int) -> WindowInfo:
    if not candidates:
        raise WindowSelectionError("no candidate target windows are available")
    for candidate in candidates:
        if candidate.hwnd == hwnd:
            return candidate
    raise WindowSelectionError("selected target window was not found")


class WindowsWindowCatalog:
    def list_candidates(self) -> tuple[WindowInfo, ...]:
        if os.name != "nt":
            raise OSError("window enumeration is only available on Windows")
        return filter_target_windows(_enumerate_top_level_windows())


def _enumerate_top_level_windows() -> tuple[WindowInfo, ...]:
    user32: Any = ctypes.windll.user32
    windows: list[WindowInfo] = []

    def collect(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title_length = int(user32.GetWindowTextLengthW(hwnd))
        if title_length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(title_length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, len(title_buffer))
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        windows.append(
            WindowInfo(
                hwnd=int(hwnd),
                title=title_buffer.value,
                process_name=_process_name(hwnd),
                client_size=(width, height),
                is_minimized=bool(user32.IsIconic(hwnd)),
            )
        )
        return True

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    if not user32.EnumWindows(callback_type(collect), 0):
        raise OSError(ctypes.get_last_error(), "EnumWindows failed")
    return tuple(windows)


def _process_name(hwnd: int) -> str:
    user32: Any = ctypes.windll.user32
    kernel32: Any = ctypes.windll.kernel32
    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    process = kernel32.OpenProcess(0x1000, False, process_id.value)
    if not process:
        return "unknown"
    try:
        buffer = ctypes.create_unicode_buffer(32_768)
        size = wintypes.DWORD(len(buffer))
        if not kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size)):
            return "unknown"
        return Path(buffer.value).name
    finally:
        kernel32.CloseHandle(process)
