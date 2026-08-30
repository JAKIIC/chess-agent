import pytest

from xiangqi_agent.platform.windows import (
    WindowInfo,
    WindowSelectionError,
    filter_target_windows,
    select_window,
)


def test_filter_target_windows_keeps_visible_wechat_or_named_target() -> None:
    windows = (
        WindowInfo(1, "微信", "Weixin.exe", (900, 1200)),
        WindowInfo(2, "天天象棋", "host.exe", (800, 900)),
        WindowInfo(3, "", "Weixin.exe", (800, 900)),
        WindowInfo(4, "记事本", "notepad.exe", (800, 600)),
        WindowInfo(5, "微信", "Weixin.exe", (0, 0)),
        WindowInfo(6, "天天象棋", "WeChatAppEx.exe", (800, 900), is_minimized=True),
    )

    assert [item.hwnd for item in filter_target_windows(windows)] == [1, 2]


def test_manual_selection_requires_a_real_candidate() -> None:
    candidates = (WindowInfo(10, "微信", "Weixin.exe", (900, 1200)),)
    assert select_window(candidates, 10) == candidates[0]
    with pytest.raises(WindowSelectionError, match="not found"):
        select_window(candidates, 99)
    with pytest.raises(WindowSelectionError, match="no candidate"):
        select_window((), 10)
