from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Protocol

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xiangqi_agent.capture.monitor import (
    CaptureMonitor,
    CaptureMonitorUpdate,
    MonitorStatus,
)
from xiangqi_agent.capture.protocol import FrameSource
from xiangqi_agent.capture.windows_capture_source import WindowsCaptureSource
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.platform.windows import WindowInfo, WindowsWindowCatalog
from xiangqi_agent.sync.live_session import (
    LiveSyncSession,
    LiveSyncStatus,
    LiveSyncUpdate,
)
from xiangqi_agent.vision.geometry import GeometryError, parse_normalized_quad

DEFAULT_QUAD = "0.315,0.132;0.678,0.132;0.678,0.862;0.315,0.862"


class WindowCatalog(Protocol):
    def list_candidates(self) -> tuple[WindowInfo, ...]: ...


type SourceFactory = Callable[[WindowInfo], FrameSource]
type BoardProvider = Callable[[], BoardState]


class _CaptureBridge(QObject):
    update = Signal(object)


class CapturePanel(QWidget):
    """Connect a user-selected visible window to fixed manual board geometry."""

    sync_update = Signal(object)

    def __init__(
        self,
        *,
        catalog: WindowCatalog | None = None,
        source_factory: SourceFactory | None = None,
        board_provider: BoardProvider | None = None,
        patch_size: int = 48,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog or WindowsWindowCatalog()
        self._source_factory = source_factory or _default_source_factory
        self._board_provider = board_provider
        self._patch_size = patch_size
        self._windows: tuple[WindowInfo, ...] = ()
        self._monitor: CaptureMonitor | None = None
        self._session: LiveSyncSession | None = None
        self._bridge = _CaptureBridge(self)
        self._bridge.update.connect(self._show_update)

        self.window_combo = QComboBox()
        self.refresh_button = QPushButton("刷新窗口")
        self.connect_button = QPushButton("连接")
        self.connect_button.setEnabled(False)
        self.quad_input = QLineEdit(DEFAULT_QUAD)
        self.quad_input.setPlaceholderText("左上;右上;右下;左下（归一化 x,y）")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItem("红方在下", Orientation.RED_BOTTOM.value)
        self.orientation_combo.addItem("黑方在下", Orientation.BLACK_BOTTOM.value)
        self.status_label = QLabel("尚未选择天天象棋窗口")
        self.status_label.setWordWrap(True)

        first_row = QHBoxLayout()
        first_row.addWidget(self.window_combo, 1)
        first_row.addWidget(self.refresh_button)
        first_row.addWidget(self.connect_button)
        second_row = QHBoxLayout()
        second_row.addWidget(self.quad_input, 1)
        second_row.addWidget(self.orientation_combo)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(first_row)
        layout.addLayout(second_row)
        layout.addWidget(self.status_label)

        self.refresh_button.clicked.connect(self._refresh_windows)
        self.connect_button.clicked.connect(self._toggle_capture)

    def close_capture(self) -> None:
        monitor = self._monitor
        session = self._session
        self._monitor = None
        self._session = None
        if monitor is not None:
            monitor.close()
        if session is not None:
            session.close()
        self.connect_button.setText("连接")
        self.window_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.quad_input.setEnabled(True)
        self.orientation_combo.setEnabled(True)
        self.connect_button.setEnabled(bool(self._windows))

    def set_board_provider(self, provider: BoardProvider) -> None:
        self._board_provider = provider

    def recover(self, board: BoardState) -> None:
        session = self._session
        if session is None:
            return
        try:
            orientation = Orientation(str(self.orientation_combo.currentData()))
            quad = parse_normalized_quad(self.quad_input.text().strip())
            session.recover(replace(board, orientation=orientation), quad=quad)
        except (GeometryError, RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(f"同步恢复失败：{exc}")

    def _refresh_windows(self) -> None:
        self.close_capture()
        try:
            self._windows = self._catalog.list_candidates()
        except (OSError, RuntimeError) as exc:
            self._windows = ()
            self.window_combo.clear()
            self.connect_button.setEnabled(False)
            self.status_label.setText(f"窗口枚举失败：{exc}")
            return

        self.window_combo.clear()
        for window in self._windows:
            self.window_combo.addItem(_window_label(window))
        if not self._windows:
            self.connect_button.setEnabled(False)
            self.status_label.setText("没有找到可用的天天象棋候选窗口")
            return
        preferred = next(
            (index for index, window in enumerate(self._windows) if "天天象棋" in window.title),
            0,
        )
        self.window_combo.setCurrentIndex(preferred)
        self.connect_button.setEnabled(True)
        self.status_label.setText(f"找到 {len(self._windows)} 个候选窗口，请确认后连接")

    def _toggle_capture(self) -> None:
        if self._monitor is not None or self._session is not None:
            self.close_capture()
            self.status_label.setText("已断开窗口捕获")
            return
        index = self.window_combo.currentIndex()
        if index < 0 or index >= len(self._windows):
            self.status_label.setText("请先刷新并选择目标窗口")
            return
        try:
            quad = parse_normalized_quad(self.quad_input.text().strip())
            orientation = Orientation(str(self.orientation_combo.currentData()))
            source = self._source_factory(self._windows[index])
        except (GeometryError, TypeError, ValueError) as exc:
            self.status_label.setText(f"四角标定无效：{exc}")
            return

        if self._board_provider is None:
            self._monitor = CaptureMonitor(
                source,
                quad,
                orientation=orientation,
                on_update=self._bridge.update.emit,
            )
        else:
            try:
                board = replace(self._board_provider(), orientation=orientation)
            except (RuntimeError, TypeError, ValueError) as exc:
                self.status_label.setText(f"当前局面无效：{exc}")
                return
            self._session = LiveSyncSession(
                source,
                board,
                quad,
                on_update=self._bridge.update.emit,
                patch_size=self._patch_size,
            )
        self.connect_button.setText("断开")
        self.window_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.quad_input.setEnabled(False)
        self.orientation_combo.setEnabled(False)
        if self._session is not None:
            self._session.start()
        elif self._monitor is not None:
            self._monitor.start()

    def _show_update(self, update: CaptureMonitorUpdate | LiveSyncUpdate) -> None:
        if isinstance(update, LiveSyncUpdate):
            self._show_live_update(update)
            return
        if update.status is MonitorStatus.CONNECTING:
            self.status_label.setText("正在连接，等待第一帧…")
        elif update.status is MonitorStatus.WATCHING:
            width, height = update.frame_size or (0, 0)
            self.status_label.setText(
                f"捕获正常：{width}×{height}，{update.point_count} 个交点；自动同步未启用"
            )
        elif update.status is MonitorStatus.GEOMETRY_REBOUND:
            width, height = update.frame_size or (0, 0)
            self.status_label.setText(
                f"窗口尺寸已变化，几何已自动适配：{width}×{height}，"
                f"{update.point_count} 个交点；同步尚未确认"
            )
        elif update.status is MonitorStatus.GEOMETRY_INVALID:
            self.status_label.setText("窗口尺寸已变化，原四角标定失效；请断开后重新标定")
        elif update.status is MonitorStatus.CLOSED:
            self.status_label.setText("目标窗口已关闭，请重新刷新选择")
            self.close_capture()
        else:
            self.status_label.setText(f"捕获失败：{update.message}")
            self.close_capture()

    def _show_live_update(self, update: LiveSyncUpdate) -> None:
        self.sync_update.emit(update)
        width, height = update.frame_size or (0, 0)
        if update.status is LiveSyncStatus.CONNECTING:
            self.status_label.setText("正在连接，等待稳定棋盘画面…")
        elif update.status is LiveSyncStatus.BASELINE_READY:
            self.status_label.setText(
                f"实时同步已启动：{width}×{height}，{update.point_count} 个交点"
            )
        elif update.status is LiveSyncStatus.WATCHING:
            self.status_label.setText("正在监听已确认局面")
        elif update.status is LiveSyncStatus.WAITING_FOR_STABLE:
            self.status_label.setText("检测到画面变化，正在等待落子动画结束…")
        elif update.status is LiveSyncStatus.WAITING_FOR_ENDPOINT:
            self.status_label.setText("已看到棋子点选，正在等待完成走棋…")
        elif update.status is LiveSyncStatus.MOVE_ACCEPTED:
            move = update.move.uci if update.move is not None else "未知"
            self.status_label.setText(f"已同步一步：{move}")
        elif update.status is LiveSyncStatus.GEOMETRY_REBOUND:
            self.status_label.setText(
                f"窗口缩放后已安全适配：{width}×{height}，{update.point_count} 个交点"
            )
        elif update.status is LiveSyncStatus.PAUSED_AMBIGUOUS:
            self.status_label.setText("变化无法唯一确认；已保留最后局面，请用 FEN 明确恢复")
        elif update.status in (
            LiveSyncStatus.CONTEXT_INVALID,
            LiveSyncStatus.MANUAL_RECOVERY_REQUIRED,
        ):
            self.status_label.setText("窗口或标定已变化；请确认 FEN 并重新标定后恢复")
        elif update.status is LiveSyncStatus.CLOSED:
            self.status_label.setText("目标窗口已关闭，请重新刷新选择")
            self.close_capture()
        else:
            self.status_label.setText(f"实时同步失败：{update.message}")
            self.close_capture()


def _default_source_factory(window: WindowInfo) -> FrameSource:
    return WindowsCaptureSource(window, fps=20)


def _window_label(window: WindowInfo) -> str:
    name = "天天象棋" if "天天象棋" in window.title else "候选窗口"
    width, height = window.client_size
    return f"{name} · {window.process_name} · {width}×{height}"
