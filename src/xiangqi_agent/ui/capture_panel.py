from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import (
    QCheckBox,
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
from xiangqi_agent.diagnostics.stage_c_live_capture import StageCTerminalEventWriter
from xiangqi_agent.domain.board import BoardState, Orientation
from xiangqi_agent.platform.windows import WindowInfo, WindowsWindowCatalog
from xiangqi_agent.sync.live_session import (
    LiveSyncSession,
    LiveSyncStatus,
    LiveSyncUpdate,
)
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.vision.geometry import (
    GeometryError,
    NormalizedQuad,
    parse_normalized_quad,
)
from xiangqi_agent.vision.occupancy import (
    KnownPositionOccupancyObserver,
    OccupancyObserver,
)

DEFAULT_QUAD = "0.315,0.132;0.678,0.132;0.678,0.862;0.315,0.862"


class WindowCatalog(Protocol):
    def list_candidates(self) -> tuple[WindowInfo, ...]: ...


type SourceFactory = Callable[[WindowInfo], FrameSource]
type BoardProvider = Callable[[], BoardState]
type OccupancyObserverFactory = Callable[[BoardState], OccupancyObserver]


class EvidenceWriter(Protocol):
    def record(
        self,
        update: LiveSyncUpdate,
        *,
        board: BoardState,
        quad: NormalizedQuad,
        session_id: str,
        event_id: str,
        client_size: tuple[int, int],
        generation_id: int,
    ) -> Path: ...


class _CaptureBridge(QObject):
    update = Signal(object)


@dataclass(frozen=True, slots=True)
class _TaggedCaptureUpdate:
    generation: int
    update: CaptureMonitorUpdate | LiveSyncUpdate


class CaptureRecoveryStatus(StrEnum):
    NOT_CONNECTED = "not_connected"
    PENDING = "pending"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CaptureRecoveryResult:
    status: CaptureRecoveryStatus
    board: BoardState | None
    message: str
    recovery_id: int | None = None


class CapturePanel(QWidget):
    """Connect a user-selected visible window to fixed manual board geometry."""

    sync_update = Signal(object)
    session_reset = Signal()
    review_event_ready = Signal(object)

    def __init__(
        self,
        *,
        catalog: WindowCatalog | None = None,
        source_factory: SourceFactory | None = None,
        board_provider: BoardProvider | None = None,
        patch_size: int = 48,
        local_root: Path | None = None,
        evidence_writer: EvidenceWriter | None = None,
        occupancy_observer_factory: OccupancyObserverFactory | None = None,
        session_id_factory: Callable[[], str] | None = None,
        event_id_factory: Callable[[], str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._catalog = catalog or WindowsWindowCatalog()
        self._source_factory = source_factory or _default_source_factory
        self._board_provider = board_provider
        self._patch_size = patch_size
        resolved_local_root = local_root or Path.cwd() / ".local"
        self._evidence_writer = evidence_writer or StageCTerminalEventWriter(
            resolved_local_root
        )
        self._occupancy_observer_factory = (
            occupancy_observer_factory or KnownPositionOccupancyObserver
        )
        self._session_id_factory = session_id_factory or (lambda: uuid4().hex)
        self._event_id_factory = event_id_factory or (lambda: uuid4().hex)
        self._windows: tuple[WindowInfo, ...] = ()
        self._monitor: CaptureMonitor | None = None
        self._session: LiveSyncSession | None = None
        self._generation = 0
        self._evidence_enabled_for_session = False
        self._evidence_board: BoardState | None = None
        self._evidence_quad: NormalizedQuad | None = None
        self._evidence_client_size: tuple[int, int] | None = None
        self._evidence_session_id: str | None = None
        self._pending_review_event: Path | None = None
        self._pending_review_next_board: BoardState | None = None
        self._evidence_out_of_sync = False
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
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("严格单步", SyncMode.STRICT_SINGLE.value)
        self.mode_combo.addItem(
            "人机练习（可同步连续应手）",
            SyncMode.HUMAN_VS_AI.value,
        )
        self.evidence_checkbox = QCheckBox("帮助改进识别（本地保存小裁片）")
        self.evidence_checkbox.setChecked(False)
        self.status_label = QLabel("尚未选择天天象棋窗口")
        self.status_label.setWordWrap(True)

        first_row = QHBoxLayout()
        first_row.addWidget(self.window_combo, 1)
        first_row.addWidget(self.refresh_button)
        first_row.addWidget(self.connect_button)
        second_row = QHBoxLayout()
        second_row.addWidget(self.quad_input, 1)
        second_row.addWidget(self.orientation_combo)
        second_row.addWidget(self.mode_combo)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(first_row)
        layout.addLayout(second_row)
        layout.addWidget(self.evidence_checkbox)
        layout.addWidget(self.status_label)

        self.refresh_button.clicked.connect(self._refresh_windows)
        self.connect_button.clicked.connect(self._toggle_capture)

    def close_capture(self) -> None:
        self._generation += 1
        monitor = self._monitor
        session = self._session
        had_active_session = monitor is not None or session is not None
        self._monitor = None
        self._session = None
        self._clear_evidence_session()
        if monitor is not None:
            monitor.close()
        if session is not None:
            session.close()
        self.connect_button.setText("连接")
        self.window_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.quad_input.setEnabled(True)
        self.orientation_combo.setEnabled(True)
        self.mode_combo.setEnabled(True)
        self.evidence_checkbox.setEnabled(True)
        self.connect_button.setEnabled(bool(self._windows))
        if had_active_session:
            self.session_reset.emit()

    def set_board_provider(self, provider: BoardProvider) -> None:
        self._board_provider = provider

    def finish_review(self, event_id: str, status: str) -> None:
        pending = self._pending_review_event
        if pending is None or pending.name != event_id:
            return
        if self._evidence_out_of_sync:
            self.close_capture()
            self.status_label.setText("复核期间棋盘继续变化；请重新连接后再采集证据")
            return
        self._evidence_board = self._pending_review_next_board
        self._pending_review_event = None
        self._pending_review_next_board = None
        self.status_label.setText(
            "本地复核已完成，可继续监听" if status == "promoted" else "事件已丢弃，可继续监听"
        )

    def recover(self, board: BoardState) -> CaptureRecoveryResult:
        session = self._session
        try:
            orientation = Orientation(str(self.orientation_combo.currentData()))
            recovered_board = replace(board, orientation=orientation)
            if session is None:
                return CaptureRecoveryResult(
                    CaptureRecoveryStatus.NOT_CONNECTED,
                    recovered_board,
                    "no active capture session",
                )
            quad = parse_normalized_quad(self.quad_input.text().strip())
            recovery_id = session.recover(recovered_board, quad=quad)
        except (GeometryError, RuntimeError, TypeError, ValueError) as exc:
            message = f"同步恢复失败：{exc}"
            self.status_label.setText(message)
            return CaptureRecoveryResult(CaptureRecoveryStatus.FAILED, None, message)
        self.status_label.setText("同步恢复已请求，正在等待新的稳定画面")
        return CaptureRecoveryResult(
            CaptureRecoveryStatus.PENDING,
            recovered_board,
            "waiting for a fresh stable capture frame",
            recovery_id,
        )

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
            sync_mode = SyncMode(str(self.mode_combo.currentData()))
        except (GeometryError, TypeError, ValueError) as exc:
            self.status_label.setText(f"四角标定无效：{exc}")
            return
        evidence_mode = self.evidence_checkbox.isChecked()
        if evidence_mode and sync_mode is not SyncMode.HUMAN_VS_AI:
            self.status_label.setText("本地证据模式只用于人机练习，请先切换同步模式")
            return
        if evidence_mode and self._board_provider is None:
            self.status_label.setText("本地证据模式需要一个已确认棋盘局面")
            return

        self._generation += 1
        generation = self._generation

        def forward(update: CaptureMonitorUpdate | LiveSyncUpdate) -> None:
            self._bridge.update.emit(_TaggedCaptureUpdate(generation, update))

        if self._board_provider is None:
            source = self._source_factory(self._windows[index])
            self._monitor = CaptureMonitor(
                source,
                quad,
                orientation=orientation,
                on_update=forward,
            )
        else:
            try:
                board = replace(self._board_provider(), orientation=orientation)
                source = self._source_factory(self._windows[index])
            except (RuntimeError, TypeError, ValueError) as exc:
                self.status_label.setText(f"当前局面无效：{exc}")
                return
            occupancy_observer = self._occupancy_observer_factory(board)
            self._session = LiveSyncSession(
                source,
                board,
                quad,
                on_update=forward,
                sync_mode=sync_mode,
                patch_size=self._patch_size,
                capture_transition_evidence=evidence_mode,
                occupancy_observer=occupancy_observer,
                require_matching_baseline=True,
                require_atomic_two_ply=evidence_mode,
            )
            if evidence_mode:
                self._evidence_enabled_for_session = True
                self._evidence_board = board
                self._evidence_quad = quad
                self._evidence_client_size = self._windows[index].client_size
                self._evidence_session_id = self._session_id_factory()
        self.connect_button.setText("断开")
        self.window_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.quad_input.setEnabled(False)
        self.orientation_combo.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.evidence_checkbox.setEnabled(False)
        if self._session is not None:
            self._session.start()
        elif self._monitor is not None:
            self._monitor.start()

    def _show_update(self, tagged: _TaggedCaptureUpdate) -> None:
        if tagged.generation != self._generation:
            return
        update = tagged.update
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
        elif update.status is LiveSyncStatus.WAITING_FOR_REPLY:
            self.status_label.setText("第一步已稳定，正在等待人机应手…")
        elif update.status is LiveSyncStatus.MOVE_ACCEPTED:
            moves = " · ".join(move.uci for move in update.moves)
            count = "两步" if len(update.moves) == 2 else "一步"
            self.status_label.setText(f"已同步{count}：{moves or '未知'}")
        elif update.status is LiveSyncStatus.RECOVERY_PENDING:
            self.status_label.setText("同步恢复已请求，正在等待新的稳定画面")
        elif update.status is LiveSyncStatus.RECOVERY_ACCEPTED:
            self.status_label.setText("同步恢复成功，已继续监听")
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
        self._record_review_event(update)

    def _record_review_event(self, update: LiveSyncUpdate) -> None:
        if (
            not self._evidence_enabled_for_session
            or update.status
            not in (LiveSyncStatus.MOVE_ACCEPTED, LiveSyncStatus.PAUSED_AMBIGUOUS)
        ):
            return
        pending = self._pending_review_event
        if pending is not None:
            next_board = self._pending_review_next_board
            if next_board is not None and update.board.position_id != next_board.position_id:
                self._evidence_out_of_sync = True
            return
        board = self._evidence_board
        quad = self._evidence_quad
        client_size = self._evidence_client_size
        session_id = self._evidence_session_id
        if board is None or quad is None or client_size is None or session_id is None:
            self.status_label.setText("本地证据上下文已失效；本次事件未保存")
            return
        event_id = self._event_id_factory()
        try:
            event_dir = self._evidence_writer.record(
                update,
                board=board,
                quad=quad,
                session_id=session_id,
                event_id=event_id,
                client_size=client_size,
                generation_id=self._generation,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self._evidence_board = update.board
            self.status_label.setText(f"本地证据未保存：{exc}")
            return
        self._pending_review_event = event_dir
        self._pending_review_next_board = update.board
        self.status_label.setText("终局事件已隔离保存；请先完成本地复核")
        self.review_event_ready.emit(event_dir)

    def _clear_evidence_session(self) -> None:
        self._evidence_enabled_for_session = False
        self._evidence_board = None
        self._evidence_quad = None
        self._evidence_client_size = None
        self._evidence_session_id = None
        self._pending_review_event = None
        self._pending_review_next_board = None
        self._evidence_out_of_sync = False


def _default_source_factory(window: WindowInfo) -> FrameSource:
    return WindowsCaptureSource(window, fps=20)


def _window_label(window: WindowInfo) -> str:
    name = "天天象棋" if "天天象棋" in window.title else "候选窗口"
    width, height = window.client_size
    return f"{name} · {window.process_name} · {width}×{height}"
