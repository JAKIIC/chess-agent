from __future__ import annotations

from pathlib import Path

from keyring.errors import KeyringError
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from xiangqi_agent.coach.client import DeepSeekClient
from xiangqi_agent.coach.evidence import build_evidence
from xiangqi_agent.coach.service import CoachExplainer, CoachService
from xiangqi_agent.config import SecretStore
from xiangqi_agent.diagnostics.stage_c_promotion import StageCPromotionService
from xiangqi_agent.diagnostics.stage_c_review import StageCReviewService, StageCReviewStore
from xiangqi_agent.domain.analysis import EngineAnalysis
from xiangqi_agent.domain.board import BoardState, Move
from xiangqi_agent.domain.coach import CoachEvidence, CoachExplanation
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import resolve_move_reference, to_chinese
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.engine.installer import load_installed_pikafish
from xiangqi_agent.engine.process import PikafishProcess
from xiangqi_agent.engine.service import AnalysisEngine, AnalysisFailure, AnalysisService
from xiangqi_agent.sync.committer import RuleStateCommitter
from xiangqi_agent.sync.live_session import LiveSyncStatus, LiveSyncUpdate
from xiangqi_agent.sync.mode import SyncMode
from xiangqi_agent.ui.analysis_view_model import analysis_rows
from xiangqi_agent.ui.board_widget import BoardWidget
from xiangqi_agent.ui.capture_panel import CapturePanel, CaptureRecoveryStatus
from xiangqi_agent.ui.coach_panel import CoachPanel
from xiangqi_agent.ui.fonts import ensure_cjk_font
from xiangqi_agent.ui.settings_dialog import DeepSeekSettingsDialog
from xiangqi_agent.ui.stage_c_review_panel import StageCReviewPanel

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class _AnalysisBridge(QObject):
    quick = Signal(object)
    deep = Signal(object)
    failed = Signal(object)
    coach_ready = Signal(object)
    coach_failed = Signal(str)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        engine: AnalysisEngine | None = None,
        coach_client: CoachExplainer | None = None,
        capture_panel: CapturePanel | None = None,
        review_panel: StageCReviewPanel | None = None,
        runtime_root: Path | None = None,
        quick_ms: int = 500,
        deep_ms: int = 3000,
        enable_live_analysis: bool = False,
    ) -> None:
        super().__init__()
        application = QApplication.instance()
        if isinstance(application, QApplication):
            application.setFont(QFont(ensure_cjk_font(), 10))
        self.setWindowTitle("天天象棋学习助手")
        self.resize(1180, 760)
        self._board: BoardState | None = None
        self._latest_analysis: EngineAnalysis | None = None
        self._analysis_generation: int | None = None
        self._active_evidence: CoachEvidence | None = None
        self._pending_manual_board: BoardState | None = None
        self._pending_recovery_id: int | None = None
        self._enable_live_analysis = enable_live_analysis
        self._secret_store = SecretStore()
        self._bridge = _AnalysisBridge(self)
        self._bridge.quick.connect(self._show_quick)
        self._bridge.deep.connect(self._show_deep)
        self._bridge.failed.connect(self._show_error)
        self._bridge.coach_ready.connect(self._show_coach)
        self._bridge.coach_failed.connect(self._show_coach_error)
        self._service: AnalysisService | None = None
        resolved_runtime_root = runtime_root or Path.cwd()

        self.board_widget = BoardWidget()
        self.capture_panel = capture_panel or CapturePanel(
            local_root=resolved_runtime_root / ".local"
        )
        self.review_panel = review_panel or StageCReviewPanel(
            review_service=StageCReviewService(
                StageCReviewStore(
                    resolved_runtime_root / ".local" / "stage-c-reviews",
                    enabled=True,
                )
            ),
            promotion_service=StageCPromotionService(),
            reviewed_root=resolved_runtime_root / ".local" / "stage-c-reviewed",
        )
        self.capture_panel.set_board_provider(self._capture_board)
        self.capture_panel.sync_update.connect(self._on_sync_update)
        self.capture_panel.session_reset.connect(self._on_capture_reset)
        self.capture_panel.review_event_ready.connect(self._load_review_event)
        self.review_panel.review_completed.connect(self.capture_panel.finish_review)
        self.fen_input = QLineEdit(START_FEN)
        self.fen_input.setPlaceholderText("输入标准中国象棋 FEN")
        self.analyse_button = QPushButton("载入局面并分析")
        self.analyse_button.clicked.connect(self._analyse_fen)
        self.phase_label = QLabel("等待载入局面")
        self.guidance_label = QLabel()
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setStyleSheet("color: #8a4b08;")
        self.coach_panel = CoachPanel()
        self.coach_panel.question_submitted.connect(self._ask_coach)
        self.user_side_combo = QComboBox()
        self.user_side_combo.addItem("我是红方", "w")
        self.user_side_combo.addItem("我是黑方", "b")
        self.settings_button = QPushButton("DeepSeek 设置")
        self.settings_button.clicked.connect(self._open_coach_settings)
        self.tabs = QTabWidget()
        self.results = QTableWidget(0, 5)
        self.results.setHorizontalHeaderLabels(
            ["推荐走法", "评分（红方）", "深度", "主要变化", "UCI"]
        )
        self.results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results.horizontalHeader().setStretchLastSection(False)
        self.results.horizontalHeader().setSectionResizeMode(3, self.results.horizontalHeader().ResizeMode.Stretch)

        self._board = parse_fen(START_FEN)
        self.board_widget.set_board(self._board)

        self._build_layout()
        resolved_coach = coach_client or self._load_default_coach()
        self._coach_service = CoachService(
            resolved_coach,
            on_ready=self._bridge.coach_ready.emit,
            on_error=self._bridge.coach_failed.emit,
        )
        resolved_engine = engine or self._load_default_engine(resolved_runtime_root)
        if resolved_engine is not None:
            self._service = AnalysisService(
                resolved_engine,
                quick_ms=quick_ms,
                deep_ms=deep_ms,
                multipv=3,
                on_quick=self._bridge.quick.emit,
                on_deep=self._bridge.deep.emit,
                on_error=self._bridge.failed.emit,
            )

    def _build_layout(self) -> None:
        panel = QWidget()
        root = QHBoxLayout(panel)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(20)
        root.addWidget(self.board_widget, 5)

        right = QVBoxLayout()
        title = QLabel("局面分析")
        title.setStyleSheet("font-size: 24px; font-weight: 600;")
        right.addWidget(title)
        boundary = QLabel("仅用于人机练习、残局训练和赛后复盘；不会点击或自动走棋。")
        boundary.setWordWrap(True)
        boundary.setStyleSheet("color: #555;")
        right.addWidget(boundary)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        right.addWidget(separator)
        right.addWidget(QLabel("可见窗口捕获与四角标定"))
        right.addWidget(self.capture_panel)
        right.addWidget(QLabel("当前局面 FEN"))
        right.addWidget(self.fen_input)
        right.addWidget(self.analyse_button)
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.addWidget(self.phase_label)
        analysis_layout.addWidget(self.guidance_label)
        analysis_layout.addWidget(self.results, 1)
        self.tabs.addTab(analysis_tab, "分析")

        coach_tab = QWidget()
        coach_layout = QVBoxLayout(coach_tab)
        coach_header = QHBoxLayout()
        coach_header.addWidget(self.user_side_combo)
        coach_header.addStretch(1)
        coach_header.addWidget(self.settings_button)
        coach_layout.addLayout(coach_header)
        coach_layout.addWidget(self.coach_panel, 1)
        self.tabs.addTab(coach_tab, "教练")
        self.tabs.addTab(self.review_panel, "复核")
        right.addWidget(self.tabs, 1)
        root.addLayout(right, 7)
        self.setCentralWidget(panel)

    def _load_default_coach(self) -> DeepSeekClient:
        try:
            api_key = self._secret_store.get_deepseek_key()
        except KeyringError:
            api_key = None
        return DeepSeekClient(api_key=api_key)

    def _open_coach_settings(self) -> None:
        dialog = DeepSeekSettingsDialog(self._secret_store, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._coach_service.close()
        self._coach_service = CoachService(
            self._load_default_coach(),
            on_ready=self._bridge.coach_ready.emit,
            on_error=self._bridge.coach_failed.emit,
        )
        self._active_evidence = None
        self.coach_panel.clear_explanation("DeepSeek 设置已更新；请重新提问")

    def _load_default_engine(self, runtime_root: Path) -> PikafishProcess | None:
        try:
            installed = load_installed_pikafish(runtime_root / ".local" / "pikafish")
        except RuntimeError:
            self.phase_label.setText("Pikafish 未安装，分析功能暂不可用")
            self.guidance_label.setText(
                "请先运行：python scripts/install_pikafish.py。棋盘和 FEN 功能不会因此崩溃。"
            )
            return None
        self.guidance_label.clear()
        return PikafishProcess(
            installed.executable,
            threads=2,
            hash_mb=256,
            eval_file=installed.eval_file,
        )

    def _analyse_fen(self) -> None:
        try:
            board = parse_fen(self.fen_input.text().strip())
            legal_moves(board)
        except ValueError as exc:
            self.phase_label.setText(f"FEN 无效：{exc}")
            return
        if (
            self.review_panel.confirmed_position_id is not None
            and self.review_panel.confirmed_position_id != board.position_id
        ):
            self.review_panel.invalidate("当前局面已经变化，旧复核卡已失效。")
        recovery = self.capture_panel.recover(board)
        if recovery.status is CaptureRecoveryStatus.FAILED:
            self._pending_manual_board = None
            self._pending_recovery_id = None
            self.phase_label.setText(recovery.message)
            return
        recovered_board = recovery.board or board
        if recovery.status is CaptureRecoveryStatus.PENDING:
            self._pending_manual_board = recovered_board
            self._pending_recovery_id = recovery.recovery_id
            self.phase_label.setText("等待新的稳定画面确认手工 FEN…")
            return
        self._pending_manual_board = None
        self._pending_recovery_id = None
        self._adopt_board(recovered_board, "正在进行快速分析…")

    def _adopt_board(
        self,
        board: BoardState,
        phase: str,
        *,
        last_move: Move | None = None,
        submit_analysis: bool = True,
    ) -> None:
        if (
            self.review_panel.confirmed_position_id is not None
            and self.review_panel.confirmed_position_id != board.position_id
        ):
            self.review_panel.invalidate("当前局面已经变化，旧复核卡已失效。")
        self._board = board
        self._latest_analysis = None
        self._analysis_generation = None
        self._active_evidence = None
        self._coach_service.clear()
        self.coach_panel.clear_explanation("等待当前局面的引擎证据")
        self.fen_input.setText(board.fen)
        self.board_widget.set_board(board, last_move=last_move)
        self.results.setRowCount(0)
        if self._service is None:
            self.phase_label.setText(f"{phase}；Pikafish 未安装")
            return
        self.phase_label.setText(phase)
        if submit_analysis:
            self._analysis_generation = self._service.submit(board)

    def _capture_board(self) -> BoardState:
        if self._board is None:
            raise RuntimeError("current board is not ready")
        return self._board

    def _on_sync_update(self, update: LiveSyncUpdate) -> None:
        if update.status in (LiveSyncStatus.ERROR, LiveSyncStatus.CLOSED):
            if self._pending_manual_board is not None:
                self._pending_manual_board = None
                self._pending_recovery_id = None
                self.phase_label.setText("同步恢复已中断；原局面保持不变")
            return
        if update.status is LiveSyncStatus.BASELINE_READY:
            current = self._board
            if current is not None and update.board.position_id == current.position_id:
                self._board = update.board
                self.fen_input.setText(update.board.fen)
                self.board_widget.set_board(update.board)
            return
        if update.status is LiveSyncStatus.RECOVERY_ACCEPTED:
            pending = self._pending_manual_board
            if pending is None or update.recovery_id != self._pending_recovery_id:
                return
            if (
                update.board.position_id != pending.position_id
                or update.board.orientation is not pending.orientation
            ):
                self._pending_manual_board = None
                self._pending_recovery_id = None
                self.phase_label.setText("同步恢复结果与待确认 FEN 不一致；已保留原局面")
                return
            self._pending_manual_board = None
            self._pending_recovery_id = None
            self._adopt_board(update.board, "手工 FEN 已确认，正在进行快速分析…")
            return
        if update.status is not LiveSyncStatus.MOVE_ACCEPTED or not update.moves:
            return
        if self._pending_manual_board is not None:
            return
        if len(update.moves) not in (1, 2):
            return
        if len(update.moves) == 2 and update.sync_mode is not SyncMode.HUMAN_VS_AI:
            return
        before = self._board
        if before is None or update.board.position_id == before.position_id:
            return
        if update.before_position_id != before.position_id:
            return
        if update.after_position_id != update.board.position_id:
            return
        projected = before
        notations: list[str] = []
        committer = RuleStateCommitter()
        try:
            for move in update.moves:
                notations.append(to_chinese(projected, move))
                projected = committer.project(projected, (move,))
        except ValueError:
            return
        if projected != update.board:
            return
        notation = " · ".join(notations)
        phase = f"已同步 {notation} · 正在进行快速分析…"
        if not self._enable_live_analysis:
            phase = f"已同步 {notation}；实时分析已锁定，等待 Stage C 盲测通过"
        self._adopt_board(
            update.board,
            phase,
            last_move=update.moves[-1],
            submit_analysis=self._enable_live_analysis,
        )

    def _on_capture_reset(self) -> None:
        if self.review_panel.card is not None:
            self.review_panel.invalidate("捕获会话已经变化，旧复核卡已失效。")
        if self._pending_manual_board is None:
            return
        self._pending_manual_board = None
        self._pending_recovery_id = None
        self.phase_label.setText("捕获会话已重置；原局面保持不变")

    def _load_review_event(self, event_dir: object) -> None:
        if not isinstance(event_dir, Path):
            self.review_panel.invalidate("本地复核事件无效。")
            return
        try:
            self.review_panel.load_event(event_dir)
        except (OSError, RuntimeError, TypeError, ValueError):
            self.review_panel.invalidate("本地复核证据无法安全读取。")
            return
        self.tabs.setCurrentWidget(self.review_panel)

    def _show_quick(self, analysis: EngineAnalysis) -> None:
        if not self._is_current(analysis):
            return
        self._render_analysis(analysis)
        self._latest_analysis = analysis
        self.coach_panel.clear_explanation("快速证据已就绪，可以开始提问")
        self.phase_label.setText(f"快速分析 · 深度 {analysis.depth} · 正在继续加深…")

    def _show_deep(self, analysis: EngineAnalysis) -> None:
        if not self._is_current(analysis):
            return
        self._render_analysis(analysis)
        self._latest_analysis = analysis
        self.coach_panel.clear_explanation("加深证据已就绪，可以开始提问")
        self.phase_label.setText(
            f"加深分析 · 深度 {analysis.depth} · {analysis.duration_ms} ms"
        )

    def _show_error(self, failure: AnalysisFailure) -> None:
        board = self._board
        if (
            board is None
            or failure.position_id != board.position_id
            or failure.generation != self._analysis_generation
        ):
            return
        self.phase_label.setText(f"分析暂停：{failure.message}")

    def _ask_coach(self, question: str) -> None:
        board = self._board
        analysis = self._latest_analysis
        if board is None or analysis is None:
            self.coach_panel.clear_explanation("请先完成当前局面的引擎分析")
            return
        user_side = self.user_side_combo.currentData()
        if user_side not in ("w", "b"):
            self.coach_panel.clear_explanation("请选择执红或执黑")
            return
        evidence = build_evidence(board, analysis, user_side=user_side)
        self._active_evidence = evidence
        referenced = resolve_move_reference(board, question)
        message = "正在生成证据约束的解释…"
        if referenced is not None and referenced.uci not in {
            candidate.uci for candidate in evidence.candidates
        }:
            message += "（已识别你的走法，但尚未进行分支评分）"
        self.coach_panel.clear_explanation(message)
        self._coach_service.submit(evidence, question)

    def _show_coach(self, explanation: CoachExplanation) -> None:
        evidence = self._active_evidence
        if evidence is None or explanation.position_id != evidence.position_id:
            return
        self.coach_panel.set_explanation(explanation, evidence.allowed_move_map)

    def _show_coach_error(self, message: str) -> None:
        self.coach_panel.clear_explanation(f"教练暂停：{message}")

    def _is_current(self, analysis: EngineAnalysis) -> bool:
        return self._board is not None and analysis.position_id == self._board.position_id

    def _render_analysis(self, analysis: EngineAnalysis) -> None:
        board = self._board
        if board is None:
            return
        rows = analysis_rows(board, analysis)
        self.results.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (row.notation, row.score, str(row.depth), row.variation, row.uci)
            for column, value in enumerate(values):
                self.results.setItem(index, column, QTableWidgetItem(value))
        self.results.resizeColumnToContents(0)
        self.results.resizeColumnToContents(1)
        self.results.resizeColumnToContents(2)
        self.results.resizeColumnToContents(4)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.review_panel.card is not None:
            self.review_panel.invalidate("学习助手已关闭，待复核卡已失效。")
        self.capture_panel.close_capture()
        self._coach_service.close()
        if self._service is not None:
            self._service.close()
        super().closeEvent(event)
