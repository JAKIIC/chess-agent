from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from xiangqi_agent.domain.analysis import EngineAnalysis
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.rules import legal_moves
from xiangqi_agent.engine.installer import load_installed_pikafish
from xiangqi_agent.engine.process import PikafishProcess
from xiangqi_agent.engine.service import AnalysisEngine, AnalysisService
from xiangqi_agent.ui.analysis_view_model import analysis_rows
from xiangqi_agent.ui.board_widget import BoardWidget
from xiangqi_agent.ui.fonts import ensure_cjk_font

START_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


class _AnalysisBridge(QObject):
    quick = Signal(object)
    deep = Signal(object)
    failed = Signal(str)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        engine: AnalysisEngine | None = None,
        runtime_root: Path | None = None,
        quick_ms: int = 500,
        deep_ms: int = 3000,
    ) -> None:
        super().__init__()
        application = QApplication.instance()
        if isinstance(application, QApplication):
            application.setFont(QFont(ensure_cjk_font(), 10))
        self.setWindowTitle("天天象棋学习助手")
        self.resize(1180, 760)
        self._board: BoardState | None = None
        self._bridge = _AnalysisBridge(self)
        self._bridge.quick.connect(self._show_quick)
        self._bridge.deep.connect(self._show_deep)
        self._bridge.failed.connect(self._show_error)
        self._service: AnalysisService | None = None

        self.board_widget = BoardWidget()
        self.fen_input = QLineEdit(START_FEN)
        self.fen_input.setPlaceholderText("输入标准中国象棋 FEN")
        self.analyse_button = QPushButton("载入局面并分析")
        self.analyse_button.clicked.connect(self._analyse_fen)
        self.phase_label = QLabel("等待载入局面")
        self.guidance_label = QLabel()
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setStyleSheet("color: #8a4b08;")
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
        resolved_engine = engine or self._load_default_engine(runtime_root or Path.cwd())
        if resolved_engine is None:
            self.analyse_button.setEnabled(False)
        else:
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
        right.addWidget(QLabel("当前局面 FEN"))
        right.addWidget(self.fen_input)
        right.addWidget(self.analyse_button)
        right.addWidget(self.phase_label)
        right.addWidget(self.guidance_label)
        right.addWidget(self.results, 1)
        root.addLayout(right, 7)
        self.setCentralWidget(panel)

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
        if self._service is None:
            return
        try:
            board = parse_fen(self.fen_input.text().strip())
            legal_moves(board)
        except ValueError as exc:
            self.phase_label.setText(f"FEN 无效：{exc}")
            return
        self._board = board
        self.board_widget.set_board(board)
        self.results.setRowCount(0)
        self.phase_label.setText("正在进行快速分析…")
        self._service.submit(board)

    def _show_quick(self, analysis: EngineAnalysis) -> None:
        if not self._is_current(analysis):
            return
        self._render_analysis(analysis)
        self.phase_label.setText(f"快速分析 · 深度 {analysis.depth} · 正在继续加深…")

    def _show_deep(self, analysis: EngineAnalysis) -> None:
        if not self._is_current(analysis):
            return
        self._render_analysis(analysis)
        self.phase_label.setText(
            f"加深分析 · 深度 {analysis.depth} · {analysis.duration_ms} ms"
        )

    def _show_error(self, message: str) -> None:
        self.phase_label.setText(f"分析暂停：{message}")

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
        if self._service is not None:
            self._service.close()
        super().closeEvent(event)
