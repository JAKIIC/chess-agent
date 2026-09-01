from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xiangqi_agent.diagnostics.stage_c_promotion import (
    PromotionStatus,
    StageCPromotionBlockedError,
)
from xiangqi_agent.diagnostics.stage_c_quarantine import QuarantineEventLoader
from xiangqi_agent.diagnostics.stage_c_review import (
    ReviewMoveChoice,
    StageCLabelKind,
    StageCReviewDraft,
    StageCReviewOutcome,
    StageCReviewService,
    legal_review_choices,
    project_review_prefix,
)
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario
from xiangqi_agent.domain.board import BoardState
from xiangqi_agent.domain.fen import parse_fen
from xiangqi_agent.domain.notation import to_chinese
from xiangqi_agent.domain.rules import apply_move, legal_moves
from xiangqi_agent.ui.board_widget import BoardWidget


class PromotionService(Protocol):
    def promote(
        self,
        event_dir: Path,
        review_path: Path,
        reviewed_root: Path,
    ) -> Path: ...


@dataclass(frozen=True, slots=True)
class ReviewMoveLine:
    uci: str
    chinese: str


@dataclass(frozen=True, slots=True)
class StageCReviewCard:
    event_id: str
    session_id: str
    confirmed_position_id: str
    candidate_lines: tuple[ReviewMoveLine, ...]


_SCENARIO_LABELS = (
    (StageCScenario.MULTIPLE_CANDIDATES, "多个候选，无法唯一判断"),
    (StageCScenario.SELECTION_HIGHLIGHT, "点选或悬停高亮"),
    (StageCScenario.CONTINUOUS_ANIMATION, "连续落子动画"),
    (StageCScenario.OCCLUSION, "棋盘被遮挡"),
    (StageCScenario.RESIZE, "窗口尺寸变化"),
    (StageCScenario.THREE_PLY, "三步画面合并"),
)
_NEEDS_REVIEW_MESSAGES = {
    "baseline_confidence": "起始棋盘的占位证据不足",
    "changed_point_confidence": "变化交点的占位证据不足",
}


class StageCReviewPanel(QWidget):
    review_completed = Signal(str, str)

    def __init__(
        self,
        *,
        review_service: StageCReviewService,
        promotion_service: PromotionService,
        reviewed_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(review_service, StageCReviewService):
            raise TypeError("review_service must be a StageCReviewService")
        if not isinstance(reviewed_root, Path):
            raise TypeError("reviewed_root must be a Path")
        self._review_service = review_service
        self._promotion_service = promotion_service
        self._reviewed_root = reviewed_root
        self._event_dir: Path | None = None
        self._board: BoardState | None = None
        self._candidate_moves: tuple[tuple[str, ...], ...] = ()
        self._mode: str | None = None
        self._submitted = False
        self.card: StageCReviewCard | None = None
        self.legal_first_moves: tuple[ReviewMoveChoice, ...] = ()
        self.legal_second_moves: tuple[ReviewMoveChoice, ...] = ()

        self.title_label = QLabel("等待本地复核事件")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.start_board = BoardWidget()
        self.start_board.setMinimumHeight(260)
        self.candidate_label = QLabel("尚无待复核走法")
        self.candidate_label.setWordWrap(True)
        self.status_label = QLabel("开启本地证据模式后，终局事件会在这里等待确认。")
        self.status_label.setWordWrap(True)

        self.first_move_combo = QComboBox()
        self.second_move_combo = QComboBox()
        self.first_move_combo.currentIndexChanged.connect(self._populate_second_moves)
        self.first_move_combo.hide()
        self.second_move_combo.hide()

        self.scenario_combo = QComboBox()
        for scenario, label in _SCENARIO_LABELS:
            self.scenario_combo.addItem(label, scenario.value)
        self.scenario_combo.hide()

        self.confirm_button = QPushButton("走法正确")
        self.correction_button = QPushButton("实际走法不同")
        self.rejection_button = QPushButton("这是干扰画面")
        self.discard_button = QPushButton("无法确定，丢弃")
        self.confirm_button.clicked.connect(self._confirm_candidate)
        self.correction_button.clicked.connect(self._choose_correction)
        self.rejection_button.clicked.connect(self._choose_rejection)
        self.discard_button.clicked.connect(self._discard)

        selectors = QHBoxLayout()
        selectors.addWidget(self.first_move_combo)
        selectors.addWidget(self.second_move_combo)
        selectors.addWidget(self.scenario_combo)
        actions = QHBoxLayout()
        actions.addWidget(self.confirm_button)
        actions.addWidget(self.correction_button)
        actions.addWidget(self.rejection_button)
        actions.addWidget(self.discard_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.title_label)
        layout.addWidget(self.start_board, 1)
        layout.addWidget(self.candidate_label)
        layout.addLayout(selectors)
        layout.addLayout(actions)
        layout.addWidget(self.status_label)
        self._set_actions_enabled(False)

    @property
    def confirmed_position_id(self) -> str | None:
        return None if self.card is None else self.card.confirmed_position_id

    def load_event(self, event_dir: Path) -> None:
        loaded = QuarantineEventLoader().load(event_dir)
        metadata = loaded.metadata
        board = replace(
            parse_fen(metadata.confirmed_fen),
            orientation=metadata.orientation,
        )
        candidate_moves = tuple(
            candidate.moves_uci for candidate in metadata.candidates[:2]
        )
        candidate_lines = tuple(
            _move_line(board, moves) for moves in candidate_moves
        )
        self._event_dir = event_dir
        self._board = board
        self._candidate_moves = candidate_moves
        self._mode = None
        self._submitted = False
        self.card = StageCReviewCard(
            metadata.event_id,
            metadata.session_id,
            metadata.confirmed_position_id,
            candidate_lines,
        )
        self.start_board.set_board(board)
        self.title_label.setText("请确认刚才实际发生的走法")
        if candidate_lines:
            self.candidate_label.setText(
                "程序预填：" + "\n".join(line.chinese for line in candidate_lines)
            )
        else:
            self.candidate_label.setText("程序没有给出可直接确认的唯一走法")
        self.status_label.setText("候选仅用于减少输入；最终标签由你确认。")
        self.first_move_combo.hide()
        self.second_move_combo.hide()
        self.scenario_combo.hide()
        self._set_actions_enabled(True)
        self.confirm_button.setEnabled(bool(candidate_moves))

    def invalidate(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("invalidation reason must be non-empty")
        self._event_dir = None
        self._board = None
        self._candidate_moves = ()
        self._mode = None
        self._submitted = True
        self.card = None
        self.first_move_combo.hide()
        self.second_move_combo.hide()
        self.scenario_combo.hide()
        self._set_actions_enabled(False)
        self.status_label.setText(reason)

    def _confirm_candidate(self) -> None:
        if not self._candidate_moves:
            self.status_label.setText("当前事件没有可直接确认的候选走法。")
            return
        self._submit(
            StageCReviewDraft(
                label_kind=StageCLabelKind.VALID_TWO_PLY,
                moves_uci=self._candidate_moves[0],
                scenario=None,
                review_outcome=StageCReviewOutcome.CANDIDATE_CONFIRMED,
            )
        )

    def _choose_correction(self) -> None:
        if self._mode != "correction":
            self._mode = "correction"
            self.scenario_combo.hide()
            self._populate_first_moves()
            self.first_move_combo.show()
            self.second_move_combo.show()
            self.status_label.setText("请选择实际第一步和随后发生的第二步，再次点击提交。")
            return
        first = self.first_move_combo.currentData()
        second = self.second_move_combo.currentData()
        if not isinstance(first, str) or not isinstance(second, str):
            self.status_label.setText("请选择两步连续合法走法。")
            return
        self._submit(
            StageCReviewDraft(
                label_kind=StageCLabelKind.VALID_TWO_PLY,
                moves_uci=(first, second),
                scenario=None,
                review_outcome=StageCReviewOutcome.LEGAL_MOVE_CORRECTION,
            )
        )

    def _choose_rejection(self) -> None:
        if self._mode != "rejection":
            self._mode = "rejection"
            self.first_move_combo.hide()
            self.second_move_combo.hide()
            self.scenario_combo.show()
            self.status_label.setText("请选择刚才出现的干扰类型，再次点击提交。")
            return
        value = self.scenario_combo.currentData()
        if not isinstance(value, str):
            self.status_label.setText("请选择干扰类型。")
            return
        self._submit(
            StageCReviewDraft(
                label_kind=StageCLabelKind.EXPECTED_REJECTION,
                moves_uci=(),
                scenario=StageCScenario(value),
                review_outcome=StageCReviewOutcome.EXPECTED_REJECTION,
            )
        )

    def _discard(self) -> None:
        self._submit(
            StageCReviewDraft(
                label_kind=StageCLabelKind.DISCARD,
                moves_uci=(),
                scenario=None,
                review_outcome=StageCReviewOutcome.DISCARDED,
            )
        )

    def _submit(self, draft: StageCReviewDraft) -> None:
        event_dir = self._event_dir
        card = self.card
        if event_dir is None or card is None or self._submitted:
            return
        try:
            review_path = self._review_service.submit(event_dir, draft)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(f"复核未保存：{exc}")
            return
        self._submitted = True
        self._set_actions_enabled(False)
        if draft.label_kind is StageCLabelKind.DISCARD:
            self.status_label.setText("该事件已丢弃，不会进入待冻结样本。")
            self.review_completed.emit(card.event_id, "discarded")
            return
        try:
            self._promotion_service.promote(
                event_dir,
                review_path,
                self._reviewed_root,
            )
        except StageCPromotionBlockedError as exc:
            if exc.decision.status is PromotionStatus.NEEDS_REVIEW:
                details = "；".join(
                    _NEEDS_REVIEW_MESSAGES.get(reason, "证据仍需人工核对")
                    for reason in exc.decision.reason_codes
                )
                self.status_label.setText(f"证据不足，暂未晋级：{details}")
            else:
                self.status_label.setText("证据与标签不一致，事件仍保留在隔离区。")
            return
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            self.status_label.setText(f"晋级失败，事件仍保留在隔离区：{exc}")
            return
        self.status_label.setText("复核完成，事件已进入待冻结样本。")
        self.review_completed.emit(card.event_id, "promoted")

    def _populate_first_moves(self) -> None:
        board = self._board
        self.first_move_combo.clear()
        self.legal_first_moves = () if board is None else legal_review_choices(board)
        for choice in self.legal_first_moves:
            self.first_move_combo.addItem(
                f"{choice.chinese}（{choice.uci}）",
                choice.uci,
            )
        self._populate_second_moves()

    def _populate_second_moves(self) -> None:
        board = self._board
        first = self.first_move_combo.currentData()
        self.second_move_combo.clear()
        if board is None or not isinstance(first, str):
            self.legal_second_moves = ()
            return
        try:
            middle = project_review_prefix(board, (first,))
        except ValueError:
            self.legal_second_moves = ()
            return
        self.legal_second_moves = legal_review_choices(middle)
        for choice in self.legal_second_moves:
            self.second_move_combo.addItem(
                f"{choice.chinese}（{choice.uci}）",
                choice.uci,
            )

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in (
            self.confirm_button,
            self.correction_button,
            self.rejection_button,
            self.discard_button,
        ):
            button.setEnabled(enabled)


def _move_line(board: BoardState, moves_uci: tuple[str, ...]) -> ReviewMoveLine:
    projected = board
    chinese: list[str] = []
    for uci in moves_uci:
        move = next((candidate for candidate in legal_moves(projected) if candidate.uci == uci), None)
        if move is None:
            raise ValueError("review candidate is not a legal sequential move")
        chinese.append(to_chinese(projected, move))
        projected = apply_move(projected, move)
    return ReviewMoveLine(" ".join(moves_uci), " · ".join(chinese))
