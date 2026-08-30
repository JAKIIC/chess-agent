from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from xiangqi_agent.domain.coach import CoachExplanation


class CoachPanel(QWidget):
    question_submitted = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._explanation: CoachExplanation | None = None
        self._allowed_move_map: dict[str, str] = {}
        self._level = 0

        self.source_label = QLabel("等待局面分析")
        self.summary_label = QLabel()
        self.plan_label = QLabel()
        self.candidate_label = QLabel()
        self.variation_label = QLabel()
        for label in (
            self.summary_label,
            self.plan_label,
            self.candidate_label,
            self.variation_label,
        ):
            label.setWordWrap(True)
            label.hide()

        self.next_hint_button = QPushButton("再揭示一层提示")
        self.next_hint_button.setEnabled(False)
        self.next_hint_button.clicked.connect(self._reveal_next)
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("例如：为什么推荐这一步？")
        self.ask_button = QPushButton("问教练")
        self.ask_button.clicked.connect(self._submit_question)
        self.question_input.returnPressed.connect(self._submit_question)

        root = QVBoxLayout(self)
        root.addWidget(self.source_label)
        root.addWidget(self.summary_label)
        root.addWidget(self.plan_label)
        root.addWidget(self.candidate_label)
        root.addWidget(self.variation_label)
        root.addWidget(self.next_hint_button)
        root.addStretch(1)
        question_row = QHBoxLayout()
        question_row.addWidget(self.question_input, 1)
        question_row.addWidget(self.ask_button)
        root.addLayout(question_row)

    def set_explanation(
        self, explanation: CoachExplanation, allowed_move_map: dict[str, str]
    ) -> None:
        if explanation.candidate_id not in allowed_move_map:
            raise ValueError("coach explanation candidate is not in the allowed move map")
        self._explanation = explanation
        self._allowed_move_map = dict(allowed_move_map)
        self.summary_label.setText(f"形势：{explanation.position_summary}")
        self.plan_label.setText(f"计划：{explanation.main_plan}")
        notation = self._allowed_move_map[explanation.candidate_id]
        self.candidate_label.setText(
            f"推荐：{notation}\n理由：{explanation.why}\n置信度：{explanation.confidence:.0%}"
        )
        alternative_names = [
            self._allowed_move_map[candidate]
            for candidate in explanation.alternatives
            if candidate in self._allowed_move_map
        ]
        alternatives = "、".join(alternative_names) if alternative_names else "无"
        self.variation_label.setText(
            f"对手威胁：{explanation.opponent_threat}\n"
            f"其他候选：{alternatives}\n训练问题：{explanation.training_question}"
        )
        self.source_label.setText(
            "来源：DeepSeek（已通过证据校验）"
            if explanation.source == "deepseek"
            else "来源：本地证据模板（未调用 DeepSeek）"
        )
        self.reveal_level(1)

    def reveal_level(self, level: int) -> None:
        if level not in (1, 2, 3, 4):
            raise ValueError("hint level must be between 1 and 4")
        self._level = level
        self.summary_label.setVisible(level >= 1)
        self.plan_label.setVisible(level >= 2)
        self.candidate_label.setVisible(level >= 3)
        self.variation_label.setVisible(level >= 4)
        self.next_hint_button.setEnabled(level < 4)

    def visible_sections(self) -> tuple[str, ...]:
        names = ("position_summary", "main_plan", "candidate", "variation")
        return names[: self._level]

    def clear_explanation(self, message: str = "等待局面分析") -> None:
        self._explanation = None
        self._allowed_move_map = {}
        self._level = 0
        self.source_label.setText(message)
        for label in (
            self.summary_label,
            self.plan_label,
            self.candidate_label,
            self.variation_label,
        ):
            label.clear()
            label.hide()
        self.next_hint_button.setEnabled(False)

    def _reveal_next(self) -> None:
        if self._explanation is not None and self._level < 4:
            self.reveal_level(self._level + 1)

    def _submit_question(self) -> None:
        question = self.question_input.text().strip()
        if not question:
            return
        self.question_input.clear()
        self.question_submitted.emit(question)
