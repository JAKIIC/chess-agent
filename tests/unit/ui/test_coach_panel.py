from __future__ import annotations

from PySide6.QtCore import Qt

from xiangqi_agent.domain.coach import CoachExplanation
from xiangqi_agent.ui.coach_panel import CoachPanel


def _explanation() -> CoachExplanation:
    return CoachExplanation(
        position_id="position-1",
        position_summary="双方子力完整。",
        main_plan="争夺中路。",
        candidate_id="candidate_1",
        why="第一候选能形成压力。",
        opponent_threat="黑方可能发展马。",
        alternatives=("candidate_2",),
        training_question="你看到了哪条线路？",
        confidence=0.9,
        source="local_fallback",
    )


def test_hints_reveal_in_learning_order(qtbot: object) -> None:
    panel = CoachPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.set_explanation(
        _explanation(), {"candidate_1": "炮二平五", "candidate_2": "马八进七"}
    )

    assert panel.visible_sections() == ("position_summary",)
    panel.reveal_level(2)
    assert panel.visible_sections() == ("position_summary", "main_plan")
    panel.reveal_level(3)
    assert "candidate" in panel.visible_sections()
    assert "炮二平五" in panel.candidate_label.text()
    panel.reveal_level(4)
    assert panel.visible_sections() == (
        "position_summary",
        "main_plan",
        "candidate",
        "variation",
    )
    assert "本地" in panel.source_label.text()


def test_question_signal_ignores_blank_input(qtbot: object) -> None:
    panel = CoachPanel()
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    submitted: list[str] = []
    panel.question_submitted.connect(submitted.append)

    panel.question_input.setText("  ")
    qtbot.mouseClick(panel.ask_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]
    panel.question_input.setText("为什么走这一步？")
    qtbot.mouseClick(panel.ask_button, Qt.MouseButton.LeftButton)  # type: ignore[attr-defined]

    assert submitted == ["为什么走这一步？"]
