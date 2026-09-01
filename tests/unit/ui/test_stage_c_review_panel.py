from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from PySide6.QtWidgets import QLabel, QPushButton

from tests.unit.diagnostics.test_stage_c_promotion import _record
from tests.unit.diagnostics.test_stage_c_quarantine import START, _event
from xiangqi_agent.diagnostics.stage_c_promotion import (
    PromotionDecision,
    PromotionStatus,
    StageCPromotionBlockedError,
)
from xiangqi_agent.diagnostics.stage_c_review import (
    StageCLabelKind,
    StageCReviewOutcome,
    StageCReviewService,
    StageCReviewStore,
    load_stage_c_review,
)
from xiangqi_agent.diagnostics.stage_c_samples import StageCScenario
from xiangqi_agent.ui.stage_c_review_panel import StageCReviewPanel


class RecordingPromotionService:
    def __init__(self, error: StageCPromotionBlockedError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[Path, Path, Path]] = []

    def promote(self, event_dir: Path, review_path: Path, reviewed_root: Path) -> Path:
        self.calls.append((event_dir, review_path, reviewed_root))
        if self.error is not None:
            raise self.error
        return reviewed_root / event_dir.parent.name / event_dir.name


def _panel(
    tmp_path: Path,
    promotion: RecordingPromotionService,
) -> StageCReviewPanel:
    reviews = StageCReviewStore(
        tmp_path / ".local" / "stage-c-reviews",
        enabled=True,
    )
    service = StageCReviewService(
        reviews,
        now_utc=lambda: datetime(2026, 9, 1, 1, tzinfo=UTC),
        review_id_factory=lambda: "review-ui-1",
    )
    return StageCReviewPanel(
        review_service=service,
        promotion_service=promotion,
        reviewed_root=tmp_path / ".local" / "stage-c-reviewed",
    )


def test_review_card_shows_only_privacy_safe_chinese_evidence_and_four_actions(
    qtbot: object,
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    panel = _panel(tmp_path, RecordingPromotionService())
    qtbot.addWidget(panel)  # type: ignore[attr-defined]

    panel.load_event(event_dir)

    assert panel.start_board.board == START
    assert panel.card is not None
    assert len(panel.card.candidate_lines) == 1
    assert "炮二平五" in panel.candidate_label.text()
    assert "h2e2" not in panel.candidate_label.text()
    button_texts = {
        button.text() for button in panel.findChildren(QPushButton)
    }
    assert button_texts == {
        "走法正确",
        "实际走法不同",
        "这是干扰画面",
        "无法确定，丢弃",
    }
    visible_text = " ".join(label.text() for label in panel.findChildren(QLabel))
    for forbidden in ("分数", "阈值", "置信度", "FEN", START.fen):
        assert forbidden not in visible_text

    panel.rejection_button.click()
    assert panel.scenario_combo.isVisibleTo(panel)
    assert {
        panel.scenario_combo.itemData(index)
        for index in range(panel.scenario_combo.count())
    } == {
        StageCScenario.MULTIPLE_CANDIDATES.value,
        StageCScenario.SELECTION_HIGHLIGHT.value,
        StageCScenario.CONTINUOUS_ANIMATION.value,
        StageCScenario.OCCLUSION.value,
        StageCScenario.RESIZE.value,
        StageCScenario.THREE_PLY.value,
    }


def test_correction_lists_only_sequential_legal_moves_and_promotes_once(
    qtbot: object,
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    promotion = RecordingPromotionService()
    panel = _panel(tmp_path, promotion)
    completed: list[tuple[str, str]] = []
    panel.review_completed.connect(lambda event_id, status: completed.append((event_id, status)))
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.load_event(event_dir)

    panel.correction_button.click()
    board = START
    assert {
        panel.first_move_combo.itemData(index)
        for index in range(panel.first_move_combo.count())
    } == {move.uci for move in panel.legal_first_moves}
    first_index = panel.first_move_combo.findData("b2b3")
    assert first_index >= 0
    panel.first_move_combo.setCurrentIndex(first_index)
    assert {
        panel.second_move_combo.itemData(index)
        for index in range(panel.second_move_combo.count())
    } == {move.uci for move in panel.legal_second_moves}
    assert panel.second_move_combo.findData("b7b6") >= 0
    panel.second_move_combo.setCurrentIndex(panel.second_move_combo.findData("b7b6"))

    panel.correction_button.click()
    panel.correction_button.click()

    assert len(promotion.calls) == 1
    review = load_stage_c_review(promotion.calls[0][1])
    assert review.moves_uci == ("b2b3", "b7b6")
    assert review.review_outcome is StageCReviewOutcome.LEGAL_MOVE_CORRECTION
    assert completed == [("event-1", "promoted")]
    assert board.position_id == panel.card.confirmed_position_id


def test_needs_review_and_stale_cards_fail_closed(
    qtbot: object,
    tmp_path: Path,
) -> None:
    blocked = StageCPromotionBlockedError(
        PromotionDecision(
            PromotionStatus.NEEDS_REVIEW,
            ("changed_point_confidence",),
            None,
        )
    )
    promotion = RecordingPromotionService(blocked)
    event_dir = _record(tmp_path, _event())
    panel = _panel(tmp_path, promotion)
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.load_event(event_dir)

    panel.confirm_button.click()

    assert len(promotion.calls) == 1
    assert "证据不足" in panel.status_label.text()
    assert event_dir.exists()
    assert not panel.confirm_button.isEnabled()

    second_root = tmp_path / "stale"
    second_event = _record(second_root, _event())
    stale_promotion = RecordingPromotionService()
    stale = _panel(second_root, stale_promotion)
    qtbot.addWidget(stale)  # type: ignore[attr-defined]
    stale.load_event(second_event)
    stale.invalidate("捕获会话已经变化")
    stale.confirm_button.click()

    assert stale_promotion.calls == []
    assert "捕获会话已经变化" in stale.status_label.text()


def test_discard_creates_one_immutable_review_without_promotion(
    qtbot: object,
    tmp_path: Path,
) -> None:
    event_dir = _record(tmp_path, _event())
    promotion = RecordingPromotionService()
    panel = _panel(tmp_path, promotion)
    completed: list[tuple[str, str]] = []
    panel.review_completed.connect(lambda event_id, status: completed.append((event_id, status)))
    qtbot.addWidget(panel)  # type: ignore[attr-defined]
    panel.load_event(event_dir)

    panel.discard_button.click()
    panel.discard_button.click()

    review_root = tmp_path / ".local" / "stage-c-reviews" / "session-1" / "event-1"
    review = load_stage_c_review(next(review_root.glob("*.json")))
    assert review.label_kind is StageCLabelKind.DISCARD
    assert promotion.calls == []
    assert completed == [("event-1", "discarded")]
