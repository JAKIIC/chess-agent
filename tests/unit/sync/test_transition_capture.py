import numpy as np
import pytest

from xiangqi_agent.sync.transition_capture import (
    TransitionCaptureEvidence,
    TransitionPointEvidence,
    build_transition_capture_evidence,
)
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad
from xiangqi_agent.vision.occupancy import OccupancyEvidence


def test_transition_evidence_requires_both_occupancy_snapshots_or_neither() -> None:
    before = OccupancyEvidence((False,) * 90, (0.9,) * 90, "test-v1")
    crop = TransitionPointEvidence(
        point_index=0,
        before=np.zeros((48, 48, 4), dtype=np.uint8),
        after=np.ones((48, 48, 4), dtype=np.uint8),
    )

    without_occupancy = TransitionCaptureEvidence(
        changed_points=(0,),
        local_differences=(1.0,) + (0.0,) * 89,
        crops=(crop,),
        decision_latency_ms=2.5,
    )

    assert without_occupancy.before_occupancy is None
    assert without_occupancy.after_occupancy is None
    with pytest.raises(ValueError, match="both be present"):
        TransitionCaptureEvidence(
            changed_points=(0,),
            local_differences=(1.0,) + (0.0,) * 89,
            crops=(crop,),
            decision_latency_ms=2.5,
            before_occupancy=before,
        )


def test_transition_builder_observes_confirmed_and_settled_frames_separately() -> None:
    before = np.zeros((100, 100, 4), dtype=np.uint8)
    before[..., 3] = 255
    after = np.full((100, 100, 4), 255, dtype=np.uint8)
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        (100, 100),
    )
    observer = _FrameValueOccupancyObserver()

    evidence = build_transition_capture_evidence(
        before,
        after,
        geometry,
        (4.0,) + (0.0,) * 89,
        decision_latency_ms=7.0,
        occupancy_observer=observer,
    )

    assert evidence.before_occupancy == OccupancyEvidence(
        (False,) * 90,
        (0.95,) * 90,
        "frame-value-v1",
    )
    assert evidence.after_occupancy == OccupancyEvidence(
        (True,) * 90,
        (0.95,) * 90,
        "frame-value-v1",
    )
    assert observer.observed_values == [0, 255]


def test_transition_builder_reuses_confirmed_occupancy_and_updates_changed_points() -> None:
    before = np.zeros((100, 100, 4), dtype=np.uint8)
    before[..., 3] = 255
    after = np.full((100, 100, 4), 255, dtype=np.uint8)
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        (100, 100),
    )
    baseline = OccupancyEvidence((False,) * 90, (0.95,) * 90, "incremental-v1")
    observer = _IncrementalOccupancyObserver()

    evidence = build_transition_capture_evidence(
        before,
        after,
        geometry,
        (4.0,) + (0.0,) * 89,
        decision_latency_ms=7.0,
        occupancy_observer=observer,
        confirmed_occupancy=baseline,
        update_changed_occupancy_only=True,
        accepted_changed_points=(0,),
    )

    assert evidence.before_occupancy == baseline
    assert evidence.after_occupancy is not None
    assert evidence.after_occupancy.occupied[0]
    assert evidence.after_occupancy.occupied[1:] == (False,) * 89
    assert observer.full_calls == 0
    assert observer.changed_calls == [(0,)]


def test_accepted_transition_uses_semantic_points_over_stronger_artifacts() -> None:
    before = np.zeros((100, 100, 4), dtype=np.uint8)
    before[..., 3] = 255
    after = np.full((100, 100, 4), 255, dtype=np.uint8)
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        (100, 100),
    )
    baseline = OccupancyEvidence((False,) * 90, (0.95,) * 90, "incremental-v1")
    observer = _IncrementalOccupancyObserver()
    differences = [0.0] * 90
    for index, value in enumerate((100.0, 90.0, 80.0, 70.0, 60.0)):
        differences[index] = value
    differences[22] = 12.0
    differences[25] = 11.0
    differences[67] = 10.0
    differences[70] = 9.0

    evidence = build_transition_capture_evidence(
        before,
        after,
        geometry,
        tuple(differences),
        decision_latency_ms=7.0,
        occupancy_observer=observer,
        confirmed_occupancy=baseline,
        update_changed_occupancy_only=True,
        accepted_changed_points=(22, 25, 67, 70),
    )

    assert evidence.changed_points == (22, 25, 67, 70)
    assert observer.changed_calls == [(22, 25, 67, 70)]
    assert evidence.after_occupancy is not None
    assert tuple(
        index for index, occupied in enumerate(evidence.after_occupancy.occupied) if occupied
    ) == (22, 25, 67, 70)


def test_transition_builder_keeps_occupancy_disabled_by_default() -> None:
    before = np.zeros((100, 100, 4), dtype=np.uint8)
    after = before.copy()
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        (100, 100),
    )

    evidence = build_transition_capture_evidence(
        before,
        after,
        geometry,
        (1.0,) + (0.0,) * 89,
        decision_latency_ms=0.0,
    )

    assert evidence.before_occupancy is None
    assert evidence.after_occupancy is None


class _FrameValueOccupancyObserver:
    def __init__(self) -> None:
        self.observed_values: list[int] = []

    def observe(
        self,
        frame: np.ndarray,
        geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        assert geometry.frame_size == (100, 100)
        value = int(frame[0, 0, 0])
        self.observed_values.append(value)
        return OccupancyEvidence(
            (value > 0,) * 90,
            (0.95,) * 90,
            "frame-value-v1",
        )


class _IncrementalOccupancyObserver:
    def __init__(self) -> None:
        self.full_calls = 0
        self.changed_calls: list[tuple[int, ...]] = []

    def observe(
        self,
        _frame: np.ndarray,
        _geometry: BoardGeometry,
    ) -> OccupancyEvidence:
        self.full_calls += 1
        raise AssertionError("cached transition must not perform a full occupancy scan")

    def observe_changed(
        self,
        _frame: np.ndarray,
        _geometry: BoardGeometry,
        baseline: OccupancyEvidence,
        point_indices: tuple[int, ...],
    ) -> OccupancyEvidence:
        self.changed_calls.append(point_indices)
        occupied = list(baseline.occupied)
        for index in point_indices:
            occupied[index] = True
        return OccupancyEvidence(
            tuple(occupied),
            baseline.confidences,
            baseline.algorithm_version,
        )
