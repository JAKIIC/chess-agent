import numpy as np

from xiangqi_agent.vision.change_detection import FrameStabilityDetector, analyze_frame_change
from xiangqi_agent.vision.geometry import BoardGeometry, NormalizedQuad


def _geometry() -> BoardGeometry:
    return BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(((4, 4), (68, 4), (68, 76), (4, 76)), (73, 81)),
        (73, 81),
    )


def _frame() -> np.ndarray:
    frame = np.zeros((81, 73, 4), dtype=np.uint8)
    frame[..., 3] = 255
    return frame


def test_change_analysis_finds_the_most_changed_intersection() -> None:
    before = _frame()
    after = before.copy()
    x, y = _geometry().grid_points()[10]
    after[round(y) - 2 : round(y) + 3, round(x) - 2 : round(x) + 3, :3] = 200

    change = analyze_frame_change(before, after, _geometry(), top_k=4)

    assert change.global_difference > 0
    assert len(change.local_differences) == 90
    assert change.most_changed_indices[0] == 10
    assert change.stable is False


def test_stability_requires_consecutive_stable_pairs_and_resets_on_change() -> None:
    detector = FrameStabilityDetector(
        _geometry(),
        required_stable_pairs=2,
        global_threshold=0.5,
        local_threshold=0.5,
    )
    stable = _frame()

    assert detector.update(stable) is None
    assert detector.update(stable.copy()).stable is False
    assert detector.update(stable.copy()).stable is True

    changed = stable.copy()
    x, y = _geometry().grid_points()[42]
    changed[round(y) - 2 : round(y) + 3, round(x) - 2 : round(x) + 3, :3] = 255
    result = detector.update(changed)

    assert result.stable is False
    assert result.most_changed_indices[0] == 42
