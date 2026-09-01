import numpy as np
import pytest

from xiangqi_agent.capture.adaptive_sampling import (
    AdaptiveBurstSampler,
    FrameSizeChangedError,
)
from xiangqi_agent.capture.protocol import CaptureFrame


def _frame(
    timestamp_ns: int,
    value: int = 0,
    *,
    shape: tuple[int, int] = (2, 2),
) -> CaptureFrame:
    pixels = np.full((*shape, 4), value, dtype=np.uint8)
    pixels[..., 3] = 255
    return CaptureFrame(timestamp_ns, 1, pixels)


def test_quiet_gap_promotes_a_buffered_steady_frame_before_a_new_change() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    baseline = _frame(0)
    first_change = _frame(50_000_000, 10)
    next_change = _frame(170_000_000, 20)
    sampler.initialize(baseline)

    assert sampler.on_frame(first_change) == ()

    assert sampler.on_frame(next_change) == (
        first_change,
        first_change,
        first_change,
        next_change,
    )


def test_quiet_clock_emits_one_complete_endpoint_sequence_for_latest_frame() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    changed = _frame(50_000_000, 10)
    sampler.initialize(_frame(0))
    sampler.set_bursting(True)
    sampler.on_frame(changed)

    assert sampler.on_clock(149_999_999) == ()
    assert sampler.on_clock(150_000_000) == (changed, changed, changed)
    assert sampler.on_clock(300_000_000) == ()


def test_burst_does_not_publish_a_mid_animation_plateau_as_the_endpoint() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=400,
        stable_repeats=2,
    )
    baseline = _frame(0)
    animation_start = _frame(50_000_000, 40)
    mid_animation = _frame(100_000_000, 80)
    resumed_animation = _frame(460_000_000, 120)
    final_position = _frame(600_000_000, 160)
    sampler.initialize(baseline)

    assert sampler.on_frame(animation_start) == (animation_start,)
    sampler.set_bursting(True)

    assert sampler.on_frame(mid_animation) == ()
    assert sampler.on_frame(_frame(200_000_000, 80)) == ()
    assert sampler.on_frame(_frame(450_000_000, 80)) == ()
    assert sampler.on_frame(resumed_animation) == ()
    assert sampler.on_frame(final_position) == ()
    assert sampler.on_frame(_frame(999_999_999, 160)) == ()

    settled = _frame(1_000_000_000, 160)
    assert sampler.on_frame(settled) == (settled, settled, settled)


def test_default_quiet_window_outlasts_a_short_mid_animation_pause() -> None:
    sampler = AdaptiveBurstSampler(steady_fps=2, stable_repeats=2)
    animation_start = _frame(50_000_000, 40)
    mid_animation = _frame(100_000_000, 80)
    sampler.initialize(_frame(0))

    assert sampler.on_frame(animation_start) == (animation_start,)
    sampler.set_bursting(True)
    assert sampler.on_frame(mid_animation) == ()

    assert sampler.on_frame(_frame(300_000_000, 80)) == ()


def test_quiet_clock_promotes_one_buffered_steady_callback_as_a_stable_endpoint() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    changed = _frame(50_000_000, 10)
    sampler.initialize(_frame(0))

    assert sampler.on_frame(changed) == ()
    assert sampler.on_clock(149_999_999) == ()
    assert sampler.on_clock(150_000_000) == (changed, changed, changed)


def test_significant_visual_change_is_emitted_before_the_steady_clock() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    baseline = _frame(0)
    changed = _frame(50_000_000, 40)
    sampler.initialize(baseline)

    assert sampler.on_frame(changed) == (changed,)


def test_steady_clock_keeps_the_two_fps_logic_without_duplicate_early_ticks() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    baseline = _frame(1_000_000_000)
    sampler.initialize(baseline)

    assert sampler.on_clock(1_499_999_999) == ()
    assert sampler.on_clock(1_500_000_000) == (baseline,)
    assert sampler.on_clock(1_500_000_001) == ()


def test_continuous_callbacks_do_not_reset_or_bypass_the_steady_two_fps_clock() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    sampler.initialize(_frame(0))

    for timestamp_ns in range(50_000_000, 500_000_000, 50_000_000):
        assert sampler.on_frame(_frame(timestamp_ns, 10)) == ()

    assert sampler.on_clock(499_999_999) == ()
    latest = sampler.on_clock(500_000_000)
    assert len(latest) == 1
    assert latest[0].timestamp_ns == 450_000_000


def test_leaving_burst_returns_to_the_steady_clock() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    changed = _frame(50_000_000, 10)
    sampler.initialize(_frame(0))
    sampler.set_bursting(True)
    sampler.on_frame(changed)
    sampler.set_bursting(False)

    assert sampler.on_clock(549_999_999) == ()
    assert sampler.on_clock(550_000_000) == (changed,)


def test_non_monotonic_callback_timestamp_is_rejected() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    sampler.initialize(_frame(10))

    with pytest.raises(ValueError, match="monotonic"):
        sampler.on_frame(_frame(9))


def test_frame_size_change_is_rejected_before_buffered_samples_are_emitted() -> None:
    sampler = AdaptiveBurstSampler(
        steady_fps=2,
        settle_ms=100,
        stable_repeats=2,
    )
    sampler.initialize(_frame(0))
    sampler.on_frame(_frame(50_000_000, 10))

    resized = _frame(170_000_000, 20, shape=(3, 2))
    with pytest.raises(FrameSizeChangedError, match="size changed") as caught:
        sampler.on_frame(resized)

    assert caught.value.frame is resized


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"steady_fps": 0}, "steady_fps"),
        ({"settle_ms": 0}, "settle_ms"),
        ({"stable_repeats": 0}, "stable_repeats"),
    ],
)
def test_sampling_policy_rejects_non_positive_values(
    kwargs: dict[str, int],
    message: str,
) -> None:
    defaults = {"steady_fps": 2, "settle_ms": 100, "stable_repeats": 2}
    defaults.update(kwargs)

    with pytest.raises(ValueError, match=message):
        AdaptiveBurstSampler(**defaults)
