from collections.abc import Sequence

import numpy as np
import pytest

from xiangqi_agent.capture.fake import FakeFrameSource
from xiangqi_agent.capture.protocol import CaptureClosedError, CaptureFrame


def test_fake_source_emits_an_owned_bgra_frame() -> None:
    received: list[CaptureFrame] = []
    source = FakeFrameSource(hwnd=9)
    source.start(received.append)
    pixels = np.zeros((20, 30, 4), dtype=np.uint8)

    source.push(pixels, timestamp_ns=123)
    pixels.fill(255)

    assert len(received) == 1
    assert received[0].hwnd == 9
    assert received[0].timestamp_ns == 123
    assert received[0].bgra.shape == (20, 30, 4)
    assert received[0].bgra.flags["OWNDATA"]
    assert np.count_nonzero(received[0].bgra) == 0


def test_fake_source_reports_target_close_once_and_close_is_idempotent() -> None:
    errors: list[CaptureClosedError] = []
    source = FakeFrameSource(hwnd=9)
    source.start(lambda _frame: None, errors.append)

    source.simulate_target_close()
    source.simulate_target_close()
    source.close()
    source.close()

    assert len(errors) == 1
    assert "closed" in str(errors[0]).lower()
    with pytest.raises(CaptureClosedError, match="closed"):
        source.push(np.zeros((2, 2, 4), dtype=np.uint8), timestamp_ns=200)


@pytest.mark.parametrize(
    "pixels",
    (
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.float32),
    ),
)
def test_capture_frame_rejects_non_bgra_uint8(pixels: Sequence[object]) -> None:
    with pytest.raises(ValueError, match="BGRA uint8"):
        CaptureFrame(timestamp_ns=1, hwnd=1, bgra=np.asarray(pixels))
