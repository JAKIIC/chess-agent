from typing import cast

import numpy as np
from numpy.typing import NDArray

from xiangqi_agent.capture.protocol import (
    CaptureClosedError,
    CaptureFrame,
    ClosedCallback,
    FrameCallback,
)


class FakeFrameSource:
    def __init__(self, hwnd: int) -> None:
        if isinstance(hwnd, bool) or not isinstance(hwnd, int) or hwnd <= 0:
            raise ValueError("hwnd must be a positive integer")
        self._hwnd = hwnd
        self._on_frame: FrameCallback | None = None
        self._on_closed: ClosedCallback | None = None
        self._started = False
        self._closed = False
        self._close_notified = False

    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None:
        if self._started:
            raise RuntimeError("frame source has already started")
        if self._closed:
            raise CaptureClosedError("frame source is closed")
        self._on_frame = on_frame
        self._on_closed = on_closed
        self._started = True

    def push(self, bgra: NDArray[np.generic], timestamp_ns: int) -> None:
        if self._closed:
            raise CaptureClosedError("frame source is closed")
        if not self._started or self._on_frame is None:
            raise RuntimeError("frame source has not started")
        self._on_frame(
            CaptureFrame(
                timestamp_ns=timestamp_ns,
                hwnd=self._hwnd,
                bgra=cast(NDArray[np.uint8], bgra),
            )
        )

    def simulate_target_close(self) -> None:
        if self._close_notified:
            return
        self._closed = True
        self._close_notified = True
        if self._on_closed is not None:
            self._on_closed(CaptureClosedError("target window closed"))

    def close(self) -> None:
        self._closed = True
