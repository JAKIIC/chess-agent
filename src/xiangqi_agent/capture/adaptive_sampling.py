from __future__ import annotations

import numpy as np

from xiangqi_agent.capture.protocol import CaptureFrame

_NANOSECONDS_PER_SECOND = 1_000_000_000
_NANOSECONDS_PER_MILLISECOND = 1_000_000
_VISUAL_TRIGGER_THRESHOLD = 12
_VISUAL_TRIGGER_STRIDE = 8


class FrameSizeChangedError(ValueError):
    """A capture callback changed size and must be explicitly rebound."""

    def __init__(self, frame: CaptureFrame) -> None:
        super().__init__("capture frame size changed during sampling")
        self.frame = frame


class AdaptiveBurstSampler:
    """Preserve quiet burst endpoints while keeping a low-rate steady clock."""

    def __init__(
        self,
        *,
        steady_fps: int = 2,
        settle_ms: int = 100,
        stable_repeats: int = 2,
    ) -> None:
        self._steady_fps = _positive_integer("steady_fps", steady_fps)
        self._settle_ms = _positive_integer("settle_ms", settle_ms)
        self._stable_repeats = _positive_integer("stable_repeats", stable_repeats)
        self._steady_interval_ns = round(_NANOSECONDS_PER_SECOND / self._steady_fps)
        self._settle_ns = self._settle_ms * _NANOSECONDS_PER_MILLISECOND
        self._latest: CaptureFrame | None = None
        self._next_steady_due_ns: int | None = None
        self._bursting = False
        self._settled_latest = False
        self._latest_emitted = False
        self._last_emitted: CaptureFrame | None = None

    @property
    def bursting(self) -> bool:
        return self._bursting

    def initialize(self, frame: CaptureFrame) -> None:
        self._latest = frame
        self._next_steady_due_ns = frame.timestamp_ns + self._steady_interval_ns
        self._bursting = False
        self._settled_latest = False
        self._latest_emitted = True
        self._last_emitted = frame

    def set_bursting(self, active: bool) -> None:
        if active == self._bursting:
            return
        self._bursting = active
        self._settled_latest = False
        if not active and self._latest is not None:
            self._next_steady_due_ns = self._latest.timestamp_ns + self._steady_interval_ns

    def on_frame(self, frame: CaptureFrame) -> tuple[CaptureFrame, ...]:
        latest = self._require_latest()
        if frame.timestamp_ns <= latest.timestamp_ns:
            raise ValueError("capture frame timestamps must be strictly monotonic")
        if frame.hwnd != latest.hwnd:
            raise ValueError("capture frame target changed during sampling")
        if frame.size != latest.size:
            raise FrameSizeChangedError(frame)

        samples: tuple[CaptureFrame, ...] = ()
        quiet_gap = frame.timestamp_ns - latest.timestamp_ns >= self._settle_ns
        if self._bursting:
            if quiet_gap and not self._settled_latest:
                samples = (latest,) * self._stable_repeats
            samples += (frame,)
        elif quiet_gap and not self._latest_emitted:
            samples = (latest,) * (self._stable_repeats + 1) + (frame,)
        elif self._visual_change_exceeds_trigger(frame):
            samples = (frame,)

        self._latest = frame
        if samples:
            self._next_steady_due_ns = frame.timestamp_ns + self._steady_interval_ns
            self._last_emitted = samples[-1]
        self._settled_latest = False
        self._latest_emitted = bool(samples)
        return samples

    def on_clock(self, timestamp_ns: int) -> tuple[CaptureFrame, ...]:
        latest = self._require_latest()
        if timestamp_ns < latest.timestamp_ns:
            raise ValueError("clock timestamp must not precede the latest capture frame")

        if self._bursting:
            if (
                not self._settled_latest
                and timestamp_ns - latest.timestamp_ns >= self._settle_ns
            ):
                self._settled_latest = True
                self._latest_emitted = True
                self._last_emitted = latest
                return (latest,) * self._stable_repeats
            return ()

        if not self._latest_emitted and timestamp_ns - latest.timestamp_ns >= self._settle_ns:
            self._settled_latest = True
            self._latest_emitted = True
            self._next_steady_due_ns = timestamp_ns + self._steady_interval_ns
            self._last_emitted = latest
            return (latest,) * (self._stable_repeats + 1)

        due = self._next_steady_due_ns
        if due is not None and timestamp_ns >= due:
            self._next_steady_due_ns = timestamp_ns + self._steady_interval_ns
            self._latest_emitted = True
            self._last_emitted = latest
            return (latest,)
        return ()

    def _visual_change_exceeds_trigger(self, frame: CaptureFrame) -> bool:
        reference = self._last_emitted
        if reference is None:
            return False
        before = reference.bgra[::_VISUAL_TRIGGER_STRIDE, ::_VISUAL_TRIGGER_STRIDE, :3]
        after = frame.bgra[::_VISUAL_TRIGGER_STRIDE, ::_VISUAL_TRIGGER_STRIDE, :3]
        difference = np.abs(before.astype(np.int16) - after.astype(np.int16))
        return bool(difference.max(initial=0) >= _VISUAL_TRIGGER_THRESHOLD)

    def next_due_ns(self) -> int | None:
        latest = self._require_latest()
        if self._bursting:
            return None if self._settled_latest else latest.timestamp_ns + self._settle_ns
        due = self._next_steady_due_ns
        if not self._latest_emitted:
            settle_due = latest.timestamp_ns + self._settle_ns
            return settle_due if due is None else min(due, settle_due)
        return due

    def _require_latest(self) -> CaptureFrame:
        if self._latest is None:
            raise RuntimeError("sampler must be initialized with a capture frame")
        return self._latest


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
