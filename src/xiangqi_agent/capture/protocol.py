from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type UInt8Image = NDArray[np.uint8]
type FrameCallback = Callable[["CaptureFrame"], None]
type ClosedCallback = Callable[["CaptureClosedError"], None]


class CaptureClosedError(RuntimeError):
    """The selected window or capture session is no longer available."""


@dataclass(frozen=True, slots=True)
class CaptureFrame:
    timestamp_ns: int
    hwnd: int
    bgra: UInt8Image

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer")
        if isinstance(self.hwnd, bool) or not isinstance(self.hwnd, int) or self.hwnd <= 0:
            raise ValueError("hwnd must be a positive integer")
        pixels = np.asarray(self.bgra)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("frame must be a BGRA uint8 image")
        owned = np.array(pixels, dtype=np.uint8, copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "bgra", owned)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.bgra.shape[1]), int(self.bgra.shape[0])


class FrameSource(Protocol):
    def start(
        self,
        on_frame: FrameCallback,
        on_closed: ClosedCallback | None = None,
    ) -> None: ...

    def close(self) -> None: ...
