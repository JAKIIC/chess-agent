from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Identity of the visual context in which a frame may be settled."""

    wgc_size: tuple[int, int]
    client_size: tuple[int, int]
    dpi_scale: float
    geometry_revision: str
    theme_fingerprint: str
    generation_id: int

    def __post_init__(self) -> None:
        _validate_size(self.wgc_size, "wgc_size")
        _validate_size(self.client_size, "client_size")
        if not isfinite(self.dpi_scale) or self.dpi_scale <= 0:
            raise ValueError("dpi_scale must be finite and positive")
        if not isinstance(self.geometry_revision, str) or not self.geometry_revision.strip():
            raise ValueError("geometry_revision must be a non-empty string")
        if not isinstance(self.theme_fingerprint, str) or not self.theme_fingerprint.strip():
            raise ValueError("theme_fingerprint must be a non-empty string")
        if (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 0
        ):
            raise ValueError("generation_id must be a non-negative integer")

    def compatible_with(self, other: CaptureContext) -> bool:
        return isinstance(other, CaptureContext) and self == other


def _validate_size(size: tuple[int, int], name: str) -> None:
    if (
        not isinstance(size, tuple)
        or len(size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size)
    ):
        raise ValueError(f"{name} must contain two positive integers")
