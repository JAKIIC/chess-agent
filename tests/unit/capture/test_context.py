import pytest

from xiangqi_agent.capture.context import CaptureContext


def _context(**changes: object) -> CaptureContext:
    values: dict[str, object] = {
        "wgc_size": (200, 300),
        "client_size": (200, 300),
        "dpi_scale": 1.25,
        "geometry_revision": "geometry-1",
        "theme_fingerprint": "theme-1",
        "generation_id": 7,
    }
    values.update(changes)
    return CaptureContext(**values)  # type: ignore[arg-type]


def test_identical_capture_contexts_are_compatible() -> None:
    assert _context().compatible_with(_context())


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("wgc_size", (201, 300)),
        ("client_size", (200, 301)),
        ("dpi_scale", 1.5),
        ("geometry_revision", "geometry-2"),
        ("theme_fingerprint", "theme-2"),
        ("generation_id", 8),
    ],
)
def test_any_capture_generation_or_visual_context_change_is_incompatible(
    field: str,
    changed: object,
) -> None:
    assert not _context().compatible_with(_context(**{field: changed}))


@pytest.mark.parametrize(
    "changes",
    [
        {"wgc_size": (0, 300)},
        {"client_size": (200, -1)},
        {"dpi_scale": 0.0},
        {"geometry_revision": ""},
        {"theme_fingerprint": ""},
        {"generation_id": -1},
    ],
)
def test_capture_context_rejects_invalid_metadata(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _context(**changes)
