import cv2
import numpy as np
import pytest

from xiangqi_agent.domain.board import Orientation
from xiangqi_agent.vision.geometry import (
    BoardGeometry,
    GeometryError,
    NormalizedQuad,
    parse_normalized_quad,
)


def test_parse_normalized_quad_accepts_cli_order_and_rejects_bad_input() -> None:
    quad = parse_normalized_quad("0.1,0.1;0.9,0.1;0.9,0.9;0.1,0.9")
    assert quad.points == ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))
    with pytest.raises(GeometryError, match="four x,y"):
        parse_normalized_quad("0.1,0.1;0.9,0.1")


def test_normalized_quad_from_pixels_produces_stable_ninety_point_order() -> None:
    quad = NormalizedQuad.from_pixels(
        ((100, 100), (900, 100), (900, 900), (100, 900)),
        frame_size=(1001, 1001),
    )
    geometry = BoardGeometry.from_quad(quad, frame_size=(1001, 1001))

    points = geometry.grid_points()

    assert len(points) == 90
    assert points[0] == pytest.approx((100.0, 100.0))
    assert points[8] == pytest.approx((900.0, 100.0))
    assert points[81] == pytest.approx((100.0, 900.0))
    assert points[89] == pytest.approx((900.0, 900.0))


def test_black_bottom_orientation_maps_internal_first_point_to_screen_bottom_right() -> None:
    quad = NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
    geometry = BoardGeometry.from_quad(
        quad,
        frame_size=(1001, 1001),
        orientation=Orientation.BLACK_BOTTOM,
    )

    assert geometry.grid_points()[0] == pytest.approx((900.0, 900.0))
    assert geometry.grid_points()[89] == pytest.approx((100.0, 100.0))


def test_crop_intersections_returns_owned_ordered_patches_and_rejects_resize() -> None:
    width, height, patch_size = 73, 81, 8
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[..., 3] = 255
    pixel_quad = ((4, 4), (68, 4), (68, 76), (4, 76))
    for index, (x, y) in enumerate(
        BoardGeometry.from_quad(
            NormalizedQuad.from_pixels(pixel_quad, (width, height)),
            (width, height),
        ).grid_points()
    ):
        cv2.circle(frame, (round(x), round(y)), 2, (index + 1, 0, 0, 255), thickness=-1)
    geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(pixel_quad, (width, height)),
        (width, height),
    )

    patches = geometry.crop_intersections(frame, size=patch_size)

    assert len(patches) == 90
    assert all(patch.shape == (patch_size, patch_size, 4) for patch in patches)
    assert all(patch.flags["OWNDATA"] for patch in patches)
    assert int(patches[0][patch_size // 2, patch_size // 2, 0]) == 1
    assert int(patches[89][patch_size // 2, patch_size // 2, 0]) == 90
    with pytest.raises(GeometryError, match="size changed"):
        geometry.crop_intersections(np.zeros((height + 1, width, 4), dtype=np.uint8), size=8)


def test_crop_selected_intersections_materializes_only_requested_stable_indices() -> None:
    width, height, patch_size = 73, 81, 8
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[..., 3] = 255
    pixel_quad = ((4, 4), (68, 4), (68, 76), (4, 76))
    geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(pixel_quad, (width, height)),
        (width, height),
    )
    for index, (x, y) in enumerate(geometry.grid_points()):
        cv2.circle(frame, (round(x), round(y)), 2, (index + 1, 0, 0, 255), thickness=-1)

    selected = geometry.crop_selected_intersections(
        frame,
        (4, 22, 67, 81),
        size=patch_size,
    )

    assert len(selected) == 4
    assert [int(patch[patch_size // 2, patch_size // 2, 0]) for patch in selected] == [
        5,
        23,
        68,
        82,
    ]
    assert all(patch.flags["OWNDATA"] for patch in selected)


def test_crop_selected_intersections_preserves_logical_order_when_black_is_bottom() -> None:
    width, height, patch_size = 73, 81, 8
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[..., 3] = 255
    pixel_quad = ((4, 4), (68, 4), (68, 76), (4, 76))
    red_geometry = BoardGeometry.from_quad(
        NormalizedQuad.from_pixels(pixel_quad, (width, height)),
        (width, height),
    )
    for index, (x, y) in enumerate(red_geometry.grid_points()):
        cv2.circle(frame, (round(x), round(y)), 2, (index + 1, 0, 0, 255), thickness=-1)
    black_geometry = BoardGeometry.from_quad(
        red_geometry.quad,
        red_geometry.frame_size,
        Orientation.BLACK_BOTTOM,
    )

    selected = black_geometry.crop_selected_intersections(frame, (0, 89), size=patch_size)

    assert int(selected[0][patch_size // 2, patch_size // 2, 0]) == 90
    assert int(selected[1][patch_size // 2, patch_size // 2, 0]) == 1


def test_geometry_rebinds_normalized_quad_when_frame_aspect_ratio_is_stable() -> None:
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        frame_size=(300, 200),
    )

    rebound = geometry.rebind((450, 300))

    assert rebound.frame_size == (450, 300)
    assert rebound.quad == geometry.quad
    assert rebound.orientation is geometry.orientation
    assert len(rebound.grid_points()) == 90


def test_geometry_rebind_rejects_a_material_frame_aspect_ratio_change() -> None:
    geometry = BoardGeometry.from_quad(
        NormalizedQuad(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))),
        frame_size=(300, 200),
    )

    with pytest.raises(GeometryError, match="aspect ratio"):
        geometry.rebind((450, 200))


@pytest.mark.parametrize(
    "points",
    (
        ((0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)),
        ((-0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)),
        ((0.1, 0.1), (0.2, 0.1), (0.2, 0.11), (0.1, 0.11)),
    ),
)
def test_normalized_quad_rejects_bow_tie_out_of_bounds_and_tiny_shapes(
    points: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises(GeometryError):
        NormalizedQuad(points)
