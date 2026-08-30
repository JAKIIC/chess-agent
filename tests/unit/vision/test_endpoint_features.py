from __future__ import annotations

import cv2
import numpy as np
import pytest

from xiangqi_agent.diagnostics.endpoint_samples import EndpointCrops
from xiangqi_agent.vision.endpoint_features import (
    AlignedGradientExtractor,
    InstanceTransferExtractor,
    MaskedLabExtractor,
    RgbBaselineExtractor,
)


def _empty(value: int) -> np.ndarray:
    patch = np.full((48, 48, 4), value, dtype=np.uint8)
    patch[..., 3] = 255
    return patch


def _piece(background: int, *, glyph: str = "vertical") -> np.ndarray:
    patch = _empty(background)
    cv2.circle(patch, (24, 24), 16, (80, 180, 240, 255), -1, lineType=cv2.LINE_AA)
    if glyph == "vertical":
        cv2.line(patch, (24, 14), (24, 34), (20, 30, 40, 255), 3, cv2.LINE_AA)
    else:
        cv2.line(patch, (14, 24), (34, 24), (20, 30, 40, 255), 3, cv2.LINE_AA)
    return patch


def _translate(patch: np.ndarray, dx: int, dy: int, border: int) -> np.ndarray:
    return cv2.warpAffine(
        patch,
        np.float32(((1, 0, dx), (0, 1, dy))),
        (48, 48),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border, border, border, 255),
    )


def _crops(*, target_after: np.ndarray | None = None) -> EndpointCrops:
    return EndpointCrops(
        source_before=_piece(30),
        source_after=_empty(30),
        target_before=_empty(190),
        target_after=_piece(190) if target_after is None else target_after,
    )


def test_masked_lab_reduces_unrelated_outer_board_background_error() -> None:
    crops = _crops()

    rgb = RgbBaselineExtractor().extract(crops)
    masked = MaskedLabExtractor().extract(crops)

    assert rgb.instance_distance > 0.15
    assert masked.instance_distance < rgb.instance_distance * 0.6


def test_aligned_gradient_recovers_a_small_endpoint_translation() -> None:
    translated = _translate(_piece(190), dx=2, dy=-1, border=190)
    crops = _crops(target_after=translated)

    unaligned = MaskedLabExtractor().extract(crops)
    aligned = AlignedGradientExtractor(max_shift=3).extract(crops)

    assert aligned.best_shift != (0, 0)
    assert max(abs(value) for value in aligned.best_shift) <= 3
    assert aligned.instance_distance < unaligned.instance_distance


def test_instance_transfer_distinguishes_a_different_piece_glyph() -> None:
    extractor = InstanceTransferExtractor(max_shift=3)

    same = extractor.extract(_crops())
    different = extractor.extract(_crops(target_after=_piece(190, glyph="horizontal")))

    assert same.instance_distance < different.instance_distance
    assert same.instance_evidence_score > different.instance_evidence_score


def test_instance_transfer_is_deterministic_for_the_same_four_crops() -> None:
    extractor = InstanceTransferExtractor(max_shift=3)
    crops = _crops()

    assert extractor.extract(crops) == extractor.extract(crops)


def test_endpoint_extractors_report_versioned_non_probability_evidence() -> None:
    features = InstanceTransferExtractor().extract(_crops())

    assert features.feature_version == "instance-transfer-v1"
    assert 0.0 <= features.instance_evidence_score <= 1.0
    assert features.source_change_distance > 0.0
    assert features.target_change_distance > 0.0


def test_endpoint_extractor_rejects_non_bgra_or_wrong_sized_crops() -> None:
    invalid = EndpointCrops(
        source_before=np.zeros((10, 10, 3), dtype=np.uint8),
        source_after=_empty(20),
        target_before=_empty(20),
        target_after=_piece(20),
    )

    with pytest.raises(ValueError, match="48x48 BGRA"):
        RgbBaselineExtractor().extract(invalid)
