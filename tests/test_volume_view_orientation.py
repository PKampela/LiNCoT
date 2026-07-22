"""Tests for MRI viewer anatomical axis mapping derived from image affine."""

from __future__ import annotations

import numpy as np

from core.frames import CoordinateFrame
from core.image import Image
from gui.viewer.render_utils import anatomical_axis_info


def _test_image(affine: np.ndarray) -> Image:
    frame = CoordinateFrame("mri", ("R", "A", "S"), "mm")
    data = np.zeros((6, 7, 8), dtype=float)
    return Image(data=data, affine=affine, frame=frame)


def test_anatomical_axis_info_handles_permuted_axes() -> None:
    """Voxel axis permutations should still map to LR/AP/SI groups correctly."""
    affine = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    image = _test_image(affine)
    mapping = anatomical_axis_info(image)

    assert mapping["lr"].voxel_axis == 1
    assert mapping["ap"].voxel_axis == 2
    assert mapping["si"].voxel_axis == 0
    assert mapping["lr"].sign == 1
    assert mapping["ap"].sign == 1
    assert mapping["si"].sign == 1


def test_anatomical_axis_info_handles_flipped_axes() -> None:
    """Axis direction flips should be represented in the mapping sign."""
    affine = np.diag([-1.0, -1.0, -1.0, 1.0])
    image = _test_image(affine)
    mapping = anatomical_axis_info(image)

    assert mapping["lr"].voxel_axis == 0
    assert mapping["ap"].voxel_axis == 1
    assert mapping["si"].voxel_axis == 2
    assert mapping["lr"].sign == -1
    assert mapping["ap"].sign == -1
    assert mapping["si"].sign == -1